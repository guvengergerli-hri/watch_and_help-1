import argparse
import datetime
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from watch.vae.dataset import WatchVAEDataset, collate_watch_vae
from watch.vae.model import GraphSequenceVAE
from watch.vae.tensorizer import WatchGraphTensorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a predicate-vector label head on top of a frozen Watch Graph VAE encoder."
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to stage-1 Watch VAE checkpoint (watch_vae_best.pt or watch_vae_last.pt).",
    )

    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--train-json", type=str, default=None)
    parser.add_argument("--val-json", type=str, default=None)
    parser.add_argument("--train-split-key", type=str, default=None)
    parser.add_argument("--val-split-key", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default="checkpoints/watch_vae_predicate_vectors")
    parser.add_argument("--tensorboard-logdir", type=str, default="log_tb/watch_vae_predicate_vectors")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--disable-tensorboard", action="store_true", default=False)

    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--max-train-demos", type=int, default=None)
    parser.add_argument("--max-val-demos", type=int, default=None)
    parser.add_argument("--stable-slots", action="store_true", default=True)
    parser.add_argument("--no-stable-slots", action="store_false", dest="stable_slots")

    parser.add_argument(
        "--labeled-fraction",
        type=float,
        default=1.0,
        help="Fraction of training samples to expose label supervision for (1.0 = fully labeled).",
    )
    parser.add_argument(
        "--label-mask-prob",
        type=float,
        default=None,
        help="Deprecated alias for (1 - labeled_fraction). If set, overrides labeled_fraction.",
    )
    parser.add_argument(
        "--label-mask-mode",
        type=str,
        default="fixed",
        choices=("fixed", "stochastic"),
        help="How to choose labeled samples when labeled_fraction < 1.0.",
    )
    parser.add_argument(
        "--label-mask-seed",
        type=int,
        default=None,
        help="Seed for fixed label subset selection (defaults to --seed).",
    )

    parser.add_argument("--save-merged-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-save-merged-checkpoint", action="store_false", dest="save_merged_checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--detect-anomaly", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    try:
        device = torch.device(device_arg)
    except Exception as exc:
        raise ValueError("Invalid --device '{}': {}".format(device_arg, exc)) from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device '{}' was requested, but torch.cuda.is_available() is False. "
                "Install CUDA-enabled PyTorch or pass --device cpu.".format(device_arg)
            )
        if device.index is not None and (device.index < 0 or device.index >= torch.cuda.device_count()):
            raise RuntimeError(
                "CUDA device index out of range for '{}'. Visible CUDA device count: {}.".format(
                    device_arg,
                    torch.cuda.device_count(),
                )
            )
    return device


def print_device_summary(device: torch.device) -> None:
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
    print("device:", device)
    if device.type == "cuda":
        idx = torch.cuda.current_device() if device.index is None else int(device.index)
        print("using cuda device index:", idx)
        print("using cuda device name:", torch.cuda.get_device_name(idx))


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def infer_paths_from_checkpoint(
    args: argparse.Namespace,
    checkpoint: Dict[str, object],
) -> Dict[str, str]:
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    metadata = args.metadata or ckpt_args.get("metadata")
    train_json = args.train_json or ckpt_args.get("train_json")
    val_json = args.val_json or ckpt_args.get("val_json") or train_json
    train_split_key = args.train_split_key or ckpt_args.get("train_split_key", "train_data")
    val_split_key = args.val_split_key or ckpt_args.get("val_split_key", "test_data")
    missing = [
        key
        for key, value in {
            "metadata": metadata,
            "train_json": train_json,
            "val_json": val_json,
            "train_split_key": train_split_key,
            "val_split_key": val_split_key,
        }.items()
        if value is None
    ]
    if len(missing) > 0:
        raise ValueError(
            "Could not infer required data paths from VAE checkpoint args. Missing: {}. "
            "Provide them explicitly.".format(", ".join(missing))
        )
    return {
        "metadata": str(metadata),
        "train_json": str(train_json),
        "val_json": str(val_json),
        "train_split_key": str(train_split_key),
        "val_split_key": str(val_split_key),
    }


def load_goal_predicates(metadata_path: str) -> Tuple[List[str], Dict[str, int]]:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    raw = metadata.get("goal_predicates", {})
    if not raw:
        raise KeyError("metadata missing goal_predicates")
    max_idx = max(int(v) for v in raw.values())
    names_by_idx = ["<PAD>"] * (max_idx + 1)
    for name, idx in raw.items():
        idx_i = int(idx)
        if 0 <= idx_i < len(names_by_idx):
            names_by_idx[idx_i] = str(name)
    names = [names_by_idx[i] for i in range(1, len(names_by_idx)) if names_by_idx[i] != "<PAD>"]
    return names, {name: i for i, name in enumerate(names)}


def encode_goal_multihot(goals_batch: List[List[str]], goal_to_col: Dict[str, int]) -> Tuple[np.ndarray, int]:
    y = np.zeros((len(goals_batch), len(goal_to_col)), dtype=np.float32)
    unknown = 0
    for i, goals in enumerate(goals_batch):
        for goal in goals:
            name = str(goal)
            col = goal_to_col.get(name)
            if col is None:
                unknown += 1
                continue
            y[i, col] = 1.0
    return y, unknown


def attach_goal_labels(
    batch: Dict[str, object],
    goal_to_col: Dict[str, int],
    device: torch.device,
    train: bool,
    label_mask_prob: float,
    label_mask_mode: str = "stochastic",
    fixed_label_keep_mask: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, object], int]:
    goals_batch = batch.get("goals", [])
    goal_labels_np, unknown = encode_goal_multihot(goals_batch, goal_to_col)
    goal_labels = torch.from_numpy(goal_labels_np).float().to(device)
    batch["goal_labels"] = goal_labels

    bsz = int(goal_labels.shape[0])
    if train and label_mask_prob > 0.0:
        if label_mask_mode == "fixed":
            if fixed_label_keep_mask is None:
                raise ValueError("fixed label mask mode requires fixed_label_keep_mask")
            if "dataset_indices" not in batch:
                raise KeyError("batch is missing dataset_indices required for fixed label masking")
            dataset_indices = batch["dataset_indices"]
            if not torch.is_tensor(dataset_indices):
                raise TypeError("batch['dataset_indices'] must be a tensor")
            dataset_indices = dataset_indices.to(device=device, dtype=torch.long)
            if dataset_indices.numel() != bsz:
                raise ValueError("dataset_indices shape mismatch for fixed label masking")
            keep = fixed_label_keep_mask.index_select(0, dataset_indices).float()
        else:
            keep = (torch.rand((bsz,), device=device) > float(label_mask_prob)).float()
    else:
        keep = torch.ones((bsz,), device=device)

    batch["goal_label_mask"] = keep
    return batch, unknown


def resolve_label_fraction(args: argparse.Namespace) -> None:
    if args.label_mask_prob is not None:
        if not (0.0 <= float(args.label_mask_prob) <= 1.0):
            raise ValueError("--label-mask-prob must be in [0, 1]")
        if abs(float(args.labeled_fraction) - 1.0) > 1e-9:
            print(
                "Warning: both --labeled-fraction and --label-mask-prob were set; "
                "using --label-mask-prob (deprecated) precedence."
            )
        effective_mask_prob = float(args.label_mask_prob)
        effective_labeled_fraction = 1.0 - effective_mask_prob
    else:
        if not (0.0 <= float(args.labeled_fraction) <= 1.0):
            raise ValueError("--labeled-fraction must be in [0, 1]")
        effective_labeled_fraction = float(args.labeled_fraction)
        effective_mask_prob = 1.0 - effective_labeled_fraction

    args.effective_label_mask_prob = effective_mask_prob
    args.effective_labeled_fraction = effective_labeled_fraction


def build_fixed_label_keep_mask(
    dataset_size: int,
    labeled_fraction: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if dataset_size <= 0:
        return torch.zeros((0,), dtype=torch.float32, device=device)
    lf = float(labeled_fraction)
    if not (0.0 <= lf <= 1.0):
        raise ValueError("labeled_fraction must be in [0, 1]")
    num_keep = int(round(lf * dataset_size))
    num_keep = max(0, min(dataset_size, num_keep))
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(dataset_size)
    keep_idx = perm[:num_keep]
    mask = torch.zeros((dataset_size,), dtype=torch.float32, device=device)
    if num_keep > 0:
        mask[torch.as_tensor(keep_idx, dtype=torch.long, device=device)] = 1.0
    return mask


def build_experiment_slug(args: argparse.Namespace, ckpt_name: str) -> str:
    raw = (
        "vaehead_ckpt{ckpt}_lr{lr}_bs{bs}_lf{lf}_mode{mode}_ss{ss}"
    ).format(
        ckpt=ckpt_name,
        lr=args.lr,
        bs=args.batch_size,
        lf=getattr(args, "effective_labeled_fraction", args.labeled_fraction),
        mode=args.label_mask_mode,
        ss=int(args.stable_slots),
    )
    raw = raw.replace(".", "p").replace("+", "").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_-]+", "", raw)


def build_experiment_title(args: argparse.Namespace, ckpt_name: str) -> str:
    return (
        "watch_vae_predicate_vectors | ckpt={ckpt} lr={lr} bs={bs} labeled={lf} mask_mode={mode}"
    ).format(
        ckpt=ckpt_name,
        lr=args.lr,
        bs=args.batch_size,
        lf=getattr(args, "effective_labeled_fraction", args.labeled_fraction),
        mode=args.label_mask_mode,
    )


def freeze_backbone_keep_head(model: GraphSequenceVAE) -> Dict[str, int]:
    for param in model.parameters():
        param.requires_grad = False
    head_params = model.predicate_head_parameters()
    if len(head_params) == 0:
        raise RuntimeError("Predicate vector head is not enabled on model")
    for param in head_params:
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable_params": int(trainable), "frozen_params": int(frozen)}


def compute_goal_losses(
    logits: torch.Tensor,
    goal_labels: torch.Tensor,
    goal_label_mask: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if goal_label_mask is None:
        mask = torch.ones((logits.shape[0],), device=logits.device)
    else:
        mask = goal_label_mask
    mask = mask.float()

    bce_vec = F.binary_cross_entropy_with_logits(logits, goal_labels, reduction="none").mean(dim=-1)
    denom = mask.sum().clamp(min=1.0)
    goal_bce_loss = (bce_vec * mask).sum() / denom

    probs = torch.sigmoid(logits)
    labeled_mask_bool = mask > 0.5
    labeled_fraction = mask.mean()

    if labeled_mask_bool.any():
        pred = (probs[labeled_mask_bool] >= 0.5).float()
        tar = goal_labels[labeled_mask_bool]
        tp = (pred * tar).sum()
        fp = (pred * (1.0 - tar)).sum()
        fn = ((1.0 - pred) * tar).sum()
        precision = tp / (tp + fp).clamp(min=1.0)
        recall = tp / (tp + fn).clamp(min=1.0)
        f1 = (2.0 * precision * recall) / (precision + recall).clamp(min=1e-6)
        exact = (pred == tar).all(dim=-1).float().mean()
        target_pos_rate = tar.mean()
    else:
        zero = goal_bce_loss * 0.0
        precision = zero
        recall = zero
        f1 = zero
        exact = zero
        target_pos_rate = zero

    return {
        "loss": goal_bce_loss,
        "goal_bce_loss": goal_bce_loss,
        "goal_labeled_fraction": labeled_fraction,
        "goal_precision_0p5": precision,
        "goal_recall_0p5": recall,
        "goal_f1_0p5": f1,
        "goal_exact_match_0p5": exact,
        "goal_target_pos_rate": target_pos_rate,
        "goal_pred_pos_rate_0p5": (probs >= 0.5).float().mean(),
    }


def run_epoch(
    model: GraphSequenceVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    goal_to_col: Dict[str, int],
    device: torch.device,
    train: bool,
    log_interval: int,
    grad_clip: float,
    label_mask_prob: float,
    label_mask_mode: str = "stochastic",
    fixed_label_keep_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    running: Dict[str, float] = {}
    n_batches = 0
    total_unknown_goals = 0

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        batch, unknown = attach_goal_labels(
            batch=batch,
            goal_to_col=goal_to_col,
            device=device,
            train=train,
            label_mask_prob=label_mask_prob,
            label_mask_mode=label_mask_mode,
            fixed_label_keep_mask=fixed_label_keep_mask,
        )
        total_unknown_goals += int(unknown)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.no_grad():
                mu_last = model.extract_last_mu(batch)  # [B, latent_size]

            logits = model.predicate_logits_from_mu(mu_last)
            losses = compute_goal_losses(
                logits=logits,
                goal_labels=batch["goal_labels"],  # type: ignore[index]
                goal_label_mask=batch.get("goal_label_mask"),  # type: ignore[arg-type]
            )
            loss = losses["loss"]

            if not torch.isfinite(loss):
                diag = {
                    "batch_idx": batch_idx,
                    "train": train,
                    "loss_items": {k: float(v.detach().cpu().item()) for k, v in losses.items()},
                }
                raise RuntimeError("Non-finite loss in predicate-vector training:\n{}".format(json.dumps(diag, indent=2)))

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.predicate_head_parameters(), max_norm=grad_clip)
                optimizer.step()

        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu().item())
        n_batches += 1

        if train and (batch_idx + 1) % log_interval == 0:
            print(
                "train step {:04d}/{:04d} loss={:.4f} goal_f1={:.4f} labeled_frac={:.4f}".format(
                    batch_idx + 1,
                    len(loader),
                    float(losses["loss"].item()),
                    float(losses["goal_f1_0p5"].item()),
                    float(losses["goal_labeled_fraction"].item()),
                )
            )

    if n_batches == 0:
        raise RuntimeError("No batches produced. Check dataset paths/settings.")

    metrics = {key: value / n_batches for key, value in running.items()}
    metrics["unknown_goal_labels"] = float(total_unknown_goals)
    return metrics


def main() -> None:
    args = parse_args()
    resolve_label_fraction(args)
    set_seed(args.seed)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    device = resolve_device(args.device)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device.index)
        torch.backends.cudnn.benchmark = True
    print_device_summary(device)
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.disable_tensorboard:
        os.makedirs(args.tensorboard_logdir, exist_ok=True)

    source_ckpt = torch.load(args.vae_checkpoint, map_location="cpu")
    if not isinstance(source_ckpt, dict):
        raise TypeError("VAE checkpoint is not a dict: {}".format(type(source_ckpt).__name__))
    if "model_config" not in source_ckpt or "model_state" not in source_ckpt:
        raise KeyError("VAE checkpoint must contain model_config and model_state")
    if "tensorizer_config" not in source_ckpt:
        raise KeyError("VAE checkpoint must contain tensorizer_config")

    paths = infer_paths_from_checkpoint(args, source_ckpt)
    goal_predicate_names, goal_to_col = load_goal_predicates(paths["metadata"])

    tensorizer = WatchGraphTensorizer.from_config(source_ckpt["tensorizer_config"])
    use_actions = bool(source_ckpt["model_config"].get("use_actions", False))
    print("num goal predicates:", len(goal_predicate_names))
    print("effective labeled fraction:", args.effective_labeled_fraction)
    print("effective label mask prob:", args.effective_label_mask_prob)
    print("label mask mode:", args.label_mask_mode)

    train_dataset = WatchVAEDataset(
        data_path=paths["train_json"],
        split_key=paths["train_split_key"],
        tensorizer=tensorizer,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        use_actions=use_actions,
        max_demos=args.max_train_demos,
        stable_slots=args.stable_slots,
    )
    val_dataset = WatchVAEDataset(
        data_path=paths["val_json"],
        split_key=paths["val_split_key"],
        tensorizer=tensorizer,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        use_actions=use_actions,
        max_demos=args.max_val_demos,
        stable_slots=args.stable_slots,
    )
    print("train demos:", len(train_dataset))
    print("val demos:", len(val_dataset))

    fixed_label_keep_mask: Optional[torch.Tensor] = None
    if args.label_mask_mode == "fixed":
        label_mask_seed = int(args.seed if args.label_mask_seed is None else args.label_mask_seed)
        fixed_label_keep_mask = build_fixed_label_keep_mask(
            dataset_size=len(train_dataset),
            labeled_fraction=args.effective_labeled_fraction,
            seed=label_mask_seed,
            device=device,
        )
        actual_fixed_fraction = (
            float(fixed_label_keep_mask.mean().item()) if fixed_label_keep_mask.numel() > 0 else 0.0
        )
        print(
            "fixed label subset: seed={} keep={}/{} ({:.4f})".format(
                label_mask_seed,
                int(fixed_label_keep_mask.sum().item()),
                len(train_dataset),
                actual_fixed_fraction,
            )
        )
        args.fixed_label_subset_seed = label_mask_seed
        args.fixed_label_subset_actual_fraction = actual_fixed_fraction
    else:
        args.fixed_label_subset_seed = None
        args.fixed_label_subset_actual_fraction = None

    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_watch_vae,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_common["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_common,
    )

    model = GraphSequenceVAE.from_config(source_ckpt["model_config"]).to(device)
    model.load_state_dict(source_ckpt["model_state"], strict=True)
    if (not model.enable_predicate_head) or (model.num_goal_predicates <= 0):
        model.enable_predicate_vector_head(num_goal_predicates=len(goal_predicate_names))
    elif model.num_goal_predicates != len(goal_predicate_names):
        old_k = int(model.num_goal_predicates)
        model.enable_predicate_vector_head(num_goal_predicates=len(goal_predicate_names))
        print(
            "Warning: source checkpoint predicate head size ({}) differed from metadata goal count ({}); "
            "reinitialized predicate head.".format(old_k, len(goal_predicate_names))
        )
    else:
        print("Using existing predicate head from source checkpoint.")
    freeze_report = freeze_backbone_keep_head(model)
    print("freeze report:", json.dumps(freeze_report, indent=2))

    optimizer = torch.optim.Adam(
        model.predicate_head_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    vae_ckpt_name = os.path.splitext(os.path.basename(args.vae_checkpoint))[0]
    experiment_slug = build_experiment_slug(args, vae_ckpt_name)
    experiment_title = build_experiment_title(args, vae_ckpt_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}_{}".format(timestamp, experiment_slug)
    run_output_dir = os.path.join(args.output_dir, run_name)
    tb_run_dir = os.path.join(args.tensorboard_logdir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    print("run_name:", run_name)
    print("run_output_dir:", run_output_dir)
    if not args.disable_tensorboard:
        print("tensorboard_run_dir:", tb_run_dir)

    writer = None
    if not args.disable_tensorboard:
        writer = SummaryWriter(log_dir=tb_run_dir)
        writer.add_text("config/args", json.dumps(vars(args), indent=2), 0)
        writer.add_text("config/experiment_title", experiment_title, 0)
        writer.add_text("config/source_vae_checkpoint", args.vae_checkpoint, 0)
        writer.add_text("config/freeze_report", json.dumps(freeze_report, indent=2), 0)
        writer.add_text("config/goal_predicates_count", str(len(goal_predicate_names)), 0)
        writer.add_scalar("config/effective_labeled_fraction", args.effective_labeled_fraction, 0)
        writer.add_scalar("config/effective_label_mask_prob", args.effective_label_mask_prob, 0)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            goal_to_col=goal_to_col,
            device=device,
            train=True,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
            label_mask_prob=args.effective_label_mask_prob,
            label_mask_mode=args.label_mask_mode,
            fixed_label_keep_mask=fixed_label_keep_mask,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            goal_to_col=goal_to_col,
            device=device,
            train=False,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
            label_mask_prob=0.0,
            label_mask_mode="fixed",
            fixed_label_keep_mask=None,
        )

        print(
            "epoch {:03d} train_loss={:.4f} val_loss={:.4f} val_goal_bce={:.4f} val_goal_f1={:.4f} val_exact={:.4f}".format(
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["goal_bce_loss"],
                val_metrics["goal_f1_0p5"],
                val_metrics["goal_exact_match_0p5"],
            )
        )

        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar("train/{}".format(key), value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar("val/{}".format(key), value, epoch)
            writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], epoch)

        checkpoint = {
            "epoch": epoch,
            "source_vae_checkpoint": args.vae_checkpoint,
            "source_vae_model_config": source_ckpt["model_config"],
            "source_vae_args": source_ckpt.get("args", {}),
            "predicate_head_state": model.get_predicate_head_state_dict(),
            "goal_predicate_names": goal_predicate_names,
            "goal_to_col": goal_to_col,
            "tensorizer_config": tensorizer.to_config(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "run_name": run_name,
            "run_output_dir": run_output_dir,
            "tensorboard_run_dir": tb_run_dir if not args.disable_tensorboard else None,
            "freeze_report": freeze_report,
            "args": vars(args),
        }
        last_path = os.path.join(run_output_dir, "watch_vae_predicate_vectors_last.pt")
        torch.save(checkpoint, last_path)

        if args.save_merged_checkpoint:
            merged_checkpoint = {
                "epoch": epoch,
                "source_vae_checkpoint": args.vae_checkpoint,
                "model_state": model.state_dict(),
                "model_config": model.get_config(),
                "tensorizer_config": tensorizer.to_config(),
                "goal_predicate_names": goal_predicate_names,
                "goal_to_col": goal_to_col,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "run_name": run_name,
                "run_output_dir": run_output_dir,
                "tensorboard_run_dir": tb_run_dir if not args.disable_tensorboard else None,
                "freeze_report": freeze_report,
                "args": vars(args),
            }
            torch.save(merged_checkpoint, os.path.join(run_output_dir, "watch_vae_with_predicates_last.pt"))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = os.path.join(run_output_dir, "watch_vae_predicate_vectors_best.pt")
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, os.path.join(run_output_dir, "best_model.pt"))
            print("updated best model: val_loss={:.6f} -> {}".format(best_val, best_path))

            if args.save_merged_checkpoint:
                torch.save(merged_checkpoint, os.path.join(run_output_dir, "watch_vae_with_predicates_best.pt"))

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        with open(os.path.join(run_output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    with open(os.path.join(run_output_dir, "run_info.json"), "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "run_output_dir": run_output_dir,
                "tensorboard_run_dir": tb_run_dir if not args.disable_tensorboard else None,
                "experiment_title": experiment_title,
                "experiment_slug": experiment_slug,
                "source_vae_checkpoint": args.vae_checkpoint,
                "save_merged_checkpoint": args.save_merged_checkpoint,
                "args": vars(args),
            },
            f,
            indent=2,
        )

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
