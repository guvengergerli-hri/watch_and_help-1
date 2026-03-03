import argparse
import datetime
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from watch.vae.dataset import WatchVAEDataset, collate_watch_vae
from watch.vae.semisup_model import GraphSequenceSemiSupVAE
from watch.vae.tensorizer import WatchGraphTensorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a semi-supervised watch VAE with continuous z and multi-label Bernoulli y (goal predicates)."
    )

    parser.add_argument("--metadata", type=str, default="dataset/watch_data/metadata.json")
    parser.add_argument("--train-json", type=str, default="dataset/watch_data/gather_data_actiongraph_train.json")
    parser.add_argument("--val-json", type=str, default="dataset/watch_data/gather_data_actiongraph_test.json")
    parser.add_argument("--train-split-key", type=str, default="train_data")
    parser.add_argument("--val-split-key", type=str, default="test_data")

    parser.add_argument("--output-dir", type=str, default="checkpoints/watch_semisup_vae")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--transformer-nhead", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--max-train-demos", type=int, default=None)
    parser.add_argument("--max-val-demos", type=int, default=None)
    parser.add_argument("--stable-slots", action="store_true", default=True)
    parser.add_argument("--no-stable-slots", action="store_false", dest="stable_slots")

    parser.add_argument("--use-actions", action="store_true", default=False)
    parser.add_argument("--reconstruct-actions", action="store_true", dest="reconstruct_actions")
    parser.add_argument("--no-reconstruct-actions", action="store_false", dest="reconstruct_actions")
    parser.set_defaults(reconstruct_actions=None)
    parser.add_argument("--action-weight", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--mask-weight", type=float, default=0.2)
    parser.add_argument("--logvar-min", type=float, default=-10.0)
    parser.add_argument("--logvar-max", type=float, default=10.0)

    parser.add_argument("--y-bce-weight", type=float, default=1.0, help="BCE weight for predicate multi-label supervision.")
    parser.add_argument("--y-kl-weight", type=float, default=0.0, help="KL weight for q(y|x) vs Bernoulli prior.")
    parser.add_argument(
        "--y-dyn-kl-weight",
        type=float,
        default=0.0,
        help="Temporal KL regularizer weight for q(y_t|x_{1:t}) vs prior p(y_t|y_{t-1}), with p(y_0)=init prob.",
    )
    parser.add_argument("--y-temperature", type=float, default=0.67)
    parser.add_argument("--y-prior-prob", type=float, default=0.1)
    parser.add_argument(
        "--y-dyn-prior-init-prob",
        type=float,
        default=0.5,
        help="Initial Bernoulli prior probability for timestep-0 belief, p(y_0=1).",
    )
    parser.add_argument("--condition-z-on-y", action="store_true", default=False)
    parser.add_argument("--no-condition-z-on-y", action="store_false", dest="condition_z_on_y")
    parser.add_argument("--condition-decoder-on-y", action="store_true", default=True)
    parser.add_argument("--no-condition-decoder-on-y", action="store_false", dest="condition_decoder_on_y")
    parser.add_argument("--use-labeled-y-for-recon", action="store_true", default=True)
    parser.add_argument("--no-use-labeled-y-for-recon", action="store_false", dest="use_labeled_y_for_recon")
    parser.add_argument(
        "--labeled-fraction",
        type=float,
        default=1.0,
        help="Fraction of training samples to expose y labels for (1.0 = fully labeled).",
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
        help="How to choose labeled samples when labeled_fraction < 1: fixed subset by dataset index or resampled each batch.",
    )
    parser.add_argument(
        "--label-mask-seed",
        type=int,
        default=None,
        help="Seed for fixed label subset selection (defaults to --seed).",
    )

    parser.add_argument(
        "--init-backbone-path",
        type=str,
        default="",
        help="Optional path to teacher encoder checkpoint (teacher_frozen.pt) or watch_vae checkpoint for backbone init.",
    )
    parser.add_argument("--freeze-backbone", action="store_true", default=False)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--tensorboard-logdir", type=str, default="log_tb/watch_semisup_vae")
    parser.add_argument("--disable-tensorboard", action="store_true", default=False)
    parser.add_argument("--belief-profile-bins", type=int, default=21)
    parser.add_argument("--belief-profile-every", type=int, default=1)
    parser.add_argument("--disable-belief-profile-plots", action="store_true", default=False)
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
        for g in goals:
            name = str(g)
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

    bsz = goal_labels.shape[0]
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


def build_experiment_slug(args: argparse.Namespace) -> str:
    raw = (
        "hid{hid}_lat{lat}_lr{lr}_bs{bs}_ss{ss}_ra{ra}_aw{aw}_kl{kl}_ybce{ybce}_ykl{ykl}_ydyn{ydyn}_yt{yt}_lf{lf}"
    ).format(
        hid=args.hidden_size,
        lat=args.latent_size,
        lr=args.lr,
        bs=args.batch_size,
        ss=int(args.stable_slots),
        ra=int(args.reconstruct_actions),
        aw=args.action_weight,
        kl=args.kl_weight,
        ybce=args.y_bce_weight,
        ykl=args.y_kl_weight,
        ydyn=args.y_dyn_kl_weight,
        yt=args.y_temperature,
        lf=getattr(args, "effective_labeled_fraction", args.labeled_fraction),
    )
    raw = raw.replace(".", "p").replace("+", "").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_-]+", "", raw)


def build_experiment_title(args: argparse.Namespace) -> str:
    return (
        "watch_semisup_vae | hid={hid} lat={lat} lr={lr} bs={bs} kl={kl} recon_act={ra} aw={aw} "
        "y_bce={ybce} y_kl={ykl} y_dyn={ydyn} y0={y0} y_temp={yt} labeled={lf} "
        "condZ={cz} condDec={cd} teacherY={ty}"
    ).format(
        hid=args.hidden_size,
        lat=args.latent_size,
        lr=args.lr,
        bs=args.batch_size,
        kl=args.kl_weight,
        ra=int(args.reconstruct_actions),
        aw=args.action_weight,
        ybce=args.y_bce_weight,
        ykl=args.y_kl_weight,
        ydyn=args.y_dyn_kl_weight,
        y0=args.y_dyn_prior_init_prob,
        yt=args.y_temperature,
        lf=getattr(args, "effective_labeled_fraction", args.labeled_fraction),
        cz=int(args.condition_z_on_y),
        cd=int(args.condition_decoder_on_y),
        ty=int(args.use_labeled_y_for_recon),
    )


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


def _backbone_prefixes() -> Tuple[str, ...]:
    return ("frame_encoder.", "temporal_encoder.", "action_embedding.", "action_fuse.")


def maybe_load_backbone(model: GraphSequenceSemiSupVAE, ckpt_path: str, freeze_backbone: bool) -> Dict[str, object]:
    report: Dict[str, object] = {
        "loaded": False,
        "path": ckpt_path,
        "num_loaded_keys": 0,
        "freeze_backbone": bool(freeze_backbone),
    }
    if not ckpt_path:
        return report

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "encoder_state" in ckpt:
        source_state = ckpt["encoder_state"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        source_state = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        source_state = ckpt
    else:
        raise ValueError("Unsupported init-backbone checkpoint format: {}".format(type(ckpt)))

    target_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    for key, value in source_state.items():
        if not any(key.startswith(prefix) for prefix in _backbone_prefixes()):
            continue
        if key not in target_state:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append({"key": key, "source": list(value.shape), "target": list(target_state[key].shape)})
            continue
        filtered[key] = value

    incompatible = model.load_state_dict(filtered, strict=False)
    report.update(
        {
            "loaded": True,
            "num_loaded_keys": len(filtered),
            "missing_keys_after_partial_load": list(incompatible.missing_keys),
            "unexpected_keys_after_partial_load": list(incompatible.unexpected_keys),
            "skipped_shape": skipped_shape,
        }
    )

    if freeze_backbone:
        frozen = 0
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in _backbone_prefixes()):
                param.requires_grad = False
                frozen += 1
        report["num_frozen_params_tensors"] = frozen
    return report


def run_epoch(
    model: GraphSequenceSemiSupVAE,
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
    if train:
        model.train()
    else:
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
            losses = model(batch)  # type: ignore[arg-type]
            loss = losses["loss"]
            if not torch.isfinite(loss):
                diag = {
                    "batch_idx": batch_idx,
                    "train": train,
                    "loss_items": {k: float(v.detach().cpu().item()) for k, v in losses.items()},
                }
                raise RuntimeError("Non-finite loss in semisup training:\n{}".format(json.dumps(diag, indent=2)))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu().item())
        n_batches += 1

        if train and (batch_idx + 1) % log_interval == 0:
            print(
                "train step {:04d}/{:04d} loss={:.4f} goal_bce={:.4f} goal_f1={:.4f} kl_z={:.4f} action={:.4f} action_acc={:.4f}".format(
                    batch_idx + 1,
                    len(loader),
                    float(losses["loss"].item()),
                    float(losses["goal_bce_loss"].item()),
                    float(losses["goal_f1_0p5"].item()),
                    float(losses["kl_loss"].item()),
                    float(losses.get("action_loss", loss * 0.0).item()),
                    float(losses.get("action_acc", loss * 0.0).item()),
                )
            )

    if n_batches == 0:
        raise RuntimeError("No batches produced. Check dataset paths/settings.")

    metrics = {key: value / n_batches for key, value in running.items()}
    metrics["unknown_goal_labels"] = float(total_unknown_goals)
    return metrics


def compute_val_belief_profile(
    model: GraphSequenceSemiSupVAE,
    loader: DataLoader,
    goal_to_col: Dict[str, int],
    device: torch.device,
    num_bins: int = 21,
) -> Dict[str, object]:
    if num_bins < 2:
        raise ValueError("--belief-profile-bins must be >= 2")

    was_training = model.training
    model.eval()

    prob_sum_pos = np.zeros((num_bins,), dtype=np.float64)
    prob_cnt_pos = np.zeros((num_bins,), dtype=np.float64)
    prob_sum_neg = np.zeros((num_bins,), dtype=np.float64)
    prob_cnt_neg = np.zeros((num_bins,), dtype=np.float64)
    ent_sum_pos = np.zeros((num_bins,), dtype=np.float64)
    ent_cnt_pos = np.zeros((num_bins,), dtype=np.float64)
    ent_sum_neg = np.zeros((num_bins,), dtype=np.float64)
    ent_cnt_neg = np.zeros((num_bins,), dtype=np.float64)
    total_unknown_goals = 0
    total_labeled_demos = 0

    eps = 1e-6
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            batch, unknown = attach_goal_labels(
                batch=batch,
                goal_to_col=goal_to_col,
                device=device,
                train=False,
                label_mask_prob=0.0,
            )
            total_unknown_goals += int(unknown)

            y_out = model.infer_y_probs_seq(batch)  # type: ignore[arg-type]
            y_probs_seq = y_out["y_probs_seq"].detach().cpu().numpy()  # [B,T,K]

            goal_labels = batch["goal_labels"].detach().cpu().numpy() > 0.5
            goal_label_mask = batch.get("goal_label_mask")
            if goal_label_mask is None:
                label_mask = np.ones((goal_labels.shape[0],), dtype=np.float32)
            else:
                label_mask = goal_label_mask.detach().cpu().numpy()
            lengths = batch["lengths"].detach().cpu().numpy().astype(np.int64)
            time_mask = batch["time_mask"].detach().cpu().numpy().astype(bool)

            for b in range(y_probs_seq.shape[0]):
                if float(label_mask[b]) <= 0.5:
                    continue
                total_labeled_demos += 1
                length = int(lengths[b])
                if length <= 0:
                    continue
                if length == 1:
                    bin_ids = np.array([0], dtype=np.int64)
                else:
                    bin_ids = np.rint(np.linspace(0, num_bins - 1, num=length)).astype(np.int64)

                gt = goal_labels[b]  # [K]
                pos_mask = gt
                neg_mask = ~gt
                n_pos = int(pos_mask.sum())
                n_neg = int(neg_mask.sum())

                for t in range(length):
                    if not bool(time_mask[b, t]):
                        continue
                    j = int(bin_ids[t])
                    p = np.clip(y_probs_seq[b, t], eps, 1.0 - eps)
                    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
                    if n_pos > 0:
                        prob_sum_pos[j] += float(p[pos_mask].sum())
                        prob_cnt_pos[j] += float(n_pos)
                        ent_sum_pos[j] += float(ent[pos_mask].sum())
                        ent_cnt_pos[j] += float(n_pos)
                    if n_neg > 0:
                        prob_sum_neg[j] += float(p[neg_mask].sum())
                        prob_cnt_neg[j] += float(n_neg)
                        ent_sum_neg[j] += float(ent[neg_mask].sum())
                        ent_cnt_neg[j] += float(n_neg)

    if was_training:
        model.train()

    def _safe_mean(sum_arr: np.ndarray, cnt_arr: np.ndarray) -> np.ndarray:
        out = np.full_like(sum_arr, np.nan, dtype=np.float64)
        valid = cnt_arr > 0
        out[valid] = sum_arr[valid] / cnt_arr[valid]
        return out

    x_norm = np.linspace(0.0, 1.0, num_bins)
    prob_gt1 = _safe_mean(prob_sum_pos, prob_cnt_pos)
    prob_gt0 = _safe_mean(prob_sum_neg, prob_cnt_neg)
    ent_gt1 = _safe_mean(ent_sum_pos, ent_cnt_pos)
    ent_gt0 = _safe_mean(ent_sum_neg, ent_cnt_neg)
    gap = prob_gt1 - prob_gt0
    valid_gap = np.isfinite(gap)
    ent_all = np.concatenate([ent_gt1[np.isfinite(ent_gt1)], ent_gt0[np.isfinite(ent_gt0)]], axis=0)

    return {
        "num_bins": int(num_bins),
        "x_norm": x_norm.tolist(),
        "prob_gt1_mean": prob_gt1.tolist(),
        "prob_gt0_mean": prob_gt0.tolist(),
        "entropy_gt1_mean": ent_gt1.tolist(),
        "entropy_gt0_mean": ent_gt0.tolist(),
        "count_gt1": prob_cnt_pos.tolist(),
        "count_gt0": prob_cnt_neg.tolist(),
        "prob_gap_mean": float(np.nanmean(gap)) if valid_gap.any() else float("nan"),
        "entropy_mean": float(np.nanmean(ent_all)) if ent_all.size > 0 else float("nan"),
        "unknown_goal_labels": float(total_unknown_goals),
        "num_labeled_demos": int(total_labeled_demos),
    }


def save_and_log_belief_profile(
    profile: Dict[str, object],
    epoch: int,
    run_output_dir: str,
    writer: Optional[SummaryWriter],
) -> None:
    out_dir = os.path.join(run_output_dir, "belief_profile")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "val_belief_profile_epoch_{:03d}.json".format(epoch))
    with open(json_path, "w") as f:
        json.dump(profile, f, indent=2)

    if writer is not None:
        prob_gap_mean = profile.get("prob_gap_mean")
        entropy_mean = profile.get("entropy_mean")
        if isinstance(prob_gap_mean, (int, float)) and np.isfinite(float(prob_gap_mean)):
            writer.add_scalar("val_belief_profile/prob_gap_mean", float(prob_gap_mean), epoch)
        if isinstance(entropy_mean, (int, float)) and np.isfinite(float(entropy_mean)):
            writer.add_scalar("val_belief_profile/entropy_mean", float(entropy_mean), epoch)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("belief profile plot skipped (matplotlib unavailable):", repr(e))
        return

    x = np.asarray(profile["x_norm"], dtype=np.float64)
    prob_gt1 = np.asarray(profile["prob_gt1_mean"], dtype=np.float64)
    prob_gt0 = np.asarray(profile["prob_gt0_mean"], dtype=np.float64)
    ent_gt1 = np.asarray(profile["entropy_gt1_mean"], dtype=np.float64)
    ent_gt0 = np.asarray(profile["entropy_gt0_mean"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_prob, ax_ent = axes

    ax_prob.plot(x, prob_gt1, label="GT=1", linewidth=2)
    ax_prob.plot(x, prob_gt0, label="GT=0", linewidth=2)
    ax_prob.set_title("Validation Belief Probability vs Time")
    ax_prob.set_xlabel("Normalized trajectory time")
    ax_prob.set_ylabel("Mean predicted belief p(y=1)")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.grid(alpha=0.25)
    ax_prob.legend(loc="best")

    ax_ent.plot(x, ent_gt1, label="GT=1", linewidth=2)
    ax_ent.plot(x, ent_gt0, label="GT=0", linewidth=2)
    ax_ent.set_title("Validation Belief Entropy vs Time")
    ax_ent.set_xlabel("Normalized trajectory time")
    ax_ent.set_ylabel("Mean Bernoulli entropy")
    ax_ent.grid(alpha=0.25)
    ax_ent.legend(loc="best")

    fig.suptitle("SSVAE Belief Evolution | epoch {:03d}".format(epoch))
    fig.tight_layout()

    png_path = os.path.join(out_dir, "val_belief_profile_epoch_{:03d}.png".format(epoch))
    fig.savefig(png_path, dpi=150)
    if writer is not None:
        writer.add_figure("val/belief_evolution_profile", fig, global_step=epoch)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.reconstruct_actions is None:
        args.reconstruct_actions = bool(args.use_actions)
    else:
        args.reconstruct_actions = bool(args.reconstruct_actions)
    if args.action_weight < 0.0:
        raise ValueError("--action-weight must be >= 0")
    if args.reconstruct_actions and (not args.use_actions):
        raise ValueError("--reconstruct-actions requires --use-actions")
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

    tensorizer = WatchGraphTensorizer(metadata_path=args.metadata, max_nodes=args.max_nodes)
    goal_predicate_names, goal_to_col = load_goal_predicates(args.metadata)
    print("num goal predicates:", len(goal_predicate_names))
    print("effective labeled fraction:", args.effective_labeled_fraction)
    print("effective label mask prob:", args.effective_label_mask_prob)
    print("label mask mode:", args.label_mask_mode)
    print("reconstruct actions:", args.reconstruct_actions)
    print("action weight:", args.action_weight)

    train_dataset = WatchVAEDataset(
        data_path=args.train_json,
        split_key=args.train_split_key,
        tensorizer=tensorizer,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        use_actions=args.use_actions,
        max_demos=args.max_train_demos,
        stable_slots=args.stable_slots,
    )
    val_dataset = WatchVAEDataset(
        data_path=args.val_json,
        split_key=args.val_split_key,
        tensorizer=tensorizer,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        use_actions=args.use_actions,
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

    model = GraphSequenceSemiSupVAE(
        num_classes=tensorizer.num_classes,
        num_states=tensorizer.num_states,
        max_nodes=tensorizer.max_nodes,
        num_goal_predicates=len(goal_predicate_names),
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        transformer_nhead=args.transformer_nhead,
        use_actions=args.use_actions,
        num_actions=tensorizer.num_actions,
        reconstruct_actions=args.reconstruct_actions,
        action_weight=args.action_weight,
        kl_weight=args.kl_weight,
        class_weight=args.class_weight,
        state_weight=args.state_weight,
        coord_weight=args.coord_weight,
        mask_weight=args.mask_weight,
        logvar_min=args.logvar_min,
        logvar_max=args.logvar_max,
        y_bce_weight=args.y_bce_weight,
        y_kl_weight=args.y_kl_weight,
        y_dyn_kl_weight=args.y_dyn_kl_weight,
        y_temperature=args.y_temperature,
        y_prior_prob=args.y_prior_prob,
        y_dyn_prior_init_prob=args.y_dyn_prior_init_prob,
        condition_z_on_y=args.condition_z_on_y,
        condition_decoder_on_y=args.condition_decoder_on_y,
        use_labeled_y_for_recon=args.use_labeled_y_for_recon,
    ).to(device)

    init_report = maybe_load_backbone(model, args.init_backbone_path, args.freeze_backbone)
    print("backbone init report:", json.dumps(init_report, indent=2))

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    experiment_slug = build_experiment_slug(args)
    experiment_title = build_experiment_title(args)
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
        writer.add_text("config/goal_predicates_count", str(len(goal_predicate_names)), 0)
        writer.add_text("config/backbone_init_report", json.dumps(init_report, indent=2), 0)
        writer.add_scalar("config/effective_labeled_fraction", args.effective_labeled_fraction, 0)
        writer.add_scalar("config/effective_label_mask_prob", args.effective_label_mask_prob, 0)
        writer.add_scalar("config/action_weight", args.action_weight, 0)

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
            "epoch {:03d} train_loss={:.4f} val_loss={:.4f} val_goal_bce={:.4f} "
            "val_goal_f1_last={:.4f} val_goal_f1_all={:.4f} val_klz={:.4f} val_action={:.4f} val_action_acc={:.4f}".format(
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["goal_bce_loss"],
                val_metrics.get("goal_f1_0p5_last", val_metrics.get("goal_f1_0p5", 0.0)),
                val_metrics.get("goal_f1_0p5_allsteps", 0.0),
                val_metrics["kl_loss"],
                val_metrics.get("action_loss", 0.0),
                val_metrics.get("action_acc", 0.0),
            )
        )

        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar("train/{}".format(key), value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar("val/{}".format(key), value, epoch)
            writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], epoch)

        if (
            not args.disable_belief_profile_plots
            and args.belief_profile_every > 0
            and (epoch % args.belief_profile_every == 0)
        ):
            val_profile = compute_val_belief_profile(
                model=model,
                loader=val_loader,
                goal_to_col=goal_to_col,
                device=device,
                num_bins=args.belief_profile_bins,
            )
            save_and_log_belief_profile(
                profile=val_profile,
                epoch=epoch,
                run_output_dir=run_output_dir,
                writer=writer,
            )

        checkpoint = {
            "epoch": epoch,
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
            "init_backbone_report": init_report,
            "args": vars(args),
        }

        last_path = os.path.join(run_output_dir, "watch_semisup_vae_last.pt")
        torch.save(checkpoint, last_path)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = os.path.join(run_output_dir, "watch_semisup_vae_best.pt")
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, os.path.join(run_output_dir, "best_model.pt"))
            print("updated best model: val_loss={:.6f} -> {}".format(best_val, best_path))

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
                "goal_predicate_names": goal_predicate_names,
                "init_backbone_report": init_report,
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
