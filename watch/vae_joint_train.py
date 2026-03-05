import argparse
import datetime
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from watch.vae.dataset import WatchVAEDataset, collate_watch_vae
from watch.vae.model import GraphSequenceVAE
from watch.vae.tensorizer import WatchGraphTensorizer

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Watch Graph VAE jointly with all-timestep predicate-vector auxiliary supervision."
    )

    parser.add_argument("--metadata", type=str, default="dataset/watch_data/metadata.json")
    parser.add_argument("--train-json", type=str, default="dataset/watch_data/gather_data_actiongraph_train.json")
    parser.add_argument("--val-json", type=str, default="dataset/watch_data/gather_data_actiongraph_test.json")
    parser.add_argument("--train-split-key", type=str, default="train_data")
    parser.add_argument("--val-split-key", type=str, default="test_data")

    parser.add_argument("--output-dir", type=str, default="checkpoints/watch_vae_joint")
    parser.add_argument("--tensorboard-logdir", type=str, default="log_tb/watch_vae_joint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--disable-tensorboard", action="store_true", default=False)
    parser.add_argument("--latent-diag-batches", type=int, default=8)
    parser.add_argument("--belief-profile-bins", type=int, default=21)
    parser.add_argument("--belief-profile-every", type=int, default=1)
    parser.add_argument("--disable-belief-profile-plots", action="store_true", default=False)

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
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--kl-warmup-start-factor", type=float, default=0.0)
    parser.add_argument("--free-bits", type=float, default=0.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--mask-weight", type=float, default=0.2)
    parser.add_argument("--logvar-min", type=float, default=-10.0)
    parser.add_argument("--logvar-max", type=float, default=10.0)

    parser.add_argument("--predicate-aux-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight-min", type=float, default=1.0)
    parser.add_argument("--pos-weight-max", type=float, default=20.0)
    parser.add_argument(
        "--labeled-fraction",
        type=float,
        default=1.0,
        help="Fraction of training demos to use predicate supervision for (1.0 = fully labeled).",
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
        help="How labeled demos are selected when labeled_fraction < 1.0.",
    )
    parser.add_argument(
        "--label-mask-seed",
        type=int,
        default=None,
        help="Seed for fixed label subset selection (defaults to --seed).",
    )

    parser.add_argument("--save-teacher-encoder", action="store_true", default=True)
    parser.add_argument("--no-save-teacher-encoder", action="store_false", dest="save_teacher_encoder")
    parser.add_argument("--teacher-prefix", type=str, default="teacher_encoder")
    parser.add_argument(
        "--teacher-scope",
        type=str,
        default="backbone",
        choices=["backbone", "full_encoder"],
        help="Which subset of encoder parameters to export for frozen teacher use.",
    )

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


def encode_goal_multihot(goals_batch: Sequence[Sequence[str]], goal_to_col: Dict[str, int]) -> Tuple[np.ndarray, int]:
    y = np.zeros((len(goals_batch), len(goal_to_col)), dtype=np.float32)
    unknown = 0
    for i, goals in enumerate(goals_batch):
        for g in goals:
            col = goal_to_col.get(str(g))
            if col is None:
                unknown += 1
                continue
            y[i, col] = 1.0
    return y, unknown


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


def _compute_pos_counts_for_indices(
    demos: Sequence[Dict[str, object]],
    selected_indices: Sequence[int],
    goal_to_col: Dict[str, int],
) -> Tuple[np.ndarray, int]:
    pos_counts = np.zeros((len(goal_to_col),), dtype=np.float64)
    unknown = 0
    for idx in selected_indices:
        goals = demos[int(idx)].get("goal", [])
        seen_cols = set()
        for g in goals:
            col = goal_to_col.get(str(g))
            if col is None:
                unknown += 1
                continue
            if col in seen_cols:
                continue
            pos_counts[col] += 1.0
            seen_cols.add(col)
    return pos_counts, unknown


def compute_predicate_pos_weight(
    train_dataset: WatchVAEDataset,
    goal_to_col: Dict[str, int],
    label_mask_mode: str,
    fixed_label_keep_mask: Optional[torch.Tensor],
    pos_weight_min: float,
    pos_weight_max: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    n = len(train_dataset)
    if n <= 0 or len(goal_to_col) == 0:
        pw = torch.ones((len(goal_to_col),), dtype=torch.float32, device=device)
        return pw, {
            "num_selected_for_pos_weight": float(0),
            "unknown_goals_for_pos_weight": float(0),
            "pos_weight_min": float(pos_weight_min),
            "pos_weight_max": float(pos_weight_max),
            "pos_weight_mean": 1.0,
            "pos_weight_std": 0.0,
        }

    if label_mask_mode == "fixed" and fixed_label_keep_mask is not None:
        selected = torch.nonzero(fixed_label_keep_mask > 0.5, as_tuple=False).flatten().detach().cpu().tolist()
    else:
        selected = list(range(n))

    if len(selected) == 0:
        pw = torch.ones((len(goal_to_col),), dtype=torch.float32, device=device)
        return pw, {
            "num_selected_for_pos_weight": float(0),
            "unknown_goals_for_pos_weight": float(0),
            "pos_weight_min": float(pw.min().item()) if pw.numel() > 0 else 1.0,
            "pos_weight_max": float(pw.max().item()) if pw.numel() > 0 else 1.0,
            "pos_weight_mean": float(pw.mean().item()) if pw.numel() > 0 else 1.0,
            "pos_weight_std": float(pw.std(unbiased=False).item()) if pw.numel() > 1 else 0.0,
        }

    pos_counts, unknown = _compute_pos_counts_for_indices(
        demos=train_dataset.demos,
        selected_indices=selected,
        goal_to_col=goal_to_col,
    )
    total = float(len(selected))
    neg_counts = total - pos_counts
    pos_weight_np = neg_counts / np.maximum(pos_counts, 1.0)
    pos_weight_np = np.clip(pos_weight_np, a_min=float(pos_weight_min), a_max=float(pos_weight_max))
    pos_weight = torch.from_numpy(pos_weight_np.astype(np.float32)).to(device)
    stats = {
        "num_selected_for_pos_weight": float(len(selected)),
        "unknown_goals_for_pos_weight": float(unknown),
        "pos_weight_min": float(pos_weight.min().item()) if pos_weight.numel() > 0 else 1.0,
        "pos_weight_max": float(pos_weight.max().item()) if pos_weight.numel() > 0 else 1.0,
        "pos_weight_mean": float(pos_weight.mean().item()) if pos_weight.numel() > 0 else 1.0,
        "pos_weight_std": float(pos_weight.std(unbiased=False).item()) if pos_weight.numel() > 1 else 0.0,
    }
    return pos_weight, stats


def build_experiment_slug(args: argparse.Namespace) -> str:
    raw = (
        "hid{hid}_lat{lat}_lr{lr}_bs{bs}_act{act}_ss{ss}_ra{ra}_aw{aw}_paux{paux}_lf{lf}_mode{mode}_kl{kl}_kwu{kwu}_kws{kws}_fb{fb}"
    ).format(
        hid=args.hidden_size,
        lat=args.latent_size,
        lr=args.lr,
        bs=args.batch_size,
        act=int(args.use_actions),
        ss=int(args.stable_slots),
        ra=int(args.reconstruct_actions),
        aw=args.action_weight,
        paux=args.predicate_aux_weight,
        lf=getattr(args, "effective_labeled_fraction", args.labeled_fraction),
        mode=args.label_mask_mode,
        kl=args.kl_weight,
        kwu=args.kl_warmup_epochs,
        kws=args.kl_warmup_start_factor,
        fb=args.free_bits,
    )
    raw = raw.replace(".", "p").replace("+", "").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_-]+", "", raw)


def build_experiment_title(args: argparse.Namespace) -> str:
    return (
        "watch_vae_joint | hid={hid} lat={lat} lr={lr} bs={bs} act={act} "
        "recon_act={ra} aw={aw} pred_aux={paux} labeled={lf} mode={mode} "
        "kl={kl} kwu={kwu} kws={kws} fb={fb}"
    ).format(
        hid=args.hidden_size,
        lat=args.latent_size,
        lr=args.lr,
        bs=args.batch_size,
        act=int(args.use_actions),
        ra=int(args.reconstruct_actions),
        aw=args.action_weight,
        paux=args.predicate_aux_weight,
        lf=getattr(args, "effective_labeled_fraction", args.labeled_fraction),
        mode=args.label_mask_mode,
        kl=args.kl_weight,
        kwu=args.kl_warmup_epochs,
        kws=args.kl_warmup_start_factor,
        fb=args.free_bits,
    )


def kl_weight_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    warmup_epochs = int(args.kl_warmup_epochs)
    if warmup_epochs <= 0:
        return float(args.kl_weight)

    if warmup_epochs == 1:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, float(epoch - 1) / float(warmup_epochs - 1)))
    start = float(args.kl_warmup_start_factor)
    scale = start + (1.0 - start) * progress
    return float(args.kl_weight) * scale


@torch.no_grad()
def collect_latent_diagnostics(
    model: GraphSequenceVAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, object]:
    model.eval()
    mu_chunks: List[torch.Tensor] = []
    logvar_chunks: List[torch.Tensor] = []
    z_chunks: List[torch.Tensor] = []
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        _, mu, logvar = model.encode_sequence(
            class_ids=batch["class_ids"],  # type: ignore[index]
            coords=batch["coords"],  # type: ignore[index]
            states=batch["states"],  # type: ignore[index]
            node_mask=batch["node_mask"],  # type: ignore[index]
            lengths=batch["lengths"],  # type: ignore[index]
            action_ids=batch.get("action_ids"),  # type: ignore[arg-type]
        )
        valid_steps = batch["time_mask"].bool()  # type: ignore[index]
        if valid_steps.sum().item() == 0:
            continue

        mu_valid = mu[valid_steps]
        logvar_valid = logvar[valid_steps]
        z_valid = model.reparameterize(mu_valid, logvar_valid)

        finite_rows = (
            torch.isfinite(mu_valid).all(dim=-1)
            & torch.isfinite(logvar_valid).all(dim=-1)
            & torch.isfinite(z_valid).all(dim=-1)
        )
        if finite_rows.any():
            mu_chunks.append(mu_valid[finite_rows].detach().cpu())
            logvar_chunks.append(logvar_valid[finite_rows].detach().cpu())
            z_chunks.append(z_valid[finite_rows].detach().cpu())

        num_batches += 1
        if num_batches >= max_batches:
            break

    if len(mu_chunks) == 0:
        return {}

    mu_all = torch.cat(mu_chunks, dim=0)
    logvar_all = torch.cat(logvar_chunks, dim=0)
    z_all = torch.cat(z_chunks, dim=0)

    return {
        "scalars": {
            "points": float(mu_all.shape[0]),
            "latent_size": float(mu_all.shape[1]),
            "mu_mean": float(mu_all.mean().item()),
            "mu_std": float(mu_all.std(unbiased=False).item()),
            "mu_abs_mean": float(mu_all.abs().mean().item()),
            "logvar_mean": float(logvar_all.mean().item()),
            "logvar_std": float(logvar_all.std(unbiased=False).item()),
            "logvar_min": float(logvar_all.min().item()),
            "logvar_max": float(logvar_all.max().item()),
            "z_mean": float(z_all.mean().item()),
            "z_std": float(z_all.std(unbiased=False).item()),
            "z_abs_mean": float(z_all.abs().mean().item()),
            "z_abs_lt_0.1_frac": float((z_all.abs() < 0.1).float().mean().item()),
        },
        "mu_all": mu_all,
        "logvar_all": logvar_all,
        "z_all": z_all,
        "mu_dim_mean": mu_all.mean(dim=0),
        "mu_dim_std": mu_all.std(dim=0, unbiased=False),
        "logvar_dim_mean": logvar_all.mean(dim=0),
        "z_dim_std": z_all.std(dim=0, unbiased=False),
    }


def make_latent_figure(latent_diag: Dict[str, object], title: str):
    if plt is None or len(latent_diag) == 0:
        return None

    mu_dim_mean = latent_diag["mu_dim_mean"].numpy()
    mu_dim_std = latent_diag["mu_dim_std"].numpy()
    logvar_dim_mean = latent_diag["logvar_dim_mean"].numpy()
    z_dim_std = latent_diag["z_dim_std"].numpy()
    dims = np.arange(len(mu_dim_mean))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.reshape(-1)

    axes[0].plot(dims, mu_dim_mean)
    axes[0].set_title("mu per-dim mean")
    axes[0].set_xlabel("latent dim")
    axes[0].set_ylabel("value")

    axes[1].plot(dims, mu_dim_std)
    axes[1].set_title("mu per-dim std")
    axes[1].set_xlabel("latent dim")
    axes[1].set_ylabel("value")

    axes[2].plot(dims, logvar_dim_mean)
    axes[2].set_title("logvar per-dim mean")
    axes[2].set_xlabel("latent dim")
    axes[2].set_ylabel("value")

    axes[3].plot(dims, z_dim_std)
    axes[3].set_title("z per-dim std")
    axes[3].set_xlabel("latent dim")
    axes[3].set_ylabel("value")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_history_figure(history: list, title: str):
    if plt is None or len(history) == 0:
        return None

    metric_keys = sorted(set(history[-1]["train"].keys()) & set(history[-1]["val"].keys()))
    epochs = [item["epoch"] for item in history]
    n = len(metric_keys)
    ncols = 3
    nrows = int(math.ceil(max(1, n) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        train_values = [item["train"][key] for item in history]
        val_values = [item["val"][key] for item in history]
        ax.plot(epochs, train_values, label="train")
        ax.plot(epochs, val_values, label="val")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend()

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def build_teacher_checkpoint(
    model: GraphSequenceVAE,
    tensorizer: WatchGraphTensorizer,
    args: argparse.Namespace,
    run_name: str,
    run_output_dir: str,
    epoch: int,
    source_checkpoint: str,
    val_metrics: Dict[str, float],
) -> Dict[str, object]:
    teacher_state = model.get_teacher_state_dict(scope=args.teacher_scope)
    return {
        "source_checkpoint": source_checkpoint,
        "epoch": epoch,
        "run_name": run_name,
        "run_output_dir": run_output_dir,
        "teacher_scope": args.teacher_scope,
        "teacher_prefixes": model.get_teacher_prefixes(scope=args.teacher_scope),
        "model_config": model.get_config(),
        "encoder_state": teacher_state,
        "tensorizer_config": tensorizer.to_config(),
        "val_metrics": val_metrics,
        "args": vars(args),
    }


def compute_goal_aux_metrics_allsteps(
    logits_seq: torch.Tensor,
    goal_labels: torch.Tensor,
    time_mask: torch.Tensor,
    goal_label_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    labels_seq = goal_labels.unsqueeze(1).expand(-1, logits_seq.shape[1], -1)
    supervision_mask = time_mask.float() * goal_label_mask.float().unsqueeze(1)
    supervision_mask_exp = supervision_mask.unsqueeze(-1)

    bce = F.binary_cross_entropy_with_logits(
        logits_seq,
        labels_seq,
        reduction="none",
        pos_weight=pos_weight.view(1, 1, -1),
    ).mean(dim=-1)
    denom_steps = supervision_mask.sum().clamp(min=1.0)
    goal_bce = (bce * supervision_mask).sum() / denom_steps

    probs = torch.sigmoid(logits_seq)
    pred = (probs >= 0.5).float()
    tp = (pred * labels_seq * supervision_mask_exp).sum()
    fp = (pred * (1.0 - labels_seq) * supervision_mask_exp).sum()
    fn = ((1.0 - pred) * labels_seq * supervision_mask_exp).sum()
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = (2.0 * precision * recall) / (precision + recall).clamp(min=1e-6)
    pred_pos_rate = (pred * supervision_mask_exp).sum() / supervision_mask_exp.sum().clamp(min=1.0)

    return {
        "goal_bce_allsteps": goal_bce,
        "goal_precision_0p5_allsteps": precision,
        "goal_recall_0p5_allsteps": recall,
        "goal_f1_0p5_allsteps": f1,
        "goal_pred_pos_rate_0p5_allsteps": pred_pos_rate,
        "goal_labeled_fraction": goal_label_mask.float().mean(),
        "goal_supervised_steps": supervision_mask.sum(),
    }


def run_epoch(
    model: GraphSequenceVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    goal_to_col: Dict[str, int],
    pos_weight: torch.Tensor,
    predicate_aux_weight: float,
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
            action_ids = batch.get("action_ids")
            encoded, mu, logvar = model.encode_sequence(
                class_ids=batch["class_ids"],  # type: ignore[index]
                coords=batch["coords"],  # type: ignore[index]
                states=batch["states"],  # type: ignore[index]
                node_mask=batch["node_mask"],  # type: ignore[index]
                lengths=batch["lengths"],  # type: ignore[index]
                action_ids=action_ids,  # type: ignore[arg-type]
            )
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z)
            vae_losses = model.compute_losses(
                recon=recon,
                class_ids=batch["class_ids"],  # type: ignore[index]
                states=batch["states"],  # type: ignore[index]
                coords=batch["coords"],  # type: ignore[index]
                node_mask=batch["node_mask"],  # type: ignore[index]
                time_mask=batch["time_mask"],  # type: ignore[index]
                mu=mu,
                logvar=logvar,
                action_ids=action_ids,  # type: ignore[arg-type]
            )

            logits_seq = model.predicate_logits_from_mu_sequence(mu)
            goal_metrics = compute_goal_aux_metrics_allsteps(
                logits_seq=logits_seq,
                goal_labels=batch["goal_labels"],  # type: ignore[index]
                time_mask=batch["time_mask"],  # type: ignore[index]
                goal_label_mask=batch["goal_label_mask"],  # type: ignore[index]
                pos_weight=pos_weight,
            )

            total_loss = vae_losses["loss"] + float(predicate_aux_weight) * goal_metrics["goal_bce_allsteps"]

            losses: Dict[str, torch.Tensor] = dict(vae_losses)
            losses["loss_vae"] = vae_losses["loss"]
            losses.update(goal_metrics)
            losses["loss"] = total_loss

            if not torch.isfinite(total_loss):
                diag = {
                    "batch_idx": int(batch_idx),
                    "train": bool(train),
                    "loss_items": {k: float(v.detach().cpu().item()) for k, v in losses.items()},
                }
                raise RuntimeError("Non-finite loss in joint training:\n{}".format(json.dumps(diag, indent=2)))

            if train:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu().item())
        n_batches += 1

        if train and (batch_idx + 1) % log_interval == 0:
            print(
                "train step {:04d}/{:04d} loss={:.4f} vae={:.4f} goal_bce={:.4f} goal_f1={:.4f} labeled={:.4f} kl={:.4f} action={:.4f} action_acc={:.4f}".format(
                    batch_idx + 1,
                    len(loader),
                    float(losses["loss"].item()),
                    float(losses["loss_vae"].item()),
                    float(losses["goal_bce_allsteps"].item()),
                    float(losses["goal_f1_0p5_allsteps"].item()),
                    float(losses["goal_labeled_fraction"].item()),
                    float(losses["kl_loss"].item()),
                    float(losses["action_loss"].item()),
                    float(losses["action_acc"].item()),
                )
            )

    if n_batches == 0:
        raise RuntimeError("No batches produced. Check dataset paths/settings.")

    metrics = {key: value / n_batches for key, value in running.items()}
    metrics["unknown_goal_labels"] = float(total_unknown_goals)
    return metrics


def compute_val_belief_profile(
    model: GraphSequenceVAE,
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

            action_ids = batch.get("action_ids")
            _, mu, _ = model.encode_sequence(
                class_ids=batch["class_ids"],  # type: ignore[index]
                coords=batch["coords"],  # type: ignore[index]
                states=batch["states"],  # type: ignore[index]
                node_mask=batch["node_mask"],  # type: ignore[index]
                lengths=batch["lengths"],  # type: ignore[index]
                action_ids=action_ids,  # type: ignore[arg-type]
            )
            logits_seq = model.predicate_logits_from_mu_sequence(mu)
            y_probs_seq = torch.sigmoid(logits_seq).detach().cpu().numpy()  # [B,T,K]

            goal_labels = batch["goal_labels"].detach().cpu().numpy() > 0.5  # type: ignore[index]
            goal_label_mask = batch.get("goal_label_mask")
            if goal_label_mask is None:
                label_mask = np.ones((goal_labels.shape[0],), dtype=np.float32)
            else:
                label_mask = goal_label_mask.detach().cpu().numpy()
            lengths = batch["lengths"].detach().cpu().numpy().astype(np.int64)  # type: ignore[index]
            time_mask = batch["time_mask"].detach().cpu().numpy().astype(bool)  # type: ignore[index]

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
    make_plot: bool = True,
    file_prefix: str = "val_belief_profile",
    plot_title_prefix: str = "Joint VAE Belief Evolution",
) -> Dict[str, str]:
    out_dir = os.path.join(run_output_dir, "belief_profile")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "{}_epoch_{:03d}.json".format(file_prefix, epoch))
    with open(json_path, "w") as f:
        json.dump(profile, f, indent=2)

    writer_prefix = file_prefix.replace("-", "_")
    if writer is not None:
        prob_gap_mean = profile.get("prob_gap_mean")
        entropy_mean = profile.get("entropy_mean")
        if isinstance(prob_gap_mean, (int, float)) and np.isfinite(float(prob_gap_mean)):
            writer.add_scalar("{}/prob_gap_mean".format(writer_prefix), float(prob_gap_mean), epoch)
        if isinstance(entropy_mean, (int, float)) and np.isfinite(float(entropy_mean)):
            writer.add_scalar("{}/entropy_mean".format(writer_prefix), float(entropy_mean), epoch)

    if not make_plot:
        return {"json_path": json_path}
    if plt is None:
        print("belief profile plot skipped (matplotlib unavailable).")
        return {"json_path": json_path}

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

    fig.suptitle("{} | epoch {:03d}".format(plot_title_prefix, epoch))
    fig.tight_layout()

    png_path = os.path.join(out_dir, "{}_epoch_{:03d}.png".format(file_prefix, epoch))
    fig.savefig(png_path, dpi=150)
    if writer is not None:
        writer.add_figure("{}/belief_evolution_profile".format(writer_prefix), fig, global_step=epoch)
    plt.close(fig)
    return {"json_path": json_path, "png_path": png_path}


def main() -> None:
    args = parse_args()
    if args.reconstruct_actions is None:
        args.reconstruct_actions = bool(args.use_actions)
    else:
        args.reconstruct_actions = bool(args.reconstruct_actions)
    resolve_label_fraction(args)

    if args.kl_warmup_epochs < 0:
        raise ValueError("--kl-warmup-epochs must be >= 0")
    if not (0.0 <= args.kl_warmup_start_factor <= 1.0):
        raise ValueError("--kl-warmup-start-factor must be in [0, 1]")
    if args.free_bits < 0.0:
        raise ValueError("--free-bits must be >= 0")
    if args.action_weight < 0.0:
        raise ValueError("--action-weight must be >= 0")
    if args.reconstruct_actions and (not args.use_actions):
        raise ValueError("--reconstruct-actions requires --use-actions")
    if args.predicate_aux_weight < 0.0:
        raise ValueError("--predicate-aux-weight must be >= 0")
    if args.pos_weight_min <= 0.0 or args.pos_weight_max <= 0.0:
        raise ValueError("--pos-weight-min/--pos-weight-max must be > 0")
    if args.pos_weight_min > args.pos_weight_max:
        raise ValueError("--pos-weight-min cannot exceed --pos-weight-max")
    if args.belief_profile_bins < 2:
        raise ValueError("--belief-profile-bins must be >= 2")
    if args.belief_profile_every < 0:
        raise ValueError("--belief-profile-every must be >= 0")

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

    train_dataset = WatchVAEDataset(
        data_path=args.train_json,
        split_key=args.train_split_key,
        tensorizer=tensorizer,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        use_actions=args.use_actions,
        max_demos=args.max_train_demos,
        stable_slots=args.stable_slots,
        strict_action_indexing=bool(args.use_actions),
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
        strict_action_indexing=bool(args.use_actions),
    )
    if args.use_actions:
        train_preflight = train_dataset.get_action_preflight_summary()
        val_preflight = val_dataset.get_action_preflight_summary()
        print("action preflight (train):", json.dumps(train_preflight, indent=2))
        print("action preflight (val):", json.dumps(val_preflight, indent=2))
        if int(train_preflight.get("unknown_count", 0)) != 0:
            raise RuntimeError(
                "Train action preflight has unknown canonical actions: {}".format(
                    train_preflight.get("unknown_by_key", {})
                )
            )
        if int(val_preflight.get("unknown_count", 0)) != 0:
            raise RuntimeError(
                "Val action preflight has unknown canonical actions: {}".format(
                    val_preflight.get("unknown_by_key", {})
                )
            )
    print("train demos:", len(train_dataset))
    print("val demos:", len(val_dataset))
    print("num classes:", tensorizer.num_classes)
    print("num states:", tensorizer.num_states)
    print("num actions:", tensorizer.num_actions)
    print("stable slots:", args.stable_slots)
    print("reconstruct actions:", args.reconstruct_actions)
    print("action weight:", args.action_weight)
    print("predicate aux weight:", args.predicate_aux_weight)
    print("belief profile bins:", args.belief_profile_bins)
    print("belief profile every:", args.belief_profile_every)
    print("belief profile plots enabled:", not args.disable_belief_profile_plots)

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

    pos_weight, pos_weight_stats = compute_predicate_pos_weight(
        train_dataset=train_dataset,
        goal_to_col=goal_to_col,
        label_mask_mode=args.label_mask_mode,
        fixed_label_keep_mask=fixed_label_keep_mask,
        pos_weight_min=args.pos_weight_min,
        pos_weight_max=args.pos_weight_max,
        device=device,
    )
    print("predicate pos_weight stats:", json.dumps(pos_weight_stats, indent=2))

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

    model = GraphSequenceVAE(
        num_classes=tensorizer.num_classes,
        num_states=tensorizer.num_states,
        max_nodes=tensorizer.max_nodes,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        transformer_nhead=args.transformer_nhead,
        use_actions=args.use_actions,
        num_actions=tensorizer.num_actions,
        reconstruct_actions=args.reconstruct_actions,
        action_weight=args.action_weight,
        kl_weight=args.kl_weight,
        free_bits=args.free_bits,
        class_weight=args.class_weight,
        state_weight=args.state_weight,
        coord_weight=args.coord_weight,
        mask_weight=args.mask_weight,
        logvar_min=args.logvar_min,
        logvar_max=args.logvar_max,
        enable_predicate_head=True,
        num_goal_predicates=len(goal_predicate_names),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
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
    plot_dir = os.path.join(run_output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("run_name:", run_name)
    print("run_output_dir:", run_output_dir)
    print("teacher_scope:", args.teacher_scope)
    if not args.disable_tensorboard:
        print("tensorboard_run_dir:", tb_run_dir)

    writer = None
    if not args.disable_tensorboard:
        writer = SummaryWriter(log_dir=tb_run_dir)
        writer.add_text("config/args", json.dumps(vars(args), indent=2), 0)
        writer.add_text("config/experiment_title", experiment_title, 0)
        writer.add_text("config/run_name", run_name, 0)
        writer.add_text("config/run_output_dir", run_output_dir, 0)
        writer.add_text("config/tensorboard_run_dir", tb_run_dir, 0)
        writer.add_text("config/teacher_scope", args.teacher_scope, 0)
        writer.add_scalar("config/kl_weight_target", args.kl_weight, 0)
        writer.add_scalar("config/kl_warmup_epochs", args.kl_warmup_epochs, 0)
        writer.add_scalar("config/kl_warmup_start_factor", args.kl_warmup_start_factor, 0)
        writer.add_scalar("config/free_bits", args.free_bits, 0)
        writer.add_scalar("config/action_weight", args.action_weight, 0)
        writer.add_scalar("config/predicate_aux_weight", args.predicate_aux_weight, 0)
        writer.add_scalar("config/effective_labeled_fraction", args.effective_labeled_fraction, 0)
        writer.add_scalar("config/effective_label_mask_prob", args.effective_label_mask_prob, 0)
        writer.add_scalar("config/predicate_pos_weight_min", pos_weight_stats["pos_weight_min"], 0)
        writer.add_scalar("config/predicate_pos_weight_max", pos_weight_stats["pos_weight_max"], 0)
        writer.add_scalar("config/predicate_pos_weight_mean", pos_weight_stats["pos_weight_mean"], 0)
        writer.add_scalar("config/predicate_pos_weight_std", pos_weight_stats["pos_weight_std"], 0)
        writer.add_scalar("config/belief_profile_bins", args.belief_profile_bins, 0)
        writer.add_scalar("config/belief_profile_every", args.belief_profile_every, 0)
    if plt is None:
        print("Warning: matplotlib not available; figure plotting disabled.")

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        current_kl_weight = kl_weight_for_epoch(args, epoch)
        model.kl_weight = current_kl_weight

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            goal_to_col=goal_to_col,
            pos_weight=pos_weight,
            predicate_aux_weight=args.predicate_aux_weight,
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
            pos_weight=pos_weight,
            predicate_aux_weight=args.predicate_aux_weight,
            device=device,
            train=False,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
            label_mask_prob=0.0,
            label_mask_mode="fixed",
            fixed_label_keep_mask=None,
        )

        print(
            "epoch {:03d} kl_w={:.6f} train_loss={:.4f} train_labeled={:.4f} train_action={:.4f} train_action_acc={:.4f} val_loss={:.4f} val_labeled={:.4f} "
            "val_vae={:.4f} val_goal_bce={:.4f} val_goal_f1={:.4f} "
            "val_kl={:.4f} val_action={:.4f} val_action_acc={:.4f}".format(
                epoch,
                current_kl_weight,
                train_metrics["loss"],
                train_metrics["goal_labeled_fraction"],
                train_metrics["action_loss"],
                train_metrics["action_acc"],
                val_metrics["loss"],
                val_metrics["goal_labeled_fraction"],
                val_metrics["loss_vae"],
                val_metrics["goal_bce_allsteps"],
                val_metrics["goal_f1_0p5_allsteps"],
                val_metrics["kl_loss"],
                val_metrics["action_loss"],
                val_metrics["action_acc"],
            )
        )

        latent_diag = collect_latent_diagnostics(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.latent_diag_batches,
        )
        latent_scalars = latent_diag.get("scalars", {}) if len(latent_diag) > 0 else {}

        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar("train/{}".format(key), value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar("val/{}".format(key), value, epoch)
            writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("optim/kl_weight", current_kl_weight, epoch)
            for key, value in latent_scalars.items():
                writer.add_scalar("latent/{}".format(key), value, epoch)
            if len(latent_diag) > 0:
                writer.add_histogram("latent/mu_hist", latent_diag["mu_all"], epoch)
                writer.add_histogram("latent/logvar_hist", latent_diag["logvar_all"], epoch)
                writer.add_histogram("latent/z_hist", latent_diag["z_all"], epoch)
                writer.add_text(
                    "latent/z_shape",
                    "{} x {}".format(int(latent_scalars.get("points", 0)), int(latent_scalars.get("latent_size", 0))),
                    epoch,
                )

        if args.belief_profile_every > 0 and (epoch % args.belief_profile_every == 0):
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
                make_plot=(not args.disable_belief_profile_plots),
            )

        latent_fig = make_latent_figure(
            latent_diag,
            "{} | epoch {} | latent diagnostics".format(experiment_title, epoch),
        )
        if latent_fig is not None:
            latent_plot_path = os.path.join(plot_dir, "latent_epoch_{:03d}.png".format(epoch))
            latent_fig.savefig(latent_plot_path, dpi=140)
            if writer is not None:
                writer.add_figure("plots/latent", latent_fig, epoch)
            plt.close(latent_fig)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": model.get_config(),
            "tensorizer_config": tensorizer.to_config(),
            "goal_predicate_names": goal_predicate_names,
            "goal_to_col": goal_to_col,
            "predicate_pos_weight": pos_weight.detach().cpu(),
            "predicate_pos_weight_stats": pos_weight_stats,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "run_name": run_name,
            "run_output_dir": run_output_dir,
            "tensorboard_run_dir": tb_run_dir if not args.disable_tensorboard else None,
            "args": vars(args),
        }

        last_path = os.path.join(run_output_dir, "watch_vae_joint_last.pt")
        torch.save(checkpoint, last_path)

        if args.save_teacher_encoder:
            teacher_last_path = os.path.join(run_output_dir, "{}_last.pt".format(args.teacher_prefix))
            teacher_last = build_teacher_checkpoint(
                model=model,
                tensorizer=tensorizer,
                args=args,
                run_name=run_name,
                run_output_dir=run_output_dir,
                epoch=epoch,
                source_checkpoint=last_path,
                val_metrics=val_metrics,
            )
            torch.save(teacher_last, teacher_last_path)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = os.path.join(run_output_dir, "watch_vae_joint_best.pt")
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, os.path.join(run_output_dir, "best_model.pt"))
            print("updated best model: val_loss={:.6f} -> {}".format(best_val, best_path))

            if args.save_teacher_encoder:
                teacher_best_path = os.path.join(run_output_dir, "{}_best.pt".format(args.teacher_prefix))
                teacher_best_alias_path = os.path.join(run_output_dir, "teacher_frozen.pt")
                teacher_best = build_teacher_checkpoint(
                    model=model,
                    tensorizer=tensorizer,
                    args=args,
                    run_name=run_name,
                    run_output_dir=run_output_dir,
                    epoch=epoch,
                    source_checkpoint=best_path,
                    val_metrics=val_metrics,
                )
                torch.save(teacher_best, teacher_best_path)
                torch.save(teacher_best, teacher_best_alias_path)
                print("updated teacher encoder: {}".format(teacher_best_path))

        history.append(
            {
                "epoch": epoch,
                "kl_weight": current_kl_weight,
                "train": train_metrics,
                "val": val_metrics,
                "latent": latent_scalars,
            }
        )
        with open(os.path.join(run_output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        history_fig = make_history_figure(
            history,
            "{} | Training curves (epoch {})".format(experiment_title, epoch),
        )
        if history_fig is not None:
            curves_path = os.path.join(plot_dir, "training_curves.png")
            curves_epoch_path = os.path.join(plot_dir, "training_curves_epoch_{:03d}.png".format(epoch))
            history_fig.savefig(curves_path, dpi=140)
            history_fig.savefig(curves_epoch_path, dpi=140)
            if writer is not None:
                writer.add_figure("plots/training_curves", history_fig, epoch)
            plt.close(history_fig)

    with open(os.path.join(run_output_dir, "run_info.json"), "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "run_output_dir": run_output_dir,
                "tensorboard_run_dir": tb_run_dir if not args.disable_tensorboard else None,
                "experiment_title": experiment_title,
                "experiment_slug": experiment_slug,
                "save_teacher_encoder": args.save_teacher_encoder,
                "teacher_prefix": args.teacher_prefix,
                "teacher_scope": args.teacher_scope,
                "goal_predicate_names": goal_predicate_names,
                "predicate_pos_weight_stats": pos_weight_stats,
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
