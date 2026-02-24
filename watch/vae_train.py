import argparse
import datetime
import json
import math
import os
import random
import re
from typing import Dict

import numpy as np
import torch
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
    parser = argparse.ArgumentParser(description="Train an amortized LSTM-VAE on watch graph trajectories.")

    parser.add_argument("--metadata", type=str, default="dataset/watch_data/metadata.json")
    parser.add_argument("--train-json", type=str, default="dataset/watch_data/gather_data_actiongraph_train.json")
    parser.add_argument("--val-json", type=str, default="dataset/watch_data/gather_data_actiongraph_test.json")
    parser.add_argument("--train-split-key", type=str, default="train_data")
    parser.add_argument("--val-split-key", type=str, default="test_data")

    parser.add_argument("--output-dir", type=str, default="checkpoints/watch_vae")
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
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--mask-weight", type=float, default=0.2)
    parser.add_argument("--logvar-min", type=float, default=-10.0)
    parser.add_argument("--logvar-max", type=float, default=10.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--tensorboard-logdir", type=str, default="log_tb/watch_vae")
    parser.add_argument("--disable-tensorboard", action="store_true", default=False)
    parser.add_argument("--detect-anomaly", action="store_true", default=False)
    parser.add_argument("--latent-diag-batches", type=int, default=8)
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

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def build_experiment_title(args: argparse.Namespace) -> str:
    return (
        "watch_vae | hid={hid} lat={lat} lr={lr} bs={bs} act={act} stable_slots={ss} "
        "kl={kl} cls={cw} st={sw} coord={ow} mask={mw}"
    ).format(
        hid=args.hidden_size,
        lat=args.latent_size,
        lr=args.lr,
        bs=args.batch_size,
        act=int(args.use_actions),
        ss=int(args.stable_slots),
        kl=args.kl_weight,
        cw=args.class_weight,
        sw=args.state_weight,
        ow=args.coord_weight,
        mw=args.mask_weight,
    )


def build_experiment_slug(args: argparse.Namespace) -> str:
    raw = (
        "hid{hid}_lat{lat}_lr{lr}_bs{bs}_act{act}_ss{ss}_kl{kl}_cw{cw}_sw{sw}_ow{ow}_mw{mw}"
    ).format(
        hid=args.hidden_size,
        lat=args.latent_size,
        lr=args.lr,
        bs=args.batch_size,
        act=int(args.use_actions),
        ss=int(args.stable_slots),
        kl=args.kl_weight,
        cw=args.class_weight,
        sw=args.state_weight,
        ow=args.coord_weight,
        mw=args.mask_weight,
    )
    raw = raw.replace(".", "p")
    raw = raw.replace("+", "")
    raw = raw.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_-]+", "", raw)


@torch.no_grad()
def collect_latent_diagnostics(
    model: GraphSequenceVAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, object]:
    model.eval()
    mu_chunks = []
    logvar_chunks = []
    z_chunks = []
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        _, mu, logvar = model.encode_sequence(
            class_ids=batch["class_ids"],
            coords=batch["coords"],
            states=batch["states"],
            node_mask=batch["node_mask"],
            lengths=batch["lengths"],
            action_ids=batch.get("action_ids"),
        )
        valid_steps = batch["time_mask"].bool()
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

    diag = {
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
    return diag


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


def run_epoch(
    model: GraphSequenceVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    log_interval: int,
    grad_clip: float,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    running: Dict[str, float] = {}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            losses = model(batch)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                nan_report = {
                    "batch_idx": int(batch_idx),
                    "train": bool(train),
                    "loss_items": {k: float(v.detach().cpu().item()) for k, v in losses.items()},
                    "names_sample": list(batch.get("names", []))[:4],
                    "task_names_sample": list(batch.get("task_names", []))[:4],
                    "batch_stats": {
                        "class_ids_min": int(batch["class_ids"].min().detach().cpu().item()),
                        "class_ids_max": int(batch["class_ids"].max().detach().cpu().item()),
                        "coords_abs_max": float(batch["coords"].abs().max().detach().cpu().item()),
                        "coords_mean_abs": float(batch["coords"].abs().mean().detach().cpu().item()),
                        "states_min": float(batch["states"].min().detach().cpu().item()),
                        "states_max": float(batch["states"].max().detach().cpu().item()),
                        "node_mask_sum_min": float(batch["node_mask"].sum(dim=(1, 2)).min().detach().cpu().item()),
                        "node_mask_sum_max": float(batch["node_mask"].sum(dim=(1, 2)).max().detach().cpu().item()),
                        "time_mask_sum_min": float(batch["time_mask"].sum(dim=1).min().detach().cpu().item()),
                        "time_mask_sum_max": float(batch["time_mask"].sum(dim=1).max().detach().cpu().item()),
                        "length_min": int(batch["lengths"].min().detach().cpu().item()),
                        "length_max": int(batch["lengths"].max().detach().cpu().item()),
                    },
                }
                raise RuntimeError("Non-finite loss detected:\\n{}".format(json.dumps(nan_report, indent=2)))

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu().item())

        n_batches += 1

        if train and (batch_idx + 1) % log_interval == 0:
            print(
                "train step {:04d}/{:04d} loss={:.4f} kl={:.4f} class={:.4f} state={:.4f} coord={:.4f}".format(
                    batch_idx + 1,
                    len(loader),
                    float(losses["loss"].item()),
                    float(losses["kl_loss"].item()),
                    float(losses["class_loss"].item()),
                    float(losses["state_loss"].item()),
                    float(losses["coord_loss"].item()),
                )
            )

    if n_batches == 0:
        raise RuntimeError("No batches were produced. Check dataset paths/splits/min_seq_len settings.")

    return {key: value / n_batches for key, value in running.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.disable_tensorboard:
        os.makedirs(args.tensorboard_logdir, exist_ok=True)

    experiment_slug = build_experiment_slug(args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}_{}".format(timestamp, experiment_slug)
    run_output_dir = os.path.join(args.output_dir, run_name)
    tb_run_dir = os.path.join(args.tensorboard_logdir, run_name)

    os.makedirs(run_output_dir, exist_ok=True)
    plot_dir = os.path.join(run_output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    experiment_title = build_experiment_title(args)
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
    if plt is None:
        print("Warning: matplotlib not available; figure plotting disabled.")

    tensorizer = WatchGraphTensorizer(metadata_path=args.metadata, max_nodes=args.max_nodes)

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
    print("num classes:", tensorizer.num_classes)
    print("num states:", tensorizer.num_states)
    print("num actions:", tensorizer.num_actions)
    print("stable slots:", args.stable_slots)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_watch_vae,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_watch_vae,
        drop_last=False,
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
        kl_weight=args.kl_weight,
        class_weight=args.class_weight,
        state_weight=args.state_weight,
        coord_weight=args.coord_weight,
        mask_weight=args.mask_weight,
        logvar_min=args.logvar_min,
        logvar_max=args.logvar_max,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train=False,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
        )

        print(
            "epoch {:03d} train_loss={:.4f} val_loss={:.4f} val_kl={:.4f} val_class={:.4f} val_state={:.4f} val_coord={:.4f}".format(
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["kl_loss"],
                val_metrics["class_loss"],
                val_metrics["state_loss"],
                val_metrics["coord_loss"],
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
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "run_name": run_name,
            "run_output_dir": run_output_dir,
            "tensorboard_run_dir": tb_run_dir if not args.disable_tensorboard else None,
            "args": vars(args),
        }

        last_path = os.path.join(run_output_dir, "watch_vae_last.pt")
        torch.save(checkpoint, last_path)

        teacher_last_path = None
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
            best_path = os.path.join(run_output_dir, "watch_vae_best.pt")
            torch.save(checkpoint, best_path)
            best_alias_path = os.path.join(run_output_dir, "best_model.pt")
            torch.save(checkpoint, best_alias_path)
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
                "train": train_metrics,
                "val": val_metrics,
                "latent": latent_scalars,
            }
        )

        with open(os.path.join(run_output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        history_fig = make_history_figure(
            history,
            "{} | through epoch {}".format(experiment_title, epoch),
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
