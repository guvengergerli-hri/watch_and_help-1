import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from watch.vae.dataset import WatchVAEDataset, collate_watch_vae
from watch.vae.model import GraphSequenceVAE
from watch.vae.tensorizer import WatchGraphTensorizer

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Watch VAE latents (last mu) with UMAP and goal-predicate presence statistics."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to watch_vae_{best,last}.pt checkpoint")
    parser.add_argument("--metadata", type=str, default=None, help="Override metadata path (defaults to checkpoint args)")
    parser.add_argument("--data-json", type=str, default=None, help="Override dataset json path (defaults to checkpoint val-json)")
    parser.add_argument("--split-key", type=str, default=None, help="Override split key (defaults to checkpoint val-split-key)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-demos", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stable-slots", action="store_true", default=True)
    parser.add_argument("--no-stable-slots", action="store_false", dest="stable_slots")

    parser.add_argument("--reducer", type=str, default="auto", choices=["auto", "umap", "tsne", "pca", "none"])
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="euclidean")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)

    parser.add_argument("--plot-top-k-predicates", type=int, default=6)
    parser.add_argument("--min-predicate-count", type=int, default=20)
    parser.add_argument(
        "--predicate-list",
        type=str,
        default="",
        help="Comma-separated predicate names to plot explicitly (e.g., holds_book,on_apple_coffeetable)",
    )
    parser.add_argument("--max-task-colors", type=int, default=10)

    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def load_metadata_goal_vocab(metadata_path: str) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    raw_goal_predicates = metadata.get("goal_predicates", {})
    if not raw_goal_predicates:
        raise KeyError("metadata.json does not contain 'goal_predicates'")

    # Metadata indices are typically 1-based with 0 reserved for padding.
    max_idx = max(int(v) for v in raw_goal_predicates.values())
    names_by_idx = ["<PAD>"] * (max_idx + 1)
    for name, idx in raw_goal_predicates.items():
        idx_int = int(idx)
        if 0 <= idx_int < len(names_by_idx):
            names_by_idx[idx_int] = str(name)

    nonpad_names = [names_by_idx[i] for i in range(1, len(names_by_idx)) if names_by_idx[i] != "<PAD>"]
    name_to_nonpad_idx = {name: i for i, name in enumerate(nonpad_names)}
    name_to_meta_idx = {str(name): int(idx) for name, idx in raw_goal_predicates.items()}
    return nonpad_names, name_to_nonpad_idx, name_to_meta_idx


def goals_to_multi_hot(
    goals_batch: Sequence[Sequence[str]],
    goal_name_to_col: Dict[str, int],
) -> Tuple[np.ndarray, int]:
    y = np.zeros((len(goals_batch), len(goal_name_to_col)), dtype=np.float32)
    unknown = 0
    for i, goals in enumerate(goals_batch):
        for goal in goals:
            goal_name = str(goal)
            col = goal_name_to_col.get(goal_name)
            if col is None:
                unknown += 1
                continue
            y[i, col] = 1.0
    return y, unknown


@torch.no_grad()
def extract_last_mu_dataset(
    model: GraphSequenceVAE,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    mu_last_chunks: List[torch.Tensor] = []
    names: List[str] = []
    task_names: List[str] = []
    goals: List[List[str]] = []

    for batch in loader:
        names.extend([str(x) for x in batch["names"]])
        task_names.extend([str(x) for x in batch["task_names"]])
        goals.extend([list(g) for g in batch["goals"]])

        batch = move_batch_to_device(batch, device)
        _, mu, _ = model.encode_sequence(
            class_ids=batch["class_ids"],
            coords=batch["coords"],
            states=batch["states"],
            node_mask=batch["node_mask"],
            lengths=batch["lengths"],
            action_ids=batch.get("action_ids"),
        )

        bsz = mu.shape[0]
        last_idx = (batch["lengths"] - 1).clamp(min=0)
        gather_idx = torch.arange(bsz, device=mu.device)
        mu_last = mu[gather_idx, last_idx]
        mu_last_chunks.append(mu_last.detach().cpu())

    if len(mu_last_chunks) == 0:
        raise RuntimeError("No batches extracted from loader.")

    mu_last_all = torch.cat(mu_last_chunks, dim=0).numpy()
    return {
        "mu_last": mu_last_all,
        "names": names,
        "task_names": task_names,
        "goals": goals,
    }


def fit_2d_embedding(
    x: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], str]:
    if args.reducer == "none":
        return None, "none"

    reducers_to_try = [args.reducer] if args.reducer != "auto" else ["umap", "tsne", "pca"]

    last_error = None
    for reducer_name in reducers_to_try:
        try:
            if reducer_name == "umap":
                try:
                    import umap  # type: ignore
                except Exception:
                    import umap.umap_ as umap  # type: ignore

                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=args.umap_n_neighbors,
                    min_dist=args.umap_min_dist,
                    metric=args.umap_metric,
                    random_state=args.seed,
                )
                return reducer.fit_transform(x), "umap"

            if reducer_name == "tsne":
                from sklearn.manifold import TSNE  # type: ignore

                perplexity = min(args.tsne_perplexity, max(5.0, float(x.shape[0] - 1) / 3.0))
                reducer = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=args.seed,
                    init="pca",
                    learning_rate="auto",
                )
                return reducer.fit_transform(x), "tsne"

            if reducer_name == "pca":
                from sklearn.decomposition import PCA  # type: ignore

                reducer = PCA(n_components=2, random_state=args.seed)
                return reducer.fit_transform(x), "pca"
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Could not fit any reducer (requested {}). Last error: {}".format(args.reducer, repr(last_error))
    )


def _safe_corr_binary(x: np.ndarray, y_binary: np.ndarray) -> float:
    x_center = x - x.mean()
    y_center = y_binary - y_binary.mean()
    denom = np.sqrt((x_center ** 2).sum() * (y_center ** 2).sum())
    if denom <= 1e-12:
        return 0.0
    return float((x_center * y_center).sum() / denom)


def predicate_dim_stats(
    mu_last: np.ndarray,
    y: np.ndarray,
    predicate_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summaries: List[Dict[str, object]] = []
    dim_rows: List[Dict[str, object]] = []

    num_dims = mu_last.shape[1]
    for pred_idx, pred_name in enumerate(predicate_names):
        target = y[:, pred_idx]
        pos_mask = target > 0.5
        neg_mask = ~pos_mask
        pos_count = int(pos_mask.sum())
        neg_count = int(neg_mask.sum())
        prevalence = float(target.mean())

        if pos_count == 0 or neg_count == 0:
            summaries.append(
                {
                    "predicate": pred_name,
                    "count": pos_count,
                    "prevalence": prevalence,
                    "top_dim": None,
                    "top_dim_abs_cohen_d": 0.0,
                    "top_dim_abs_corr": 0.0,
                }
            )
            continue

        best_dim = -1
        best_abs_d = -1.0
        best_abs_corr = -1.0

        for dim in range(num_dims):
            x = mu_last[:, dim]
            x_pos = x[pos_mask]
            x_neg = x[neg_mask]
            mean_pos = float(x_pos.mean())
            mean_neg = float(x_neg.mean())
            std_pos = float(x_pos.std(ddof=0))
            std_neg = float(x_neg.std(ddof=0))
            pooled = np.sqrt((std_pos ** 2 + std_neg ** 2) / 2.0)
            cohen_d = 0.0 if pooled <= 1e-12 else float((mean_pos - mean_neg) / pooled)
            corr = _safe_corr_binary(x.astype(np.float64), target.astype(np.float64))

            dim_rows.append(
                {
                    "predicate": pred_name,
                    "predicate_index": pred_idx,
                    "dim": dim,
                    "count_pos": pos_count,
                    "count_neg": neg_count,
                    "prevalence": prevalence,
                    "mean_pos": mean_pos,
                    "mean_neg": mean_neg,
                    "std_pos": std_pos,
                    "std_neg": std_neg,
                    "mean_diff": mean_pos - mean_neg,
                    "cohen_d": cohen_d,
                    "corr_binary": corr,
                    "abs_cohen_d": abs(cohen_d),
                    "abs_corr_binary": abs(corr),
                }
            )

            if abs(cohen_d) > best_abs_d:
                best_abs_d = abs(cohen_d)
                best_dim = dim
            if abs(corr) > best_abs_corr:
                best_abs_corr = abs(corr)

        summaries.append(
            {
                "predicate": pred_name,
                "count": pos_count,
                "prevalence": prevalence,
                "top_dim": best_dim,
                "top_dim_abs_cohen_d": float(best_abs_d),
                "top_dim_abs_corr": float(best_abs_corr),
            }
        )

    summaries.sort(key=lambda row: (-float(row["top_dim_abs_cohen_d"]), -int(row["count"])))
    return summaries, dim_rows


def save_json(path: str, obj: object) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if len(rows) == 0:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pick_predicates_to_plot(
    predicate_names: Sequence[str],
    y: np.ndarray,
    args: argparse.Namespace,
) -> List[int]:
    explicit = [x.strip() for x in args.predicate_list.split(",") if x.strip()]
    name_to_idx = {name: i for i, name in enumerate(predicate_names)}
    selected: List[int] = []

    for name in explicit:
        idx = name_to_idx.get(name)
        if idx is not None:
            selected.append(idx)

    if len(selected) > 0:
        return selected

    counts = y.sum(axis=0)
    candidates = [
        (int(i), int(counts[i]))
        for i in range(len(predicate_names))
        if int(counts[i]) >= args.min_predicate_count
    ]
    candidates.sort(key=lambda x: (-x[1], predicate_names[x[0]]))
    return [idx for idx, _ in candidates[: args.plot_top_k_predicates]]


def _scatter_base(ax, coords_2d: np.ndarray) -> None:
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(alpha=0.2)
    if coords_2d is None:
        ax.text(0.5, 0.5, "No 2D embedding", ha="center", va="center")


def make_task_scatter(coords_2d: np.ndarray, task_names: Sequence[str], max_task_colors: int):
    if plt is None or coords_2d is None:
        return None

    task_counts: Dict[str, int] = {}
    for t in task_names:
        task_counts[t] = task_counts.get(t, 0) + 1

    ranked = sorted(task_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_tasks = [k for k, _ in ranked[:max_task_colors]]
    top_set = set(top_tasks)

    fig, ax = plt.subplots(figsize=(8, 7))
    # Background: all points in light gray.
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=8, alpha=0.15, c="lightgray", edgecolors="none")

    cmap = plt.get_cmap("tab10")
    for i, task in enumerate(top_tasks):
        mask = np.array([t == task for t in task_names], dtype=bool)
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=10,
            alpha=0.7,
            c=[cmap(i % 10)],
            edgecolors="none",
            label="{} ({})".format(task, int(mask.sum())),
        )

    other_count = sum(1 for t in task_names if t not in top_set)
    ax.set_title("Last mu latent embedding colored by task (top {} + other={})".format(len(top_tasks), other_count))
    _scatter_base(ax, coords_2d)
    if len(top_tasks) > 0:
        ax.legend(loc="best", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return fig


def make_predicate_scatter(coords_2d: np.ndarray, y_col: np.ndarray, predicate_name: str):
    if plt is None or coords_2d is None:
        return None

    pos_mask = y_col > 0.5
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        coords_2d[neg_mask, 0],
        coords_2d[neg_mask, 1],
        s=8,
        alpha=0.15,
        c="lightgray",
        edgecolors="none",
        label="absent ({})".format(int(neg_mask.sum())),
    )
    ax.scatter(
        coords_2d[pos_mask, 0],
        coords_2d[pos_mask, 1],
        s=12,
        alpha=0.8,
        c="#d62728",
        edgecolors="none",
        label="present ({})".format(int(pos_mask.sum())),
    )
    ax.set_title("Last mu latent embedding | predicate presence: {}".format(predicate_name))
    _scatter_base(ax, coords_2d)
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    metadata_path = args.metadata or ckpt_args.get("metadata")
    data_json = args.data_json or ckpt_args.get("val_json") or ckpt_args.get("train_json")
    split_key = args.split_key or ckpt_args.get("val_split_key") or ckpt_args.get("train_split_key")

    if metadata_path is None or data_json is None or split_key is None:
        raise ValueError(
            "Could not infer metadata/data-json/split-key from checkpoint. Pass --metadata --data-json --split-key explicitly."
        )

    model = GraphSequenceVAE.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tensorizer = WatchGraphTensorizer.from_config(checkpoint["tensorizer_config"])
    dataset = WatchVAEDataset(
        data_path=data_json,
        split_key=split_key,
        tensorizer=tensorizer,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        use_actions=bool(checkpoint["model_config"].get("use_actions", False)),
        max_demos=args.max_demos,
        stable_slots=args.stable_slots,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_watch_vae,
        drop_last=False,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_dir = os.path.join(ckpt_dir, "latent_analysis_{}_{}".format(ckpt_base, timestamp))
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("checkpoint:", args.checkpoint)
    print("metadata:", metadata_path)
    print("data_json:", data_json)
    print("split_key:", split_key)
    print("demos:", len(dataset))
    print("stable_slots:", args.stable_slots)
    print("output_dir:", output_dir)

    extracted = extract_last_mu_dataset(model=model, loader=loader, device=device)
    mu_last = extracted["mu_last"]
    names = extracted["names"]
    task_names = extracted["task_names"]
    goals = extracted["goals"]

    predicate_names, goal_name_to_col, _ = load_metadata_goal_vocab(metadata_path)
    y, unknown_goal_count = goals_to_multi_hot(goals, goal_name_to_col)

    coords_2d, reducer_used = fit_2d_embedding(mu_last, args)
    print("reducer_used:", reducer_used)
    print("last_mu shape:", list(mu_last.shape))
    print("goal_presence shape:", list(y.shape))
    print("unknown goal labels skipped:", unknown_goal_count)

    mu_dim_mean = mu_last.mean(axis=0)
    mu_dim_std = mu_last.std(axis=0)
    active_dim_threshold = 0.05
    active_dims = int((mu_dim_std > active_dim_threshold).sum())

    summary = {
        "checkpoint": args.checkpoint,
        "metadata_path": metadata_path,
        "data_json": data_json,
        "split_key": split_key,
        "num_demos": len(dataset),
        "num_points": int(mu_last.shape[0]),
        "latent_dim": int(mu_last.shape[1]),
        "stable_slots": bool(args.stable_slots),
        "reducer_requested": args.reducer,
        "reducer_used": reducer_used,
        "unknown_goal_labels_skipped": int(unknown_goal_count),
        "mu_last_mean": float(mu_last.mean()),
        "mu_last_std": float(mu_last.std()),
        "mu_last_abs_mean": float(np.abs(mu_last).mean()),
        "active_dim_threshold": active_dim_threshold,
        "active_dims": active_dims,
        "top_dim_std": [
            {"dim": int(i), "std": float(mu_dim_std[i]), "mean": float(mu_dim_mean[i])}
            for i in np.argsort(-mu_dim_std)[: min(10, len(mu_dim_std))]
        ],
    }
    save_json(os.path.join(output_dir, "summary.json"), summary)

    predicate_summary_rows, predicate_dim_rows = predicate_dim_stats(mu_last, y, predicate_names)
    save_csv(os.path.join(output_dir, "predicate_dim_stats.csv"), predicate_dim_rows)
    save_csv(os.path.join(output_dir, "predicate_summary.csv"), predicate_summary_rows)
    save_json(os.path.join(output_dir, "predicate_summary_top20.json"), predicate_summary_rows[:20])

    np.savez_compressed(
        os.path.join(output_dir, "latent_last_mu_data.npz"),
        mu_last=mu_last.astype(np.float32),
        embedding_2d=np.zeros((mu_last.shape[0], 2), dtype=np.float32) if coords_2d is None else coords_2d.astype(np.float32),
        goal_presence=y.astype(np.float32),
        task_names=np.array(task_names, dtype=object),
        names=np.array(names, dtype=object),
        predicate_names=np.array(predicate_names, dtype=object),
    )

    save_json(
        os.path.join(output_dir, "analysis_config.json"),
        {
            "args": vars(args),
            "checkpoint_args": ckpt_args,
        },
    )

    if plt is None:
        print("matplotlib not available; skipping plots.")
        return

    if coords_2d is not None:
        task_fig = make_task_scatter(coords_2d, task_names, max_task_colors=args.max_task_colors)
        if task_fig is not None:
            task_fig.savefig(os.path.join(plots_dir, "last_mu_{}_task_scatter.png".format(reducer_used)), dpi=160)
            plt.close(task_fig)

        predicate_indices = _pick_predicates_to_plot(predicate_names, y, args)
        for pred_idx in predicate_indices:
            pred_name = predicate_names[pred_idx]
            fig = make_predicate_scatter(coords_2d, y[:, pred_idx], pred_name)
            if fig is not None:
                safe_name = pred_name.replace("/", "_")
                fig.savefig(
                    os.path.join(plots_dir, "last_mu_{}_predicate_{}.png".format(reducer_used, safe_name)),
                    dpi=160,
                )
                plt.close(fig)

    # Dim std bar plot.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(len(mu_dim_std)), mu_dim_std, color="#1f77b4", alpha=0.85)
    ax.axhline(active_dim_threshold, color="#d62728", linestyle="--", linewidth=1.2, label="active threshold")
    ax.set_title("Last mu per-dimension std (active dims: {}/{})".format(active_dims, len(mu_dim_std)))
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("std")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "last_mu_dim_std.png"), dpi=160)
    plt.close(fig)

    # Predicate summary top-k plot by effect size.
    top_k = min(15, len(predicate_summary_rows))
    rows = predicate_summary_rows[:top_k]
    if top_k > 0:
        fig, ax = plt.subplots(figsize=(12, 0.45 * top_k + 2.5))
        vals = [float(r["top_dim_abs_cohen_d"]) for r in rows]
        labels = ["{} (n={})".format(r["predicate"], int(r["count"])) for r in rows]
        ax.barh(np.arange(top_k), vals, color="#2ca02c", alpha=0.85)
        ax.set_yticks(np.arange(top_k))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("max |Cohen's d| across latent dims")
        ax.set_title("Goal predicate vs last-mu dimension association (top {})".format(top_k))
        ax.grid(alpha=0.2, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "predicate_top_effects.png"), dpi=160)
        plt.close(fig)

    print("Saved analysis to:", output_dir)
    print("Key files:")
    print(" -", os.path.join(output_dir, "summary.json"))
    print(" -", os.path.join(output_dir, "predicate_summary.csv"))
    print(" -", os.path.join(output_dir, "predicate_dim_stats.csv"))
    print(" -", os.path.join(output_dir, "latent_last_mu_data.npz"))
    print(" -", plots_dir)


if __name__ == "__main__":
    main()
