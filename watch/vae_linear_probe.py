import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from watch.vae.dataset import WatchVAEDataset, collate_watch_vae
from watch.vae.model import GraphSequenceVAE
from watch.vae.tensorizer import WatchGraphTensorizer
from watch.vae_latent_analysis import (
    extract_last_mu_dataset,
    goals_to_multi_hot,
    load_metadata_goal_vocab,
    set_seed,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
except Exception as exc:
    raise ImportError(
        "scikit-learn is required for watch/vae_linear_probe.py. "
        "Install it in the active env (vh38) and rerun."
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate linear probes on Watch VAE last-mu latents.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to watch_vae_{best,last}.pt checkpoint")
    parser.add_argument("--metadata", type=str, default=None, help="Override metadata path (defaults to checkpoint args)")
    parser.add_argument("--train-json", type=str, default=None, help="Override train json (defaults to checkpoint args)")
    parser.add_argument("--val-json", type=str, default=None, help="Override val json (defaults to checkpoint args)")
    parser.add_argument("--train-split-key", type=str, default=None)
    parser.add_argument("--val-split-key", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-demos", type=int, default=4000)
    parser.add_argument("--max-val-demos", type=int, default=1500)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stable-slots", action="store_true", default=True)
    parser.add_argument("--no-stable-slots", action="store_false", dest="stable_slots")

    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization for LogisticRegression")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--class-weight-balanced", action="store_true", default=True)
    parser.add_argument("--no-class-weight-balanced", action="store_false", dest="class_weight_balanced")
    parser.add_argument("--min-train-positives", type=int, default=10)
    parser.add_argument("--min-val-positives", type=int, default=5)
    parser.add_argument("--min-val-negatives", type=int, default=5)
    parser.add_argument("--top-k-dims", type=int, default=5)
    parser.add_argument("--top-k-report", type=int, default=15)

    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _infer_paths_from_checkpoint(args: argparse.Namespace, checkpoint: Dict[str, object]) -> Dict[str, str]:
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    metadata = args.metadata or ckpt_args.get("metadata")
    train_json = args.train_json or ckpt_args.get("train_json")
    val_json = args.val_json or ckpt_args.get("val_json") or train_json
    train_split_key = args.train_split_key or ckpt_args.get("train_split_key", "train_data")
    val_split_key = args.val_split_key or ckpt_args.get("val_split_key", "test_data")

    missing = [k for k, v in {
        "metadata": metadata,
        "train_json": train_json,
        "val_json": val_json,
        "train_split_key": train_split_key,
        "val_split_key": val_split_key,
    }.items() if v is None]
    if missing:
        raise ValueError(
            "Could not infer required inputs from checkpoint. Missing: {}. Pass them explicitly.".format(", ".join(missing))
        )

    return {
        "metadata": str(metadata),
        "train_json": str(train_json),
        "val_json": str(val_json),
        "train_split_key": str(train_split_key),
        "val_split_key": str(val_split_key),
    }


def _make_loader(
    data_path: str,
    split_key: str,
    tensorizer: WatchGraphTensorizer,
    use_actions: bool,
    batch_size: int,
    num_workers: int,
    max_demos: int,
    max_seq_len: int,
    min_seq_len: int,
    stable_slots: bool,
) -> Tuple[WatchVAEDataset, DataLoader]:
    dataset = WatchVAEDataset(
        data_path=data_path,
        split_key=split_key,
        tensorizer=tensorizer,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        use_actions=use_actions,
        max_demos=max_demos,
        stable_slots=stable_slots,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_watch_vae,
        drop_last=False,
    )
    return dataset, loader


def _standardize(train_x: np.ndarray, val_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std, mean[0], std[0]


def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if len(rows) == 0:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fit_probe_rows(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    predicate_names: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    rows: List[Dict[str, object]] = []
    val_scores = np.full((val_y.shape[0], val_y.shape[1]), np.nan, dtype=np.float32)

    for pred_idx, pred_name in enumerate(predicate_names):
        y_tr = train_y[:, pred_idx].astype(np.int64)
        y_va = val_y[:, pred_idx].astype(np.int64)

        pos_tr = int(y_tr.sum())
        neg_tr = int((1 - y_tr).sum())
        pos_va = int(y_va.sum())
        neg_va = int((1 - y_va).sum())

        row: Dict[str, object] = {
            "predicate": pred_name,
            "predicate_index": pred_idx,
            "train_pos": pos_tr,
            "train_neg": neg_tr,
            "val_pos": pos_va,
            "val_neg": neg_va,
            "train_prevalence": _safe_float(y_tr.mean()),
            "val_prevalence": _safe_float(y_va.mean()),
            "status": "ok",
            "auroc": float("nan"),
            "ap": float("nan"),
            "f1_0p5": float("nan"),
            "precision_0p5": float("nan"),
            "recall_0p5": float("nan"),
            "coef_l2_norm": float("nan"),
            "intercept": float("nan"),
        }

        if pos_tr < args.min_train_positives or neg_tr < args.min_train_positives:
            row["status"] = "skip_train_imbalance"
            rows.append(row)
            continue
        if pos_va < args.min_val_positives or neg_va < args.min_val_negatives:
            row["status"] = "skip_val_imbalance"
            rows.append(row)
            continue

        try:
            clf = LogisticRegression(
                C=args.c,
                penalty="l2",
                solver="liblinear",
                max_iter=args.max_iter,
                class_weight="balanced" if args.class_weight_balanced else None,
                random_state=args.seed,
            )
            clf.fit(train_x, y_tr)

            if hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(val_x)[:, 1]
            else:
                logits = clf.decision_function(val_x)
                scores = 1.0 / (1.0 + np.exp(-logits))

            preds = (scores >= 0.5).astype(np.int64)
            val_scores[:, pred_idx] = scores.astype(np.float32)

            row["auroc"] = _safe_float(roc_auc_score(y_va, scores))
            row["ap"] = _safe_float(average_precision_score(y_va, scores))
            row["f1_0p5"] = _safe_float(f1_score(y_va, preds, zero_division=0))
            tp = int(((preds == 1) & (y_va == 1)).sum())
            fp = int(((preds == 1) & (y_va == 0)).sum())
            fn = int(((preds == 0) & (y_va == 1)).sum())
            row["precision_0p5"] = _safe_float(tp / max(1, tp + fp))
            row["recall_0p5"] = _safe_float(tp / max(1, tp + fn))

            coef = clf.coef_[0].astype(np.float64)
            abs_order = np.argsort(-np.abs(coef))
            row["coef_l2_norm"] = _safe_float(np.linalg.norm(coef))
            row["intercept"] = _safe_float(clf.intercept_[0]) if np.ndim(clf.intercept_) > 0 else _safe_float(clf.intercept_)
            for k in range(min(args.top_k_dims, coef.shape[0])):
                dim = int(abs_order[k])
                row["top_dim_{}".format(k + 1)] = dim
                row["top_coef_{}".format(k + 1)] = _safe_float(coef[dim])
                row["top_abscoef_{}".format(k + 1)] = _safe_float(abs(coef[dim]))
        except Exception as exc:
            row["status"] = "fit_error"
            row["error"] = repr(exc)

        rows.append(row)

    return rows, val_scores


def _summarize_probe_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    valid_rows = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("auroc", np.nan)) and np.isfinite(r.get("ap", np.nan))]
    summary: Dict[str, object] = {
        "num_predicates_total": len(rows),
        "num_predicates_valid": len(valid_rows),
        "num_predicates_skipped": len(rows) - len(valid_rows),
    }
    if len(valid_rows) == 0:
        return summary

    aurocs = np.array([float(r["auroc"]) for r in valid_rows], dtype=np.float64)
    aps = np.array([float(r["ap"]) for r in valid_rows], dtype=np.float64)
    f1s = np.array([float(r["f1_0p5"]) for r in valid_rows], dtype=np.float64)
    weights = np.array([max(1, int(r["val_pos"])) for r in valid_rows], dtype=np.float64)
    weights = weights / weights.sum()

    summary.update(
        {
            "macro_auroc": float(np.mean(aurocs)),
            "macro_ap": float(np.mean(aps)),
            "macro_f1_0p5": float(np.mean(f1s)),
            "weighted_auroc_by_val_pos": float(np.sum(weights * aurocs)),
            "weighted_ap_by_val_pos": float(np.sum(weights * aps)),
            "weighted_f1_by_val_pos": float(np.sum(weights * f1s)),
            "median_auroc": float(np.median(aurocs)),
            "median_ap": float(np.median(aps)),
            "mean_val_pos": float(np.mean([float(r["val_pos"]) for r in valid_rows])),
        }
    )
    return summary


def _make_probe_plots(rows: List[Dict[str, object]], plots_dir: str, top_k: int) -> None:
    if plt is None:
        return
    valid_rows = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("auroc", np.nan)) and np.isfinite(r.get("ap", np.nan))]
    if len(valid_rows) == 0:
        return

    by_ap = sorted(valid_rows, key=lambda r: (-float(r["ap"]), -float(r["auroc"])))[:top_k]
    by_auroc = sorted(valid_rows, key=lambda r: (-float(r["auroc"]), -float(r["ap"])))[:top_k]

    for metric_name, subset in [("ap", by_ap), ("auroc", by_auroc)]:
        fig, ax = plt.subplots(figsize=(12, 0.45 * len(subset) + 2.0))
        values = [float(r[metric_name]) for r in subset]
        labels = ["{} (n={})".format(r["predicate"], int(r["val_pos"])) for r in subset]
        ax.barh(np.arange(len(subset)), values, color="#1f77b4", alpha=0.85)
        ax.set_yticks(np.arange(len(subset)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(metric_name.upper())
        ax.set_title("Linear probe top predicates by {}".format(metric_name.upper()))
        ax.grid(alpha=0.2, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "probe_top_{}.png".format(metric_name)), dpi=160)
        plt.close(fig)

    # Scatter: prevalence vs AP
    fig, ax = plt.subplots(figsize=(7, 6))
    prevalences = np.array([float(r["val_prevalence"]) for r in valid_rows], dtype=np.float64)
    aps = np.array([float(r["ap"]) for r in valid_rows], dtype=np.float64)
    aurocs = np.array([float(r["auroc"]) for r in valid_rows], dtype=np.float64)
    ax.scatter(prevalences, aps, s=25, alpha=0.7, c=aurocs, cmap="viridis", edgecolors="none")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", alpha=0.4)
    ax.set_xlabel("Val prevalence")
    ax.set_ylabel("Average Precision")
    ax.set_title("Linear probe predicate AP vs prevalence (color=AUROC)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "probe_ap_vs_prevalence.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    paths = _infer_paths_from_checkpoint(args, checkpoint)
    model = GraphSequenceVAE.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tensorizer = WatchGraphTensorizer.from_config(checkpoint["tensorizer_config"])

    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt_base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(ckpt_dir, "probe_analysis_{}_{}".format(ckpt_base, timestamp))
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    use_actions = bool(checkpoint["model_config"].get("use_actions", False))

    train_dataset, train_loader = _make_loader(
        data_path=paths["train_json"],
        split_key=paths["train_split_key"],
        tensorizer=tensorizer,
        use_actions=use_actions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_demos=args.max_train_demos,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        stable_slots=args.stable_slots,
    )
    val_dataset, val_loader = _make_loader(
        data_path=paths["val_json"],
        split_key=paths["val_split_key"],
        tensorizer=tensorizer,
        use_actions=use_actions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_demos=args.max_val_demos,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        stable_slots=args.stable_slots,
    )

    print("checkpoint:", args.checkpoint)
    print("output_dir:", output_dir)
    print("train dataset:", paths["train_json"], paths["train_split_key"], "demos=", len(train_dataset))
    print("val dataset:", paths["val_json"], paths["val_split_key"], "demos=", len(val_dataset))
    print("stable slots:", args.stable_slots)

    train_extracted = extract_last_mu_dataset(model=model, loader=train_loader, device=device)
    val_extracted = extract_last_mu_dataset(model=model, loader=val_loader, device=device)
    train_mu_last = np.asarray(train_extracted["mu_last"], dtype=np.float32)
    val_mu_last = np.asarray(val_extracted["mu_last"], dtype=np.float32)

    predicate_names, goal_name_to_col, _ = load_metadata_goal_vocab(paths["metadata"])
    train_y, train_unknown = goals_to_multi_hot(train_extracted["goals"], goal_name_to_col)
    val_y, val_unknown = goals_to_multi_hot(val_extracted["goals"], goal_name_to_col)

    train_x, val_x, feat_mean, feat_std = _standardize(train_mu_last, val_mu_last)
    rows, val_scores = _fit_probe_rows(train_x, val_x, train_y, val_y, predicate_names, args)
    summary = _summarize_probe_rows(rows)

    summary.update(
        {
            "checkpoint": args.checkpoint,
            "metadata": paths["metadata"],
            "train_json": paths["train_json"],
            "val_json": paths["val_json"],
            "train_split_key": paths["train_split_key"],
            "val_split_key": paths["val_split_key"],
            "train_num_demos": len(train_dataset),
            "val_num_demos": len(val_dataset),
            "latent_dim": int(train_mu_last.shape[1]),
            "stable_slots": bool(args.stable_slots),
            "train_unknown_goals_skipped": int(train_unknown),
            "val_unknown_goals_skipped": int(val_unknown),
            "train_mu_abs_mean": float(np.abs(train_mu_last).mean()),
            "val_mu_abs_mean": float(np.abs(val_mu_last).mean()),
        }
    )

    valid_rows = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("ap", np.nan))]
    top_by_ap = sorted(valid_rows, key=lambda r: (-float(r["ap"]), -float(r["auroc"])))[: args.top_k_report]
    top_by_auroc = sorted(valid_rows, key=lambda r: (-float(r["auroc"]), -float(r["ap"])))[: args.top_k_report]

    _save_json(os.path.join(output_dir, "probe_summary.json"), summary)
    _save_json(os.path.join(output_dir, "probe_top_by_ap.json"), top_by_ap)
    _save_json(os.path.join(output_dir, "probe_top_by_auroc.json"), top_by_auroc)
    _save_csv(os.path.join(output_dir, "predicate_probe_metrics.csv"), rows)
    _save_json(
        os.path.join(output_dir, "probe_config.json"),
        {
            "args": vars(args),
            "checkpoint_args": checkpoint.get("args", {}),
        },
    )
    np.savez_compressed(
        os.path.join(output_dir, "probe_predictions_val.npz"),
        val_scores=val_scores.astype(np.float32),
        val_labels=val_y.astype(np.float32),
        val_mu_last=val_mu_last.astype(np.float32),
        train_mu_last=train_mu_last.astype(np.float32),
        predicate_names=np.array(predicate_names, dtype=object),
        val_names=np.array(val_extracted["names"], dtype=object),
        val_task_names=np.array(val_extracted["task_names"], dtype=object),
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
    )

    _make_probe_plots(rows, plots_dir=plots_dir, top_k=args.top_k_report)

    print("Valid predicates:", summary.get("num_predicates_valid"), "/", summary.get("num_predicates_total"))
    if "macro_auroc" in summary:
        print(
            "macro AUROC={:.4f}  macro AP={:.4f}  macro F1@0.5={:.4f}".format(
                float(summary["macro_auroc"]),
                float(summary["macro_ap"]),
                float(summary["macro_f1_0p5"]),
            )
        )
    print("Saved probe analysis to:", output_dir)
    print("Key files:")
    print(" -", os.path.join(output_dir, "probe_summary.json"))
    print(" -", os.path.join(output_dir, "predicate_probe_metrics.csv"))
    print(" -", os.path.join(output_dir, "probe_top_by_ap.json"))
    print(" -", os.path.join(output_dir, "probe_top_by_auroc.json"))
    print(" -", os.path.join(output_dir, "probe_predictions_val.npz"))
    print(" -", plots_dir)


if __name__ == "__main__":
    main()
