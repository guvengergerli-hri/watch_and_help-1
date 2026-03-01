#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Headless-safe backend; plotting is optional.
os.environ.setdefault("MPLBACKEND", "Agg")


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print("[eval-pik {}] {}".format(ts, msg), flush=True)


def _safe_name(value: Any) -> str:
    raw = str(value)
    cleaned = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    text = "".join(cleaned).strip("._")
    return text or "unnamed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and plot metrics from test result pickles (results_0.pik + logs_agent_*.pik)."
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        required=True,
        help="Directory containing results_0.pik / logs_agent_*.pik (e.g. ../test_results/multiBob_env_task_set_20_hybrid_truegoal).",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Optional explicit results pickle path. If omitted, auto-detect in --record-dir.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional dataset pickle to map episode index -> task_name. Defaults to dataset/test_env_set_help.pik if found.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Episode step horizon used for normalized score.",
    )
    parser.add_argument(
        "--baseline-record-dir",
        type=str,
        default=None,
        help="Optional baseline result directory for speedup/success comparison.",
    )
    parser.add_argument(
        "--baseline-results-file",
        type=str,
        default=None,
        help="Optional explicit baseline results pickle path.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for summary/plots. Default: outputs/test_results_analysis/<record_dir_name>_<timestamp>/",
    )
    parser.add_argument("--skip-plots", action="store_true", default=False, help="Skip PNG plot generation.")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    return parser.parse_args()


def _coerce_episode_key(key: Any) -> Optional[int]:
    try:
        return int(key)
    except Exception:
        return None


def _to_float_maybe(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_success_maybe(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        v = float(value)
        return 1.0 if v > 0.5 else 0.0
    except Exception:
        return None


def _detect_results_path(record_dir: Path) -> Path:
    candidates = [
        record_dir / "results_0.pik",
        record_dir / "results.pkl",
        record_dir / "results.pik",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No results pickle found in {} (checked: {}).".format(record_dir, candidates))


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_results_dict(path: Path) -> Dict[int, Dict[str, List[Any]]]:
    payload = _load_pickle(path)
    out: Dict[int, Dict[str, List[Any]]] = {}

    if isinstance(payload, dict):
        for key, value in payload.items():
            ep_id = _coerce_episode_key(key)
            if ep_id is None or not isinstance(value, dict):
                continue
            s = value.get("S", [])
            l = value.get("L", [])
            if not isinstance(s, list):
                s = [s]
            if not isinstance(l, list):
                l = [l]
            out[ep_id] = {"S": s, "L": l}
        return out

    if isinstance(payload, list):
        # Best-effort compatibility for older formats.
        for ep_id, value in enumerate(payload):
            if not isinstance(value, dict):
                continue
            s = value.get("S", [])
            l = value.get("L", [])
            if not isinstance(s, list):
                s = [s]
            if not isinstance(l, list):
                l = [l]
            out[int(ep_id)] = {"S": s, "L": l}
        if len(out) > 0:
            return out

    raise TypeError("Unsupported results payload type at {}: {}".format(path, type(payload).__name__))


def _load_dataset_meta(dataset_path: Optional[Path], num_episodes_hint: int) -> Dict[int, Dict[str, Any]]:
    meta: Dict[int, Dict[str, Any]] = {}
    if dataset_path is None or not dataset_path.is_file():
        return meta

    try:
        payload = _load_pickle(dataset_path)
    except Exception as e:
        _log("Dataset load failed at {}: {}".format(dataset_path, repr(e)))
        return meta

    if not isinstance(payload, list):
        return meta

    for ep_id, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        meta[ep_id] = {
            "task_name": item.get("task_name"),
            "task_id": item.get("task_id"),
            "env_id": item.get("env_id"),
        }

    if num_episodes_hint > 0 and len(meta) > 0 and len(meta) < num_episodes_hint:
        _log(
            "Dataset has fewer entries than results episodes (dataset={}, results~={}); task mapping may be partial.".format(
                len(meta), num_episodes_hint
            )
        )
    return meta


def _parse_log_filename(path: Path) -> Optional[Dict[str, Any]]:
    # Expected: logs_agent_<task_id>_<task_name>_<seed>.pik
    stem = path.stem
    if not stem.startswith("logs_agent_"):
        return None
    body = stem[len("logs_agent_") :]
    parts = body.split("_")
    if len(parts) < 3:
        return None
    try:
        task_id = int(parts[0])
        seed = int(parts[-1])
    except Exception:
        return None
    task_name = "_".join(parts[1:-1]).strip()
    if len(task_name) == 0:
        return None
    return {"task_id": task_id, "task_name": task_name, "seed": seed}


def _build_task_meta_fallback_from_logs(record_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    Fallback only when dataset mapping is unavailable.
    Assumes episode_id ~= task_id (true for current test_hybrid outputs).
    """
    out: Dict[int, Dict[str, Any]] = {}
    for log_path in sorted(record_dir.glob("logs_agent_*.pik")):
        parsed = _parse_log_filename(log_path)
        if parsed is None:
            continue
        ep_guess = int(parsed["task_id"])
        if ep_guess not in out:
            out[ep_guess] = {
                "task_name": parsed["task_name"],
                "task_id": parsed["task_id"],
                "env_id": None,
            }
    return out


def _aggregate_results(
    results: Dict[int, Dict[str, List[Any]]],
    task_meta: Dict[int, Dict[str, Any]],
    max_steps: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_episode_rows: List[Dict[str, Any]] = []
    all_trial_successes: List[float] = []
    all_trial_lengths: List[float] = []
    all_trial_scores: List[float] = []

    task_buckets: Dict[str, Dict[str, Any]] = {}
    tries_hist: Dict[int, int] = {}

    for ep_id in sorted(results.keys()):
        entry = results[ep_id]
        s_raw = entry.get("S", [])
        l_raw = entry.get("L", [])
        n = min(len(s_raw), len(l_raw))
        trial_successes: List[float] = []
        trial_lengths: List[float] = []
        trial_scores: List[float] = []

        for idx in range(n):
            s = _to_success_maybe(s_raw[idx])
            l = _to_float_maybe(l_raw[idx])
            if s is None or l is None:
                continue
            trial_successes.append(float(s))
            trial_lengths.append(float(l))
            trial_scores.append(float(s - (l / float(max_steps))))

        if len(trial_successes) == 0:
            continue

        all_trial_successes.extend(trial_successes)
        all_trial_lengths.extend(trial_lengths)
        all_trial_scores.extend(trial_scores)

        tries_count = len(trial_successes)
        tries_hist[tries_count] = tries_hist.get(tries_count, 0) + 1

        meta = task_meta.get(ep_id, {})
        task_name = str(meta.get("task_name")) if meta.get("task_name") is not None else "unknown"
        task_id = meta.get("task_id")
        env_id = meta.get("env_id")

        row = {
            "episode_id": ep_id,
            "task_name": task_name,
            "task_id": task_id,
            "env_id": env_id,
            "tries": tries_count,
            "success_rate": sum(trial_successes) / float(tries_count),
            "mean_length": sum(trial_lengths) / float(tries_count),
            "median_length": statistics.median(trial_lengths),
            "mean_norm_score": sum(trial_scores) / float(tries_count),
            "trial_successes": trial_successes,
            "trial_lengths": trial_lengths,
        }
        per_episode_rows.append(row)

        if task_name not in task_buckets:
            task_buckets[task_name] = {
                "task_name": task_name,
                "episodes": 0,
                "tries": 0,
                "success_sum": 0.0,
                "length_sum": 0.0,
                "score_sum": 0.0,
            }
        bucket = task_buckets[task_name]
        bucket["episodes"] += 1
        bucket["tries"] += tries_count
        bucket["success_sum"] += sum(trial_successes)
        bucket["length_sum"] += sum(trial_lengths)
        bucket["score_sum"] += sum(trial_scores)

    per_task_rows: List[Dict[str, Any]] = []
    for task_name in sorted(task_buckets.keys()):
        bucket = task_buckets[task_name]
        tries = int(bucket["tries"])
        if tries <= 0:
            continue
        per_task_rows.append(
            {
                "task_name": task_name,
                "episodes": int(bucket["episodes"]),
                "tries": tries,
                "success_rate": float(bucket["success_sum"] / float(tries)),
                "mean_length": float(bucket["length_sum"] / float(tries)),
                "mean_norm_score": float(bucket["score_sum"] / float(tries)),
            }
        )

    episodes = len(per_episode_rows)
    trials = len(all_trial_successes)
    if trials == 0:
        raise RuntimeError("No valid trials were found in results payload.")

    summary = {
        "episodes_with_data": episodes,
        "total_trials": trials,
        "success_rate_over_trials": float(sum(all_trial_successes) / float(trials)),
        "mean_length_over_trials": float(sum(all_trial_lengths) / float(trials)),
        "median_length_over_trials": float(statistics.median(all_trial_lengths)),
        "mean_norm_score_over_trials": float(sum(all_trial_scores) / float(trials)),
        "success_rate_over_episodes": float(
            sum(row["success_rate"] for row in per_episode_rows) / float(episodes)
        )
        if episodes > 0
        else None,
        "mean_length_over_episodes": float(
            sum(row["mean_length"] for row in per_episode_rows) / float(episodes)
        )
        if episodes > 0
        else None,
        "tries_histogram": {str(k): int(v) for k, v in sorted(tries_hist.items())},
        "num_tasks": len(per_task_rows),
    }

    return summary, per_episode_rows, per_task_rows


def _compare_with_baseline(
    current_per_episode: List[Dict[str, Any]],
    baseline_per_episode: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cur_by_ep = {int(r["episode_id"]): r for r in current_per_episode}
    base_by_ep = {int(r["episode_id"]): r for r in baseline_per_episode}

    shared_eps = sorted(set(cur_by_ep.keys()).intersection(set(base_by_ep.keys())))
    if len(shared_eps) == 0:
        return {
            "shared_episodes": 0,
            "message": "No overlapping episode ids between current and baseline results.",
        }

    speedups: List[float] = []
    delta_successes: List[float] = []
    cur_lengths: List[float] = []
    base_lengths: List[float] = []

    for ep_id in shared_eps:
        cur = cur_by_ep[ep_id]
        base = base_by_ep[ep_id]
        cur_len = float(cur["mean_length"])
        base_len = float(base["mean_length"])
        cur_sr = float(cur["success_rate"])
        base_sr = float(base["success_rate"])

        if base_len > 0:
            speedups.append((base_len - cur_len) / base_len)
        delta_successes.append(cur_sr - base_sr)
        cur_lengths.append(cur_len)
        base_lengths.append(base_len)

    out = {
        "shared_episodes": len(shared_eps),
        "mean_length_current": (sum(cur_lengths) / len(cur_lengths)) if len(cur_lengths) > 0 else None,
        "mean_length_baseline": (sum(base_lengths) / len(base_lengths)) if len(base_lengths) > 0 else None,
        "mean_success_delta": (sum(delta_successes) / len(delta_successes)) if len(delta_successes) > 0 else None,
        "mean_speedup_fraction": (sum(speedups) / len(speedups)) if len(speedups) > 0 else None,
        "median_speedup_fraction": statistics.median(speedups) if len(speedups) > 0 else None,
        "shared_episode_ids": shared_eps,
    }
    if out["mean_length_current"] and out["mean_length_baseline"]:
        out["length_ratio_baseline_over_current"] = out["mean_length_baseline"] / out["mean_length_current"]
    return out


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _plot_metrics(
    out_dir: Path,
    per_episode_rows: List[Dict[str, Any]],
    per_task_rows: List[Dict[str, Any]],
    baseline_per_task_rows: Optional[List[Dict[str, Any]]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        _log("Plotting skipped (matplotlib unavailable): {}".format(repr(e)))
        return

    # Plot 1: tries-per-episode histogram.
    tries = [int(r["tries"]) for r in per_episode_rows]
    if len(tries) > 0:
        counts: Dict[int, int] = {}
        for x in tries:
            counts[x] = counts.get(x, 0) + 1
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.bar([str(x) for x in xs], ys)
        ax.set_title("Tries Per Episode")
        ax.set_xlabel("Recorded tries")
        ax.set_ylabel("Episodes")
        fig.tight_layout()
        fig.savefig(out_dir / "tries_per_episode.png", dpi=140)
        plt.close(fig)

    # Plot 2: trial length distribution.
    lengths: List[float] = []
    for row in per_episode_rows:
        lengths.extend([float(x) for x in row["trial_lengths"]])
    if len(lengths) > 0:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.hist(lengths, bins=25)
        ax.set_title("Trial Length Distribution")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "length_distribution.png", dpi=140)
        plt.close(fig)

    # Plot 3/4: per-task bars.
    if len(per_task_rows) > 0:
        task_names = [str(r["task_name"]) for r in per_task_rows]
        sr_vals = [float(r["success_rate"]) for r in per_task_rows]
        len_vals = [float(r["mean_length"]) for r in per_task_rows]

        fig = plt.figure(figsize=(max(8, 1.2 * len(task_names)), 4.2))
        ax = fig.add_subplot(111)
        ax.bar(task_names, sr_vals)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Success Rate By Task")
        ax.set_xlabel("Task")
        ax.set_ylabel("Success rate")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "success_by_task.png", dpi=140)
        plt.close(fig)

        fig = plt.figure(figsize=(max(8, 1.2 * len(task_names)), 4.2))
        ax = fig.add_subplot(111)
        ax.bar(task_names, len_vals)
        ax.set_title("Mean Length By Task")
        ax.set_xlabel("Task")
        ax.set_ylabel("Steps")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "length_by_task.png", dpi=140)
        plt.close(fig)

    # Plot 5: current vs baseline length by task (if available).
    if baseline_per_task_rows:
        base_by_task = {str(r["task_name"]): r for r in baseline_per_task_rows}
        shared = [r for r in per_task_rows if str(r["task_name"]) in base_by_task]
        if len(shared) > 0:
            names = [str(r["task_name"]) for r in shared]
            cur = [float(r["mean_length"]) for r in shared]
            base = [float(base_by_task[n]["mean_length"]) for n in names]

            fig = plt.figure(figsize=(max(8, 1.4 * len(names)), 4.2))
            ax = fig.add_subplot(111)
            xs = list(range(len(names)))
            width = 0.42
            ax.bar([x - width / 2.0 for x in xs], base, width=width, label="baseline")
            ax.bar([x + width / 2.0 for x in xs], cur, width=width, label="current")
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=20)
            ax.set_title("Mean Length By Task (Baseline vs Current)")
            ax.set_xlabel("Task")
            ax.set_ylabel("Steps")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "length_by_task_vs_baseline.png", dpi=140)
            plt.close(fig)


def _default_dataset_path(repo_root: Path) -> Optional[Path]:
    cand = repo_root / "dataset" / "test_env_set_help.pik"
    if cand.is_file():
        return cand
    return None


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    record_dir = Path(args.record_dir)
    if not record_dir.is_dir():
        raise NotADirectoryError("record-dir does not exist: {}".format(record_dir))

    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = _detect_results_path(record_dir)
    if not results_path.is_file():
        raise FileNotFoundError("Results file not found: {}".format(results_path))

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = repo_root / "outputs" / "test_results_analysis" / (
            _safe_name(record_dir.name) + "_" + time.strftime("%Y%m%d_%H%M%S")
        )
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError("Output dir exists: {} (pass --overwrite)".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path: Optional[Path]
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = _default_dataset_path(repo_root)

    _log("Loading results: {}".format(results_path))
    results = _load_results_dict(results_path)
    _log("Loaded episode entries: {}".format(len(results)))

    task_meta = _load_dataset_meta(dataset_path, len(results))
    if len(task_meta) == 0:
        _log("No dataset mapping found; using logs_agent_* fallback for task names (best-effort).")
        task_meta = _build_task_meta_fallback_from_logs(record_dir)

    summary, per_episode_rows, per_task_rows = _aggregate_results(
        results=results,
        task_meta=task_meta,
        max_steps=int(args.max_steps),
    )

    meta = {
        "record_dir": str(record_dir),
        "results_file": str(results_path),
        "dataset_path": None if dataset_path is None else str(dataset_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_steps": int(args.max_steps),
    }

    _write_json(out_dir / "summary.json", {"meta": meta, "summary": summary})
    _write_csv(
        out_dir / "per_episode.csv",
        per_episode_rows,
        fieldnames=[
            "episode_id",
            "task_name",
            "task_id",
            "env_id",
            "tries",
            "success_rate",
            "mean_length",
            "median_length",
            "mean_norm_score",
        ],
    )
    if len(per_task_rows) > 0:
        _write_csv(
            out_dir / "per_task.csv",
            per_task_rows,
            fieldnames=["task_name", "episodes", "tries", "success_rate", "mean_length", "mean_norm_score"],
        )

    baseline_per_task_rows: Optional[List[Dict[str, Any]]] = None
    if args.baseline_record_dir:
        baseline_dir = Path(args.baseline_record_dir)
        if not baseline_dir.is_dir():
            raise NotADirectoryError("baseline-record-dir does not exist: {}".format(baseline_dir))
        if args.baseline_results_file:
            baseline_results_path = Path(args.baseline_results_file)
        else:
            baseline_results_path = _detect_results_path(baseline_dir)
        _log("Loading baseline results: {}".format(baseline_results_path))
        baseline_results = _load_results_dict(baseline_results_path)

        baseline_task_meta = _load_dataset_meta(dataset_path, len(baseline_results))
        if len(baseline_task_meta) == 0:
            baseline_task_meta = _build_task_meta_fallback_from_logs(baseline_dir)

        baseline_summary, baseline_per_episode_rows, baseline_per_task_rows = _aggregate_results(
            results=baseline_results,
            task_meta=baseline_task_meta,
            max_steps=int(args.max_steps),
        )
        compare = _compare_with_baseline(per_episode_rows, baseline_per_episode_rows)
        _write_json(
            out_dir / "baseline_compare.json",
            {
                "current_summary": summary,
                "baseline_summary": baseline_summary,
                "comparison": compare,
                "baseline_record_dir": str(baseline_dir),
                "baseline_results_file": str(baseline_results_path),
            },
        )

    if not args.skip_plots:
        _plot_metrics(
            out_dir=out_dir,
            per_episode_rows=per_episode_rows,
            per_task_rows=per_task_rows,
            baseline_per_task_rows=baseline_per_task_rows,
        )

    _log(
        "Done. success={:.4f} mean_len={:.2f} trials={} -> {}".format(
            summary["success_rate_over_trials"],
            summary["mean_length_over_trials"],
            summary["total_trials"],
            out_dir,
        )
    )


if __name__ == "__main__":
    main()
