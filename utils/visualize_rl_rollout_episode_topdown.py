#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Headless-safe matplotlib backend before importing plotting helpers.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

if __package__:
    from .visualize_dataset_episode_topdown import (
        _safe_name,
        action_text_and_ids,
        detect_char_id,
        goal_ids_for_frame,
        normalize_graph_frame,
        run_ffmpeg,
        visible_ids_for_frame,
    )
    from .utils_plot import plot_graph_2d
else:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils.visualize_dataset_episode_topdown import (
        _safe_name,
        action_text_and_ids,
        detect_char_id,
        goal_ids_for_frame,
        normalize_graph_frame,
        run_ffmpeg,
        visible_ids_for_frame,
    )
    from utils.utils_plot import plot_graph_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an RL rollout log (with graph_seq/action) as top-down frames and optional MP4."
    )
    parser.add_argument("--rollout-log", type=str, required=True, help="Path to rollout log pickle (e.g., logs_agent_*.pik).")
    parser.add_argument(
        "--action-agent-id",
        type=int,
        default=None,
        help="Action stream to anchor character selection/highlights. Auto-detected if omitted.",
    )
    parser.add_argument("--alice-agent-id", type=int, default=0, help="Agent id used for Alice action overlay text.")
    parser.add_argument("--bob-agent-id", type=int, default=1, help="Agent id used for Bob action overlay text.")
    parser.add_argument(
        "--belief-agent-id",
        type=int,
        default=None,
        help="Agent id used for belief vectors. Defaults to bob-agent-id.",
    )
    parser.add_argument(
        "--char-id",
        type=int,
        default=None,
        help="Character node id to highlight. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="Override episode name shown in overlays and output path.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output dir (default under outputs/rl_topdown_viz/...).")
    parser.add_argument("--fps", type=int, default=5, help="Video FPS.")
    parser.add_argument("--dpi", type=int, default=120, help="Frame DPI.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rendered steps.")
    parser.add_argument("--top-k-text", type=int, default=5, help="Top-k beliefs shown in overlay text.")
    parser.add_argument("--plot-k", type=int, default=6, help="Number of belief predicate curves on the right panel.")
    parser.add_argument("--no-belief-panel", action="store_true", default=False)
    parser.add_argument("--skip-video", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    return parser.parse_args()


def _load_rollout(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError("Rollout log is not a dict: {}".format(type(payload).__name__))
    return payload


def _normalize_action_map(action_bundle: Any) -> Dict[int, List[Any]]:
    out: Dict[int, List[Any]] = {}
    if not isinstance(action_bundle, dict):
        return out
    for key, seq in action_bundle.items():
        try:
            agent_id = int(key)
        except Exception:
            continue
        if isinstance(seq, list):
            out[agent_id] = seq
    return out


def _normalize_seq_map(bundle: Any) -> Dict[int, List[Any]]:
    out: Dict[int, List[Any]] = {}
    if not isinstance(bundle, dict):
        return out
    for key, seq in bundle.items():
        try:
            agent_id = int(key)
        except Exception:
            continue
        if isinstance(seq, list):
            out[agent_id] = seq
    return out


def _extract_goal_class_names(log_payload: Dict[str, Any]) -> List[str]:
    names = set()

    def _accumulate_from_predicates(pred_keys: Sequence[str]) -> None:
        for pred in pred_keys:
            parts = str(pred).split("_")
            if len(parts) < 2:
                continue
            for token in parts[1:]:
                if not token or str(token).isdigit():
                    continue
                names.add(str(token))

    gt_goals = log_payload.get("gt_goals")
    if isinstance(gt_goals, dict):
        _accumulate_from_predicates(list(gt_goals.keys()))

    goals = log_payload.get("goals")
    if isinstance(goals, dict):
        if 0 in goals and isinstance(goals[0], dict):
            _accumulate_from_predicates(list(goals[0].keys()))
        if "0" in goals and isinstance(goals["0"], dict):
            _accumulate_from_predicates(list(goals["0"].keys()))

    return sorted(names)


def _extract_gt_goal_counts(log_payload: Dict[str, Any]) -> Dict[str, int]:
    gt_raw = log_payload.get("gt_goals", {})
    if not isinstance(gt_raw, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in gt_raw.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            continue
    return out


def _satisfied_value_to_count(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return int(len(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return max(0, int(value))
    return 0


def _extract_progress_counts(log_payload: Dict[str, Any], step_idx: int) -> Dict[str, int]:
    goals_finished = log_payload.get("goals_finished")
    if not isinstance(goals_finished, list) or step_idx < 0 or step_idx >= len(goals_finished):
        return {}
    row = goals_finished[step_idx]
    if not isinstance(row, dict):
        return {}
    return {str(key): _satisfied_value_to_count(val) for key, val in row.items()}


def _choose_char_id(nodes: Sequence[Dict[str, Any]], action_agent_id: Optional[int], forced_char_id: Optional[int]) -> int:
    if forced_char_id is not None:
        return int(forced_char_id)

    if action_agent_id is not None:
        preferred = int(action_agent_id) + 1
        for node in nodes:
            try:
                if int(node.get("id", -1)) == preferred:
                    return preferred
            except Exception:
                continue

    return int(detect_char_id(nodes))


def _safe_action_text(action_seq: Sequence[Any], step_idx: int) -> str:
    if step_idx < 0 or step_idx >= len(action_seq):
        return ""
    action_item = action_seq[step_idx]
    if action_item is None:
        return ""
    return str(action_item)


def _resolve_episode_name(log_payload: Dict[str, Any], rollout_path: Path, override_name: Optional[str]) -> str:
    if override_name:
        return str(override_name)

    task_name = log_payload.get("episode_task_name") or log_payload.get("task_name")
    task_id = log_payload.get("episode_task_id")
    if task_name is not None and task_id is not None:
        return "logs_agent_{}_{}".format(task_id, task_name)
    if task_name is not None:
        return str(task_name)
    return rollout_path.stem


def _to_1d_float_array(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _extract_belief_payload(
    payload: Dict[str, Any],
    belief_agent_id: int,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[Optional[str]], Optional[List[str]]]:
    probs_map = _normalize_seq_map(payload.get("belief_probs", {}))
    logits_map = _normalize_seq_map(payload.get("belief_logits", {}))
    source_map = _normalize_seq_map(payload.get("belief_source", {}))

    raw_probs = probs_map.get(int(belief_agent_id), [])
    raw_logits = logits_map.get(int(belief_agent_id), [])
    raw_source = source_map.get(int(belief_agent_id), [])

    probs_seq: List[np.ndarray] = []
    logits_seq: List[Optional[np.ndarray]] = []
    source_seq: List[Optional[str]] = []

    for idx, raw_prob in enumerate(raw_probs):
        probs = _to_1d_float_array(raw_prob)
        if probs is None:
            continue
        logits = None
        if idx < len(raw_logits):
            logits = _to_1d_float_array(raw_logits[idx])
            if logits is not None and logits.shape[0] != probs.shape[0]:
                logits = None

        source = None
        if idx < len(raw_source):
            source_val = raw_source[idx]
            if source_val is not None:
                source = str(source_val)

        probs_seq.append(probs)
        logits_seq.append(logits)
        source_seq.append(source)

    predicate_names: Optional[List[str]] = None
    raw_name_map = payload.get("belief_predicate_names")
    if isinstance(raw_name_map, dict):
        if int(belief_agent_id) in raw_name_map and isinstance(raw_name_map[int(belief_agent_id)], list):
            predicate_names = [str(x) for x in raw_name_map[int(belief_agent_id)]]
        elif str(int(belief_agent_id)) in raw_name_map and isinstance(raw_name_map[str(int(belief_agent_id))], list):
            predicate_names = [str(x) for x in raw_name_map[str(int(belief_agent_id))]]
    elif isinstance(raw_name_map, list):
        predicate_names = [str(x) for x in raw_name_map]

    return probs_seq, logits_seq, source_seq, predicate_names


def _choose_belief_agent_id(
    explicit_agent_id: Optional[int],
    bob_agent_id: int,
    probs_map: Dict[int, List[Any]],
) -> Optional[int]:
    if explicit_agent_id is not None:
        return int(explicit_agent_id)
    if bob_agent_id in probs_map:
        return int(bob_agent_id)
    if len(probs_map) == 0:
        return None
    ranked = sorted(probs_map.keys(), key=lambda aid: len(probs_map[aid]), reverse=True)
    return int(ranked[0]) if len(ranked) > 0 else None


def _choose_tracked_predicates(
    y_probs_seq: np.ndarray,
    gt_indices: Sequence[int],
    plot_k: int,
) -> List[int]:
    k = max(1, int(plot_k))
    tracked: List[int] = []
    seen = set()

    for idx in gt_indices:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= y_probs_seq.shape[1] or idx_i in seen:
            continue
        tracked.append(idx_i)
        seen.add(idx_i)
        if len(tracked) >= k:
            return tracked

    peak = y_probs_seq.max(axis=0)
    drift = y_probs_seq.max(axis=0) - y_probs_seq.min(axis=0)
    score = peak + 0.5 * drift
    order = np.argsort(score)[::-1]
    for idx in order.tolist():
        if idx in seen:
            continue
        tracked.append(int(idx))
        seen.add(int(idx))
        if len(tracked) >= k:
            break
    return tracked


def _short_label(text: str, max_len: int = 28) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _format_goal_lines(goal_counts: Dict[str, int], max_items: int = 6) -> List[str]:
    if len(goal_counts) == 0:
        return ["GT goals: <none>"]
    items = sorted(goal_counts.items())
    lines = ["GT goals (predicate:count):"]
    for idx, (pred, count) in enumerate(items[:max_items]):
        lines.append("  {}:{}".format(pred, int(count)))
    if len(items) > max_items:
        lines.append("  ... +{} more".format(len(items) - max_items))
    return lines


def _format_progress_lines(
    goal_counts: Dict[str, int],
    progress_counts: Dict[str, int],
    max_items: int = 6,
) -> List[str]:
    if len(goal_counts) == 0:
        return ["Progress: <none>"]
    lines = ["Progress at step (done/target):"]
    items = sorted(goal_counts.items())
    for idx, (pred, target) in enumerate(items[:max_items]):
        done = int(progress_counts.get(pred, 0))
        lines.append("  {}:{}/{}".format(pred, done, int(target)))
    if len(items) > max_items:
        lines.append("  ... +{} more".format(len(items) - max_items))
    return lines


def _step_topk(y_probs_t: np.ndarray, predicate_names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
    if y_probs_t.size == 0:
        return []
    k = max(1, min(int(top_k), int(y_probs_t.shape[0])))
    order = np.argsort(y_probs_t)[::-1][:k]
    return [
        {
            "predicate_index": int(i),
            "predicate": str(predicate_names[int(i)]) if int(i) < len(predicate_names) else "pred_{:03d}".format(int(i)),
            "prob": float(y_probs_t[int(i)]),
        }
        for i in order
    ]


def _step_gt_probs(y_probs_t: np.ndarray, predicate_names: Sequence[str], gt_indices: Sequence[int]) -> List[Dict[str, Any]]:
    items = []
    for idx in gt_indices:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= y_probs_t.shape[0]:
            continue
        name = str(predicate_names[idx_i]) if idx_i < len(predicate_names) else "pred_{:03d}".format(idx_i)
        items.append(
            {
                "predicate_index": idx_i,
                "predicate": name,
                "prob": float(y_probs_t[idx_i]),
            }
        )
    items.sort(key=lambda d: d["prob"], reverse=True)
    return items


def _render_step_frame(
    *,
    frame_path: Path,
    graph: Dict[str, Any],
    char_id: int,
    visible_ids: Sequence[int],
    action_ids: Sequence[int],
    goal_ids: Sequence[int],
    step_idx: int,
    total_steps: int,
    episode_name: str,
    task_name: str,
    alice_action_text: str,
    bob_action_text: str,
    gt_goal_counts: Dict[str, int],
    progress_counts: Dict[str, int],
    y_probs_seq: Optional[np.ndarray],
    predicate_names: Sequence[str],
    tracked_pred_indices: Sequence[int],
    gt_pred_indices: Sequence[int],
    top_k_text: int,
    dpi: int,
    show_belief_panel: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fig = plot_graph_2d(
        graph=graph,
        char_id=char_id,
        visible_ids=list(visible_ids),
        action_ids=list(action_ids),
        goal_ids=list(goal_ids),
    )

    panel_enabled = bool(show_belief_panel and y_probs_seq is not None and y_probs_seq.shape[0] > 0)
    if panel_enabled:
        fig.subplots_adjust(top=0.80, right=0.61)
    else:
        fig.subplots_adjust(top=0.80, right=0.98)

    wrapped_alice = textwrap.fill(alice_action_text if alice_action_text else "<none>", width=74)
    wrapped_bob = textwrap.fill(bob_action_text if bob_action_text else "<none>", width=74)

    header = "{} | task: {}".format(episode_name, task_name)
    step_line = "Step: {}/{}".format(step_idx, max(total_steps - 1, 0))
    overlay_lines = [
        header,
        step_line,
        "Alice action: {}".format(wrapped_alice),
        "Bob action: {}".format(wrapped_bob),
    ]
    overlay_lines.extend(_format_goal_lines(gt_goal_counts))
    overlay_lines.extend(_format_progress_lines(gt_goal_counts, progress_counts))

    topk: List[Dict[str, Any]] = []
    gt_probs: List[Dict[str, Any]] = []
    if y_probs_seq is not None and step_idx < y_probs_seq.shape[0]:
        y_t = y_probs_seq[step_idx]
        topk = _step_topk(y_t, predicate_names, top_k=top_k_text)
        gt_probs = _step_gt_probs(y_t, predicate_names, gt_pred_indices)
        overlay_lines.append("Top belief @ step:")
        if len(topk) == 0:
            overlay_lines.append("  <none>")
        else:
            for item in topk:
                overlay_lines.append("  {:>5.3f} {}".format(item["prob"], item["predicate"]))

    fig.text(
        0.01,
        0.985,
        "\n".join(overlay_lines),
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.94, "edgecolor": "black", "pad": 4},
    )

    if panel_enabled and y_probs_seq is not None:
        ax_bel = fig.add_axes([0.64, 0.18, 0.34, 0.62])
        t_axis = np.arange(y_probs_seq.shape[0])
        gt_set = set([int(i) for i in gt_pred_indices])
        for idx in tracked_pred_indices:
            idx_i = int(idx)
            if idx_i < 0 or idx_i >= y_probs_seq.shape[1]:
                continue
            y_curve = y_probs_seq[:, idx_i]
            is_gt = idx_i in gt_set
            label = str(predicate_names[idx_i]) if idx_i < len(predicate_names) else "pred_{:03d}".format(idx_i)
            ax_bel.plot(
                t_axis,
                y_curve,
                linewidth=2.0 if is_gt else 1.4,
                linestyle="-" if is_gt else "--",
                alpha=0.95 if is_gt else 0.85,
                label=_short_label(label),
            )
            if step_idx < y_curve.shape[0]:
                ax_bel.scatter([step_idx], [y_curve[step_idx]], s=14, zorder=4)

        ax_bel.axvline(step_idx, color="black", linestyle=":", linewidth=1.2, alpha=0.8)
        ax_bel.set_xlim(0, max(y_probs_seq.shape[0] - 1, 1))
        ax_bel.set_ylim(-0.02, 1.02)
        ax_bel.set_xlabel("Timestep")
        ax_bel.set_ylabel("p(y=1)")
        ax_bel.set_title("Bob belief (selected predicates)", fontsize=9)
        ax_bel.grid(alpha=0.25)
        if len(tracked_pred_indices) > 0:
            ax_bel.legend(fontsize=6, loc="upper left", framealpha=0.9)

    fig.savefig(str(frame_path), dpi=dpi)
    plt.close(fig)
    return topk, gt_probs


def _serialize_array(arr: Optional[np.ndarray]) -> Optional[List[float]]:
    if arr is None:
        return None
    return [float(x) for x in arr.reshape(-1).tolist()]


def main() -> None:
    args = parse_args()
    rollout_path = Path(args.rollout_log)
    if not rollout_path.is_file():
        raise FileNotFoundError("Rollout log not found: {}".format(rollout_path))

    payload = _load_rollout(rollout_path)

    graph_seq_raw = payload.get("graph_seq")
    if not isinstance(graph_seq_raw, list) or len(graph_seq_raw) == 0:
        graph_seq_raw = payload.get("graph")
    if not isinstance(graph_seq_raw, list) or len(graph_seq_raw) == 0:
        raise KeyError(
            "Rollout log has no graph_seq list. Need logs with graph snapshots (e.g., policy_episode_log.pik or logs_agent_*.pik from arena.run)."
        )

    action_map = _normalize_action_map(payload.get("action", {}))
    alice_action_seq = action_map.get(int(args.alice_agent_id), [])
    bob_action_seq = action_map.get(int(args.bob_agent_id), [])

    if args.action_agent_id is not None:
        action_agent_id = int(args.action_agent_id)
    elif len(bob_action_seq) > 0:
        action_agent_id = int(args.bob_agent_id)
    elif len(alice_action_seq) > 0:
        action_agent_id = int(args.alice_agent_id)
    else:
        action_agent_id = None

    max_action_len = 0
    for seq in action_map.values():
        if isinstance(seq, list):
            max_action_len = max(max_action_len, len(seq))

    if max_action_len > 0:
        if len(graph_seq_raw) == max_action_len + 1:
            total_steps = int(max_action_len)
        else:
            total_steps = int(min(len(graph_seq_raw), max_action_len))
    else:
        total_steps = int(len(graph_seq_raw))

    belief_probs_map = _normalize_seq_map(payload.get("belief_probs", {}))
    belief_agent_id = _choose_belief_agent_id(args.belief_agent_id, int(args.bob_agent_id), belief_probs_map)

    belief_probs_seq: List[np.ndarray] = []
    belief_logits_seq: List[Optional[np.ndarray]] = []
    belief_source_seq: List[Optional[str]] = []
    predicate_names: Optional[List[str]] = None
    if belief_agent_id is not None:
        belief_probs_seq, belief_logits_seq, belief_source_seq, predicate_names = _extract_belief_payload(
            payload=payload,
            belief_agent_id=int(belief_agent_id),
        )

    if len(belief_probs_seq) > 0:
        total_steps = min(total_steps, len(belief_probs_seq))

    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))
    if total_steps <= 0:
        raise ValueError("No steps to render after alignment/max-steps.")

    if len(belief_probs_seq) > total_steps:
        belief_probs_seq = belief_probs_seq[:total_steps]
        belief_logits_seq = belief_logits_seq[:total_steps]
        belief_source_seq = belief_source_seq[:total_steps]

    y_probs_seq: Optional[np.ndarray] = None
    if len(belief_probs_seq) > 0:
        belief_dim = int(belief_probs_seq[0].shape[0])
        filtered_probs: List[np.ndarray] = []
        filtered_logits: List[Optional[np.ndarray]] = []
        filtered_source: List[Optional[str]] = []
        for idx, probs in enumerate(belief_probs_seq):
            if probs.shape[0] != belief_dim:
                continue
            filtered_probs.append(probs)
            filtered_logits.append(belief_logits_seq[idx] if idx < len(belief_logits_seq) else None)
            filtered_source.append(belief_source_seq[idx] if idx < len(belief_source_seq) else None)

        if len(filtered_probs) > 0:
            belief_probs_seq = filtered_probs[:total_steps]
            belief_logits_seq = filtered_logits[:total_steps]
            belief_source_seq = filtered_source[:total_steps]
            total_steps = min(total_steps, len(belief_probs_seq))
            y_probs_seq = np.stack(belief_probs_seq, axis=0)
        else:
            belief_probs_seq = []
            belief_logits_seq = []
            belief_source_seq = []

    if y_probs_seq is not None:
        belief_dim = int(y_probs_seq.shape[1])
        if predicate_names is None or len(predicate_names) != belief_dim:
            predicate_names = ["pred_{:03d}".format(i) for i in range(belief_dim)]
    else:
        predicate_names = []

    episode_name = _resolve_episode_name(payload, rollout_path, args.episode_name)
    task_name = str(payload.get("task_name") or payload.get("episode_task_name") or "unknown")
    goal_class_names = _extract_goal_class_names(payload)
    gt_goal_counts = _extract_gt_goal_counts(payload)

    gt_pred_indices: List[int] = []
    if y_probs_seq is not None:
        goal_pred_to_idx = {str(name): idx for idx, name in enumerate(predicate_names)}
        for pred_name in sorted(gt_goal_counts.keys()):
            if pred_name in goal_pred_to_idx:
                gt_pred_indices.append(int(goal_pred_to_idx[pred_name]))
        tracked_pred_indices = _choose_tracked_predicates(y_probs_seq, gt_pred_indices, plot_k=int(args.plot_k))
    else:
        tracked_pred_indices = []

    if args.out_dir is None:
        out_dir = Path("outputs") / "rl_topdown_viz" / _safe_name(rollout_path.stem)
    else:
        out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError("Output directory exists: {} (pass --overwrite to reuse)".format(out_dir))
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale in frames_dir.glob("frame_*.png"):
            try:
                stale.unlink()
            except OSError:
                pass

    steps_manifest: List[Dict[str, Any]] = []

    for step_idx in range(total_steps):
        graph = normalize_graph_frame(graph_seq_raw[step_idx])
        nodes = graph["nodes"]

        char_id = _choose_char_id(nodes, action_agent_id=action_agent_id, forced_char_id=args.char_id)
        visible_ids = visible_ids_for_frame(nodes)

        alice_text_raw = _safe_action_text(alice_action_seq, step_idx)
        bob_text_raw = _safe_action_text(bob_action_seq, step_idx)
        alice_action_text, alice_ids = action_text_and_ids(alice_text_raw)
        bob_action_text, bob_ids = action_text_and_ids(bob_text_raw)

        visible_set = set(visible_ids)
        action_ids = [obj_id for obj_id in (alice_ids + bob_ids) if obj_id in visible_set]
        goal_ids = [obj_id for obj_id in goal_ids_for_frame(nodes, goal_class_names) if obj_id in visible_set]

        progress_counts = _extract_progress_counts(payload, step_idx)

        frame_file = "frame_{:04d}.png".format(step_idx)
        frame_path = frames_dir / frame_file

        topk, gt_probs = _render_step_frame(
            frame_path=frame_path,
            graph=graph,
            char_id=char_id,
            visible_ids=visible_ids,
            action_ids=action_ids,
            goal_ids=goal_ids,
            step_idx=step_idx,
            total_steps=total_steps,
            episode_name=episode_name,
            task_name=task_name,
            alice_action_text=alice_action_text,
            bob_action_text=bob_action_text,
            gt_goal_counts=gt_goal_counts,
            progress_counts=progress_counts,
            y_probs_seq=y_probs_seq,
            predicate_names=predicate_names,
            tracked_pred_indices=tracked_pred_indices,
            gt_pred_indices=gt_pred_indices,
            top_k_text=int(args.top_k_text),
            dpi=int(args.dpi),
            show_belief_panel=not bool(args.no_belief_panel),
        )

        belief_probs_step = None
        belief_logits_step = None
        belief_source_step = None
        if step_idx < len(belief_probs_seq):
            belief_probs_step = belief_probs_seq[step_idx]
        if step_idx < len(belief_logits_seq):
            belief_logits_step = belief_logits_seq[step_idx]
        if step_idx < len(belief_source_seq):
            belief_source_step = belief_source_seq[step_idx]

        steps_manifest.append(
            {
                "step_idx": int(step_idx),
                "frame_file": str(Path("frames") / frame_file),
                "graph_index": int(step_idx),
                "char_id": int(char_id),
                "alice_agent_id": int(args.alice_agent_id),
                "bob_agent_id": int(args.bob_agent_id),
                "alice_action_text": alice_action_text,
                "bob_action_text": bob_action_text,
                "parsed_alice_action_object_ids": alice_ids,
                "parsed_bob_action_object_ids": bob_ids,
                "highlighted_action_object_ids": action_ids,
                "goal_ids": goal_ids,
                "gt_goal_counts": {k: int(v) for k, v in gt_goal_counts.items()},
                "progress_counts": {k: int(v) for k, v in progress_counts.items()},
                "belief_agent_id": None if belief_agent_id is None else int(belief_agent_id),
                "belief_source": None if belief_source_step is None else str(belief_source_step),
                "belief_topk": topk,
                "belief_gt_probs": gt_probs,
            }
        )

    meta = {
        "rollout_log": str(rollout_path),
        "episode_name": episode_name,
        "task_name": task_name,
        "task_id": payload.get("task_id") or payload.get("episode_task_id"),
        "env_id": payload.get("env_id") or payload.get("episode_env_id"),
        "finished": payload.get("finished"),
        "alice_agent_id": int(args.alice_agent_id),
        "bob_agent_id": int(args.bob_agent_id),
        "belief_agent_id": None if belief_agent_id is None else int(belief_agent_id),
        "action_agent_id": None if action_agent_id is None else int(action_agent_id),
        "char_id_override": None if args.char_id is None else int(args.char_id),
        "num_graph_frames": len(graph_seq_raw),
        "num_actions_alice": len(alice_action_seq),
        "num_actions_bob": len(bob_action_seq),
        "num_rendered_steps": int(total_steps),
        "goal_class_names": goal_class_names,
        "gt_goal_counts": {k: int(v) for k, v in gt_goal_counts.items()},
        "belief_available": bool(y_probs_seq is not None),
        "belief_predicate_names": list(predicate_names),
        "fps": int(args.fps),
    }

    belief_json = {
        "rollout_log": str(rollout_path),
        "belief_available": bool(y_probs_seq is not None),
        "belief_agent_id": None if belief_agent_id is None else int(belief_agent_id),
        "predicate_names": list(predicate_names),
        "num_steps": int(total_steps),
        "steps": [],
    }
    for step_idx in range(total_steps):
        probs = belief_probs_seq[step_idx] if step_idx < len(belief_probs_seq) else None
        logits = belief_logits_seq[step_idx] if step_idx < len(belief_logits_seq) else None
        source = belief_source_seq[step_idx] if step_idx < len(belief_source_seq) else None
        belief_json["steps"].append(
            {
                "step_idx": int(step_idx),
                "belief_source": None if source is None else str(source),
                "belief_probs": _serialize_array(probs),
                "belief_logits": _serialize_array(logits),
            }
        )

    with open(out_dir / "episode_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "steps.json", "w") as f:
        json.dump(steps_manifest, f, indent=2)
    with open(out_dir / "belief_seq_full.json", "w") as f:
        json.dump(belief_json, f, indent=2)

    video_path = out_dir / "episode_topdown.mp4"
    if not args.skip_video:
        try:
            run_ffmpeg(frames_dir=frames_dir, out_path=video_path, fps=int(args.fps))
        except FileNotFoundError as e:
            print("ffmpeg not found; frames were rendered but video export was skipped: {}".format(repr(e)), file=sys.stderr)

    print("Rendered {} frames to {}".format(total_steps, frames_dir))
    if not args.skip_video and video_path.exists():
        print("Video written to {}".format(video_path))
    print("Metadata: {}".format(out_dir / "episode_meta.json"))
    print("Steps manifest: {}".format(out_dir / "steps.json"))
    print("Belief sequence: {}".format(out_dir / "belief_seq_full.json"))


if __name__ == "__main__":
    main()
