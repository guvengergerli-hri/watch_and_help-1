#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Headless-safe matplotlib backend before importing plotting helpers.
os.environ.setdefault("MPLBACKEND", "Agg")

if __package__:
    from .visualize_dataset_episode_topdown import (
        _safe_name,
        action_text_and_ids,
        detect_char_id,
        goal_ids_for_frame,
        normalize_graph_frame,
        render_step_frame,
        run_ffmpeg,
        visible_ids_for_frame,
    )
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
        render_step_frame,
        run_ffmpeg,
        visible_ids_for_frame,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an RL rollout log (with graph_seq/action) as top-down frames and optional MP4."
    )
    parser.add_argument("--rollout-log", type=str, required=True, help="Path to rollout log pickle (e.g., logs_agent_*.pik).")
    parser.add_argument(
        "--action-agent-id",
        type=int,
        default=None,
        help="Action stream to overlay. Auto-detected if omitted (prefers agent 1 then 0).",
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


def _choose_action_agent_id(action_map: Dict[int, List[Any]], preferred: Optional[int]) -> Optional[int]:
    if len(action_map) == 0:
        return None
    if preferred is not None and preferred in action_map:
        return int(preferred)

    def _nonempty_count(seq: Sequence[Any]) -> int:
        count = 0
        for x in seq:
            if isinstance(x, str) and x.strip():
                count += 1
        return count

    # Prefer the stream with most actual (non-empty) actions.
    ranked = sorted(
        action_map.keys(),
        key=lambda k: (_nonempty_count(action_map[k]), len(action_map[k])),
        reverse=True,
    )
    if len(ranked) > 0:
        return int(ranked[0])
    return None


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


def main() -> None:
    args = parse_args()
    rollout_path = Path(args.rollout_log)
    if not rollout_path.is_file():
        raise FileNotFoundError("Rollout log not found: {}".format(rollout_path))

    payload = _load_rollout(rollout_path)

    graph_seq_raw = payload.get("graph_seq")
    if not isinstance(graph_seq_raw, list) or len(graph_seq_raw) == 0:
        # Fallback for some eval logs.
        graph_seq_raw = payload.get("graph")
    if not isinstance(graph_seq_raw, list) or len(graph_seq_raw) == 0:
        raise KeyError(
            "Rollout log has no graph_seq list. Need logs with graph snapshots (e.g., policy_episode_log.pik or logs_agent_*.pik from arena.run)."
        )

    action_map = _normalize_action_map(payload.get("action", {}))
    action_agent_id = _choose_action_agent_id(action_map, args.action_agent_id)
    action_seq: List[Any] = action_map.get(action_agent_id, []) if action_agent_id is not None else []

    if len(action_seq) > 0:
        if len(graph_seq_raw) == len(action_seq) + 1:
            total_steps = len(action_seq)
        else:
            total_steps = min(len(graph_seq_raw), len(action_seq))
    else:
        total_steps = len(graph_seq_raw)

    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))
    if total_steps <= 0:
        raise ValueError("No steps to render after alignment/max-steps.")

    episode_name = _resolve_episode_name(payload, rollout_path, args.episode_name)
    task_name = str(payload.get("task_name") or payload.get("episode_task_name") or "unknown")
    goal_class_names = _extract_goal_class_names(payload)

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
        action_text = _safe_action_text(action_seq, step_idx)
        action_text_for_parse = action_text
        action_text, parsed_action_ids = action_text_and_ids(action_text_for_parse)

        visible_set = set(visible_ids)
        action_ids = [obj_id for obj_id in parsed_action_ids if obj_id in visible_set]
        goal_ids = [obj_id for obj_id in goal_ids_for_frame(nodes, goal_class_names) if obj_id in visible_set]

        frame_file = "frame_{:04d}.png".format(step_idx)
        frame_path = frames_dir / frame_file

        render_step_frame(
            frame_path=frame_path,
            graph=graph,
            char_id=char_id,
            visible_ids=visible_ids,
            action_ids=action_ids,
            goal_ids=goal_ids,
            step_idx=step_idx,
            total_steps=total_steps,
            action_text=action_text,
            episode_name=episode_name,
            task_name=task_name,
            dpi=int(args.dpi),
        )

        steps_manifest.append(
            {
                "step_idx": int(step_idx),
                "frame_file": str(Path("frames") / frame_file),
                "graph_index": int(step_idx),
                "char_id": int(char_id),
                "action_agent_id": None if action_agent_id is None else int(action_agent_id),
                "action_text": action_text,
                "parsed_action_object_ids": parsed_action_ids,
                "highlighted_action_object_ids": action_ids,
                "goal_ids": goal_ids,
            }
        )

    meta = {
        "rollout_log": str(rollout_path),
        "episode_name": episode_name,
        "task_name": task_name,
        "task_id": payload.get("task_id") or payload.get("episode_task_id"),
        "env_id": payload.get("env_id") or payload.get("episode_env_id"),
        "finished": payload.get("finished"),
        "action_agent_id": None if action_agent_id is None else int(action_agent_id),
        "char_id_override": None if args.char_id is None else int(args.char_id),
        "num_graph_frames": len(graph_seq_raw),
        "num_actions_for_agent": len(action_seq),
        "num_rendered_steps": int(total_steps),
        "goal_class_names": goal_class_names,
        "fps": int(args.fps),
    }

    with open(out_dir / "episode_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "steps.json", "w") as f:
        json.dump(steps_manifest, f, indent=2)

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


if __name__ == "__main__":
    main()
