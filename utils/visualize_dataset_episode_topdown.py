#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Headless-safe matplotlib backend before importing utils.utils_plot.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

if __package__:
    from .utils_plot import plot_graph_2d
else:
    # Support direct execution: python utils/visualize_dataset_episode_topdown.py
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils.utils_plot import plot_graph_2d


ACTION_ID_PATTERN = re.compile(r"\((\d+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a dataset episode as top-down frames with timestep/action overlays and optionally export MP4."
    )
    parser.add_argument(
        "--data-json",
        type=str,
        default="dataset/watch_data/gather_data_actiongraph_test.json",
        help="Path to gather_data_actiongraph_{train,test,new_test}.json",
    )
    parser.add_argument(
        "--split-key",
        type=str,
        default=None,
        help="Top-level split key inside the JSON (e.g., test_data, train_data, new_test_data). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index within the split (ignored if --episode-name is provided).",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="Exact demo name to render (e.g., logs_agent_141_read_book).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/dataset_topdown_viz/<data_stem>/<episode_name>/",
    )
    parser.add_argument("--fps", type=int, default=5, help="Video frame rate for ffmpeg export.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on rendered steps (useful for quick tests).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="PNG DPI for saved frames.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        default=False,
        help="Render frames only and skip ffmpeg MP4 export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "episode"


def load_split(data_json: Path, split_key: Optional[str]) -> Tuple[str, List[Dict[str, Any]]]:
    with open(data_json, "r") as f:
        content = json.load(f)

    if split_key is None:
        if len(content) == 1:
            split_key = next(iter(content.keys()))
        else:
            raise KeyError(
                "JSON has multiple top-level keys ({}). Please pass --split-key.".format(list(content.keys()))
            )

    if split_key not in content:
        raise KeyError(
            "Split key '{}' not found in {}. Available keys: {}".format(split_key, data_json, list(content.keys()))
        )

    demos = content[split_key]
    if not isinstance(demos, list):
        raise TypeError("Split '{}' in {} is not a list.".format(split_key, data_json))
    return split_key, demos


def select_episode(
    demos: Sequence[Dict[str, Any]],
    episode_index: int,
    episode_name: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    if episode_name is not None:
        for idx, demo in enumerate(demos):
            if str(demo.get("name", "")) == episode_name:
                return idx, demo
        raise KeyError("Episode name '{}' not found in split.".format(episode_name))

    if episode_index < 0 or episode_index >= len(demos):
        raise IndexError("episode-index {} out of range [0, {}).".format(episode_index, len(demos)))
    return episode_index, demos[episode_index]


def normalize_graph_frame(frame: Any) -> Dict[str, Any]:
    if isinstance(frame, dict):
        nodes = frame.get("nodes", [])
        edges = frame.get("edges", [])
        if not isinstance(nodes, list):
            raise TypeError("Frame graph['nodes'] is not a list.")
        if not isinstance(edges, list):
            edges = []
        return {"nodes": nodes, "edges": edges}

    if isinstance(frame, list):
        return {"nodes": frame, "edges": []}

    raise TypeError("Unsupported frame type: {}".format(type(frame).__name__))


def has_valid_bbox(node: Dict[str, Any]) -> bool:
    bbox = node.get("bounding_box")
    if not isinstance(bbox, dict):
        return False
    center = bbox.get("center")
    size = bbox.get("size")
    if not isinstance(center, (list, tuple)) or not isinstance(size, (list, tuple)):
        return False
    if len(center) != 3 or len(size) != 3:
        return False
    return True


def detect_char_id(nodes: Sequence[Dict[str, Any]]) -> int:
    for node in nodes:
        if "character" in str(node.get("class_name", "")).lower():
            try:
                return int(node["id"])
            except Exception:
                break

    for node in nodes:
        try:
            if int(node.get("id", -1)) == 1:
                return 1
        except Exception:
            continue
    raise ValueError("Could not find a character node in frame.")


def action_text_and_ids(action_entry: Any) -> Tuple[str, List[int]]:
    if isinstance(action_entry, (list, tuple)):
        action_text = str(action_entry[0]) if len(action_entry) > 0 else ""
    else:
        action_text = "" if action_entry is None else str(action_entry)

    action_ids = [int(m) for m in ACTION_ID_PATTERN.findall(action_text)]
    return action_text, action_ids


def extract_goal_class_names(goal_list: Sequence[Any]) -> List[str]:
    names = set()
    for goal in goal_list:
        parts = str(goal).split("_")
        if len(parts) < 2:
            continue
        for token in parts[1:]:
            if not token or token.isdigit():
                continue
            names.add(token)
    return sorted(names)


def goal_ids_for_frame(nodes: Sequence[Dict[str, Any]], goal_class_names: Sequence[str]) -> List[int]:
    if not goal_class_names:
        return []
    goal_name_set = set([str(x) for x in goal_class_names])
    out = []
    for node in nodes:
        if not has_valid_bbox(node):
            continue
        if str(node.get("class_name", "")) in goal_name_set:
            try:
                out.append(int(node["id"]))
            except Exception:
                continue
    return out


def visible_ids_for_frame(nodes: Sequence[Dict[str, Any]]) -> List[int]:
    out = []
    for node in nodes:
        if not has_valid_bbox(node):
            continue
        try:
            out.append(int(node["id"]))
        except Exception:
            continue
    return out


def render_step_frame(
    *,
    frame_path: Path,
    graph: Dict[str, Any],
    char_id: int,
    visible_ids: Sequence[int],
    action_ids: Sequence[int],
    goal_ids: Sequence[int],
    step_idx: int,
    total_steps: int,
    action_text: str,
    episode_name: str,
    task_name: str,
    dpi: int,
) -> None:
    fig = plot_graph_2d(
        graph=graph,
        char_id=char_id,
        visible_ids=list(visible_ids),
        action_ids=list(action_ids),
        goal_ids=list(goal_ids),
    )

    # Leave space for overlay text at the top.
    fig.subplots_adjust(top=0.83)

    wrapped_action = textwrap.fill(action_text if action_text else "<none>", width=70)
    header = "{} | task: {}".format(episode_name, task_name)
    step_line = "Step: {}/{}".format(step_idx, max(total_steps - 1, 0))
    action_line = "Alice valid action: {}".format(wrapped_action)
    overlay_text = "{}\n{}\n{}".format(header, step_line, action_line)

    fig.text(
        0.01,
        0.985,
        overlay_text,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "black", "pad": 4},
    )

    fig.savefig(str(frame_path), dpi=dpi)
    plt.close(fig)


def run_ffmpeg(frames_dir: Path, out_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    data_json = Path(args.data_json)
    split_key, demos = load_split(data_json, args.split_key)
    selected_index, demo = select_episode(demos, args.episode_index, args.episode_name)

    graphs = demo.get("graphs", [])
    actions = demo.get("valid_action_with_walk", [])
    if not isinstance(graphs, list) or not isinstance(actions, list):
        raise TypeError("Episode graphs/actions are not lists.")

    if len(graphs) == 0:
        raise ValueError("Selected episode has no graph frames.")
    if len(actions) == 0:
        raise ValueError("Selected episode has no actions.")

    total_steps = min(len(graphs), len(actions))
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))
    if total_steps <= 0:
        raise ValueError("No steps to render after applying --max-steps.")

    episode_name = str(demo.get("name", "episode_{}".format(selected_index)))
    task_name = str(demo.get("task_name", "unknown"))
    goal_list = demo.get("goal", [])
    goal_class_names = extract_goal_class_names(goal_list)

    if args.out_dir is None:
        out_dir = Path("outputs") / "dataset_topdown_viz" / data_json.stem / _safe_name(episode_name)
    else:
        out_dir = Path(args.out_dir)

    frames_dir = out_dir / "frames"
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError("Output directory exists: {} (pass --overwrite to reuse)".format(out_dir))
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale_frame in frames_dir.glob("frame_*.png"):
            try:
                stale_frame.unlink()
            except OSError:
                pass

    steps_manifest: List[Dict[str, Any]] = []

    for step_idx in range(total_steps):
        graph = normalize_graph_frame(graphs[step_idx])
        nodes = graph["nodes"]

        char_id = detect_char_id(nodes)
        visible_ids = visible_ids_for_frame(nodes)
        action_text, parsed_action_ids = action_text_and_ids(actions[step_idx])

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
                "step_idx": step_idx,
                "frame_file": str(Path("frames") / frame_file),
                "action_text": action_text,
                "parsed_action_object_ids": parsed_action_ids,
                "highlighted_action_object_ids": action_ids,
                "goal_ids": goal_ids,
                "char_id": char_id,
            }
        )

    meta = {
        "data_json": str(data_json),
        "split_key": split_key,
        "episode_index": selected_index,
        "episode_name": episode_name,
        "task_name": task_name,
        "goal": goal_list,
        "goal_class_names": goal_class_names,
        "num_graph_frames": len(graphs),
        "num_actions": len(actions),
        "num_rendered_steps": total_steps,
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
        except subprocess.CalledProcessError as e:
            print("ffmpeg failed; frames were rendered but video export failed: {}".format(repr(e)), file=sys.stderr)
            raise

    print("Rendered {} frames to {}".format(total_steps, frames_dir))
    if not args.skip_video and video_path.exists():
        print("Video written to {}".format(video_path))
    print("Metadata: {}".format(out_dir / "episode_meta.json"))
    print("Steps manifest: {}".format(out_dir / "steps.json"))


if __name__ == "__main__":
    main()
