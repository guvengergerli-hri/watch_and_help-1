#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Headless-safe backend before importing helper modules that may touch matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pillow is required for Unity overlay rendering: {}".format(repr(e)))

if __package__:
    from .visualize_dataset_episode_topdown import _safe_name, run_ffmpeg
    from .utils_environment import convert_action
else:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils.visualize_dataset_episode_topdown import _safe_name, run_ffmpeg
    from utils.utils_environment import convert_action

from arguments import get_args
from algos.a2c_mp import A2C as A2C_MP
from algos.arena_mp2 import ArenaMP
from agents import HRL_agent, MCTS_agent
from envs.unity_environment import UnityEnvironment
from utils import utils_rl_agent


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print("[hybrid-unity-viz {}] {}".format(ts, msg), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one saved hybrid RL policy episode in Unity and render per-step Unity snapshots."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved hybrid policy checkpoint (.pt).")
    parser.add_argument(
        "--unity-executable",
        type=str,
        default="../executable/linux_exec/linux_exec.v2.2.4.x86_64",
        help="Path to Unity executable.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/test_env_set_help.pik",
        help="Path to RL evaluation dataset (env_task_set pickle).",
    )
    parser.add_argument(
        "--task-set",
        type=str,
        default="full",
        help="Task filter (e.g., full/read_book). Matches testing_agents/test_hybrid.py behavior.",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index after task-set filtering.")
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="Optional episode selector. Matches either task_name or logs_agent_<task_id>_<task_name>.",
    )
    parser.add_argument("--base-port", type=int, default=8689, help="Unity base port.")
    parser.add_argument("--x-display", type=str, default=None, help="X display (e.g., 1) for Unity executable startup.")
    parser.add_argument("--no-graphics", action="store_true", default=False, help="Launch Unity with no_graphics=True.")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id override for policy inference.")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Force CPU inference.")

    parser.add_argument(
        "--use-alice",
        action="store_true",
        default=True,
        help="Use Alice+Bob hybrid setup (default).",
    )
    parser.add_argument("--no-use-alice", action="store_false", dest="use_alice")
    parser.add_argument(
        "--deterministic-policy",
        action="store_true",
        default=True,
        help="Use greedy policy actions (mode) for visualization reproducibility.",
    )
    parser.add_argument("--stochastic-policy", action="store_false", dest="deterministic_policy")
    parser.add_argument("--seed", type=int, default=0, help="Seed for agent rollouts.")
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=250,
        help="Max Unity episode length (env-side).",
    )
    parser.add_argument(
        "--max-number-steps",
        type=int,
        default=250,
        help="Max RL decision steps in ArenaMP rollout.",
    )

    parser.add_argument(
        "--capture-char-index",
        type=int,
        default=None,
        help="Character index to capture. Defaults to RL policy character index.",
    )
    parser.add_argument(
        "--camera-index-within-char",
        type=int,
        default=1,
        help="Character camera slot index (0=top, 1=front in wrappers).",
    )
    parser.add_argument("--capture-phase", type=str, default="before", choices=["before", "after"])
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--processing-time-limit", type=int, default=20)
    parser.add_argument("--time-scale", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional replay/capture cap.")

    parser.add_argument("--out-dir", type=str, default=None, help="Output dir for logs/frames/video.")
    parser.add_argument("--skip-video", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    parser.add_argument("--stop-on-action-failure", action="store_true", default=False)

    return parser.parse_args()


def _make_rl_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    # Build args object with the repo's existing defaults first, then override.
    prev_argv = sys.argv[:]
    try:
        sys.argv = [prev_argv[0]]
        rl_args = get_args()
    finally:
        sys.argv = prev_argv

    rl_args.max_episode_length = int(cli_args.max_episode_length)
    rl_args.max_number_steps = int(cli_args.max_number_steps)
    rl_args.num_per_apartment = 20
    rl_args.mode = "hybrid_truegoal"
    rl_args.evaluation = True
    rl_args.use_alice = bool(cli_args.use_alice)
    rl_args.obs_type = "partial"
    rl_args.dataset_path = str(cli_args.dataset_path)
    rl_args.task_set = str(cli_args.task_set)
    rl_args.executable_file = str(cli_args.unity_executable)
    rl_args.base_port = int(cli_args.base_port)
    rl_args.simulator_type = "unity"
    rl_args.num_processes = 1
    rl_args.load_model = str(cli_args.checkpoint)
    rl_args.no_cuda = bool(cli_args.no_cuda)

    if cli_args.gpu_id is not None:
        rl_args.gpu_id = int(cli_args.gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu_id)

    rl_args.cuda = (not rl_args.no_cuda) and torch.cuda.is_available()
    if rl_args.cuda:
        torch.cuda.set_device(0)

    return rl_args


def _load_env_task_set(dataset_path: Path, task_set: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "rb") as f:
        env_task_set = pickle.load(f)

    if not isinstance(env_task_set, list):
        raise TypeError("Expected env_task_set list in {}".format(dataset_path))

    if task_set != "full":
        env_task_set = [env_task for env_task in env_task_set if env_task.get("task_name") == task_set]

    if len(env_task_set) == 0:
        raise ValueError("No episodes found after task-set filtering (task_set={}).".format(task_set))

    return env_task_set


def _resolve_episode_index(
    env_task_set: Sequence[Dict[str, Any]],
    episode_index: int,
    episode_name: Optional[str],
) -> int:
    if episode_name is not None:
        target = str(episode_name)
        for idx, env_task in enumerate(env_task_set):
            task_id = env_task.get("task_id")
            task_name = env_task.get("task_name")
            log_name = "logs_agent_{}_{}".format(task_id, task_name)
            if target == str(task_name) or target == log_name:
                return int(idx)
        raise KeyError("Episode name '{}' not found after filtering.".format(target))

    if episode_index < 0 or episode_index >= len(env_task_set):
        raise IndexError("episode-index {} out of range [0, {}).".format(episode_index, len(env_task_set)))
    return int(episode_index)


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


def _capture_camera_rgb(comm: Any, camera_index: int, image_width: int, image_height: int) -> np.ndarray:
    ok, images = comm.camera_image(
        [int(camera_index)],
        mode="normal",
        image_width=int(image_width),
        image_height=int(image_height),
    )
    if not ok:
        raise RuntimeError("Unity camera_image failed for camera {}".format(camera_index))
    if not isinstance(images, list) or len(images) == 0:
        raise RuntimeError("Unity camera_image returned no images")

    img = images[0]
    if not isinstance(img, np.ndarray):
        raise TypeError("Unity camera_image returned non-array image: {}".format(type(img).__name__))
    if img.ndim == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] >= 3:
        # Decoded by OpenCV on comm side, so channel order is BGR.
        img_rgb = img[:, :, :3][:, :, ::-1]
    else:
        raise ValueError("Unsupported image shape {}".format(tuple(img.shape)))

    return np.ascontiguousarray(img_rgb)


def _build_overlay_text(
    *,
    checkpoint: str,
    task_name: str,
    task_id: Any,
    step_idx: int,
    total_steps: int,
    capture_phase: str,
    focus_action: str,
    action_dict: Dict[int, str],
    action_exec_success: Optional[bool],
    action_exec_message: Optional[str],
) -> str:
    status = "unknown"
    if action_exec_success is True:
        status = "ok"
    elif action_exec_success is False:
        status = "failed"

    action_pairs = ["char{}: {}".format(aid, act) for aid, act in sorted(action_dict.items())]
    action_line = " | ".join(action_pairs) if len(action_pairs) > 0 else "<no-op>"
    wrapped_focus = textwrap.fill(focus_action if focus_action else "<none>", width=92)
    wrapped_all = textwrap.fill(action_line, width=92)

    lines = [
        "checkpoint: {}".format(Path(checkpoint).name),
        "task: {} (id={})".format(task_name, task_id),
        "step: {}/{} | capture_phase={} | action_exec={}".format(
            int(step_idx),
            max(int(total_steps) - 1, 0),
            str(capture_phase),
            status,
        ),
        "focus action: {}".format(wrapped_focus),
        "all actions: {}".format(wrapped_all),
    ]
    if action_exec_message:
        lines.append("message: {}".format(textwrap.shorten(str(action_exec_message), width=180, placeholder="...")))
    return "\n".join(lines)


def _draw_text_overlay_and_save(
    image_rgb: np.ndarray,
    frame_path: Path,
    overlay_text: str,
    action_exec_success: Optional[bool],
) -> None:
    img = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img, mode="RGBA")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 8
    try:
        text_bbox = draw.multiline_textbbox((pad, pad), overlay_text, font=font, spacing=2)
    except Exception:
        lines = overlay_text.splitlines() or [""]
        max_w = 0
        line_h = 12
        for line in lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                w = max(0, int(bbox[2] - bbox[0]))
                h = max(0, int(bbox[3] - bbox[1]))
            except Exception:
                w, h = draw.textsize(line, font=font)
            max_w = max(max_w, int(w))
            line_h = max(line_h, int(h))
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * 2
        text_bbox = (pad, pad, pad + max_w, pad + total_h)

    bg_color = (0, 0, 0, 180)
    outline = (255, 255, 255, 180)
    if action_exec_success is False:
        outline = (255, 80, 80, 220)

    rect_xy = (
        max(0, text_bbox[0] - pad),
        max(0, text_bbox[1] - pad),
        min(img.width - 1, text_bbox[2] + pad),
        min(img.height - 1, text_bbox[3] + pad),
    )
    try:
        draw.rounded_rectangle(rect_xy, radius=6, fill=bg_color, outline=outline, width=2)
    except Exception:
        draw.rectangle(rect_xy, fill=bg_color, outline=outline, width=2)

    draw.multiline_text((pad, pad), overlay_text, fill=(255, 255, 255, 255), font=font, spacing=2)
    img.save(str(frame_path))


def _build_paths(args: argparse.Namespace, checkpoint_path: Path, task_name: str, episode_index: int) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)
    return (
        Path("outputs")
        / "rl_policy_unity_viz"
        / _safe_name(checkpoint_path.stem)
        / "ep_{:04d}_{}".format(int(episode_index), _safe_name(task_name))
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset_path)
    unity_exec = Path(args.unity_executable)

    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    if not dataset_path.is_file():
        raise FileNotFoundError("Dataset path not found: {}".format(dataset_path))
    if not unity_exec.is_file():
        raise FileNotFoundError("Unity executable not found: {}".format(unity_exec))

    _log("Loading RL setup and dataset")
    rl_args = _make_rl_args(args)
    env_task_set = _load_env_task_set(dataset_path=dataset_path, task_set=str(args.task_set))
    episode_index = _resolve_episode_index(env_task_set, int(args.episode_index), args.episode_name)

    selected = env_task_set[episode_index]
    task_name = str(selected.get("task_name", "unknown"))
    task_id = selected.get("task_id")
    env_id = selected.get("env_id")

    out_dir = _build_paths(args, checkpoint_path=checkpoint_path, task_name=task_name, episode_index=episode_index)
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

    executable_args = {
        "file_name": str(unity_exec),
        "x_display": 0 if args.x_display in (None, "") else args.x_display,
        "no_graphics": bool(args.no_graphics),
    }

    _log("Building Unity env + agents")
    agent_goal = "full"
    num_agents = 1
    agent_goals = [agent_goal]
    if bool(args.use_alice):
        num_agents = 2
        observation_types = ["partial", rl_args.obs_type]
        agent_goals.append(agent_goal)
        rl_agent_id = 2
        rl_char_index = 1
    else:
        observation_types = [rl_args.obs_type]
        rl_agent_id = 1
        rl_char_index = 0

    def env_fn(env_id_local: int) -> UnityEnvironment:
        return UnityEnvironment(
            num_agents=num_agents,
            max_episode_length=rl_args.max_episode_length,
            port_id=env_id_local,
            env_task_set=env_task_set,
            agent_goals=agent_goals,
            observation_types=observation_types,
            use_editor=rl_args.use_editor,
            executable_args=executable_args,
            base_port=rl_args.base_port,
            seed=None,
        )

    graph_helper = utils_rl_agent.GraphHelper(
        max_num_objects=rl_args.max_num_objects,
        max_num_edges=rl_args.max_num_edges,
        current_task=None,
        simulator_type=rl_args.simulator_type,
    )

    def mcts_agent_fn(arena_id: int, env: UnityEnvironment):
        args_mcts = dict(
            recursive=False,
            max_episode_length=5,
            num_simulation=100,
            max_rollout_steps=5,
            c_init=0.1,
            c_base=1000000,
            num_samples=1,
            num_processes=1,
            logging=True,
            logging_graphs=True,
            agent_id=1,
            char_index=0,
        )
        return MCTS_agent(**args_mcts)

    def hrl_agent_fn(arena_id: int, env: UnityEnvironment):
        args_agent2 = {
            "agent_id": rl_agent_id,
            "char_index": rl_agent_id - 1,
            "args": rl_args,
            "graph_helper": graph_helper,
            "seed": arena_id,
        }
        return HRL_agent(**args_agent2)

    agents = [hrl_agent_fn]
    if bool(args.use_alice):
        agents = [mcts_agent_fn] + agents

    arena = ArenaMP(rl_args.max_number_steps, 0, env_fn, agents)
    a2c = A2C_MP([arena], graph_helper, rl_args)

    saved_info: Dict[str, Any]
    success: bool
    steps: int
    replay_steps_rendered = 0

    try:
        _log("Loading policy checkpoint: {}".format(checkpoint_path))
        a2c.load_model(str(checkpoint_path))

        _log("Running policy rollout on selected episode")
        for idx, agent in enumerate(arena.agents):
            agent.seed = int(args.seed) + idx
            if hasattr(agent, "epsilon"):
                agent.epsilon = 0.0
            if bool(args.deterministic_policy) and hasattr(agent, "deterministic"):
                agent.deterministic = True

        arena.reset(task_id=episode_index)
        success, steps, saved_info = arena.run()

        saved_info["policy_checkpoint"] = str(checkpoint_path)
        saved_info["dataset_path"] = str(dataset_path)
        saved_info["task_set_filter"] = str(args.task_set)
        saved_info["episode_index"] = int(episode_index)
        saved_info["episode_task_name"] = task_name
        saved_info["episode_task_id"] = task_id
        saved_info["episode_env_id"] = env_id
        saved_info["run_seed"] = int(args.seed)
        saved_info["deterministic_policy"] = bool(args.deterministic_policy)
        saved_info["use_alice"] = bool(args.use_alice)

        _log("Saving rollout artifacts")
        with open(out_dir / "policy_episode_log.pik", "wb") as f:
            pickle.dump(saved_info, f)

        summary = {
            "policy_checkpoint": str(checkpoint_path),
            "dataset_path": str(dataset_path),
            "task_set_filter": str(args.task_set),
            "episode_index": int(episode_index),
            "task_id": task_id,
            "task_name": task_name,
            "env_id": env_id,
            "finished": bool(success),
            "steps": int(steps),
            "use_alice": bool(args.use_alice),
            "rl_char_index": int(rl_char_index),
            "deterministic_policy": bool(args.deterministic_policy),
            "run_seed": int(args.seed),
        }
        with open(out_dir / "episode_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        _log("Resetting Unity scene for replay capture")
        arena.env.reset(task_id=episode_index)

        capture_char_index = int(args.capture_char_index) if args.capture_char_index is not None else int(rl_char_index)
        if capture_char_index < 0 or capture_char_index >= int(num_agents):
            raise ValueError(
                "capture-char-index {} invalid for num_agents {}".format(capture_char_index, num_agents)
            )

        static_camera_count = int(getattr(arena.env, "offset_cameras", 0))
        num_cam_per_agent = int(getattr(arena.env, "num_camera_per_agent", 1))
        camera_index = static_camera_count + capture_char_index * num_cam_per_agent + int(args.camera_index_within_char)

        _log(
            "Replay camera index={} (static={} + char={}*{} + slot={})".format(
                camera_index,
                static_camera_count,
                capture_char_index,
                num_cam_per_agent,
                int(args.camera_index_within_char),
            )
        )

        action_map = _normalize_action_map(saved_info.get("action", {}))
        if len(action_map) == 0:
            raise RuntimeError("No action sequence found in rollout log")

        total_steps = max([len(seq) for seq in action_map.values()])
        if args.max_steps is not None:
            total_steps = min(total_steps, int(args.max_steps))
        if total_steps <= 0:
            raise RuntimeError("No steps to replay")

        steps_manifest: List[Dict[str, Any]] = []

        _log("Starting Unity replay capture for {} steps (phase={})".format(total_steps, args.capture_phase))
        for step_idx in range(total_steps):
            if step_idx == 0 or (step_idx + 1) % 10 == 0 or step_idx == total_steps - 1:
                _log("Replay capture progress: step {}/{}".format(step_idx + 1, total_steps))

            per_agent_actions: Dict[int, str] = {}
            for aid, seq in action_map.items():
                if step_idx < len(seq):
                    action_val = seq[step_idx]
                    if isinstance(action_val, str) and action_val.strip():
                        per_agent_actions[int(aid)] = str(action_val)

            focus_action = per_agent_actions.get(capture_char_index)
            if focus_action is None:
                focus_action = "<none>"

            action_exec_success: Optional[bool] = None
            action_exec_message: Optional[str] = None

            if args.capture_phase == "before":
                image_rgb = _capture_camera_rgb(
                    comm=arena.env.comm,
                    camera_index=camera_index,
                    image_width=int(args.image_width),
                    image_height=int(args.image_height),
                )

            if len(per_agent_actions) > 0:
                script_list = convert_action(per_agent_actions)
                action_exec_success, action_exec_message = arena.env.comm.render_script(
                    script_list,
                    recording=False,
                    skip_animation=False,
                    processing_time_limit=int(args.processing_time_limit),
                    time_scale=float(args.time_scale),
                )
            else:
                script_list = [""]

            if args.capture_phase == "after":
                image_rgb = _capture_camera_rgb(
                    comm=arena.env.comm,
                    camera_index=camera_index,
                    image_width=int(args.image_width),
                    image_height=int(args.image_height),
                )

            overlay_text = _build_overlay_text(
                checkpoint=str(checkpoint_path),
                task_name=task_name,
                task_id=task_id,
                step_idx=step_idx,
                total_steps=total_steps,
                capture_phase=str(args.capture_phase),
                focus_action=str(focus_action),
                action_dict=per_agent_actions,
                action_exec_success=action_exec_success,
                action_exec_message=action_exec_message,
            )
            frame_name = "frame_{:04d}.png".format(step_idx)
            frame_path = frames_dir / frame_name
            _draw_text_overlay_and_save(
                image_rgb=image_rgb,
                frame_path=frame_path,
                overlay_text=overlay_text,
                action_exec_success=action_exec_success,
            )

            steps_manifest.append(
                {
                    "step_idx": int(step_idx),
                    "frame_file": str(Path("frames") / frame_name),
                    "capture_phase": str(args.capture_phase),
                    "camera_index": int(camera_index),
                    "capture_char_index": int(capture_char_index),
                    "actions": {str(k): v for k, v in sorted(per_agent_actions.items())},
                    "focus_action": str(focus_action),
                    "script_list": script_list,
                    "action_exec_success": None if action_exec_success is None else bool(action_exec_success),
                    "action_exec_message": None if action_exec_message is None else str(action_exec_message),
                }
            )
            replay_steps_rendered += 1

            if action_exec_success is False and bool(args.stop_on_action_failure):
                _log("Stopping replay early due to action failure at step {}".format(step_idx))
                break

        with open(out_dir / "replay_steps.json", "w") as f:
            json.dump(steps_manifest, f, indent=2)

        with open(out_dir / "replay_meta.json", "w") as f:
            json.dump(
                {
                    "capture_phase": str(args.capture_phase),
                    "camera_index": int(camera_index),
                    "capture_char_index": int(capture_char_index),
                    "image_width": int(args.image_width),
                    "image_height": int(args.image_height),
                    "processing_time_limit": int(args.processing_time_limit),
                    "time_scale": float(args.time_scale),
                    "replay_steps_rendered": int(replay_steps_rendered),
                    "fps": int(args.fps),
                },
                f,
                indent=2,
            )

        video_path = out_dir / "episode_unity_policy.mp4"
        if not bool(args.skip_video):
            try:
                run_ffmpeg(frames_dir=frames_dir, out_path=video_path, fps=int(args.fps))
            except FileNotFoundError as e:
                _log("ffmpeg not found; frames rendered but video skipped: {}".format(repr(e)))

        print("Policy log: {}".format(out_dir / "policy_episode_log.pik"))
        print("Episode summary: {}".format(out_dir / "episode_summary.json"))
        print("Replay meta: {}".format(out_dir / "replay_meta.json"))
        print("Replay steps: {}".format(out_dir / "replay_steps.json"))
        print("Frames: {}".format(frames_dir))
        if (not bool(args.skip_video)) and video_path.exists():
            print("Video: {}".format(video_path))

    finally:
        _log("Closing Unity arena")
        try:
            arena.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
