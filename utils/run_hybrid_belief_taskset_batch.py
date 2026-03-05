#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print("[hybrid-batch {}] {}".format(ts, msg), flush=True)


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


def _run_subprocess(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    _log("Command: {}".format(" ".join(cmd)))
    try:
        completed = subprocess.run(cmd, check=False, env=env)
    except Exception as e:
        _log("Command failed to start: {}".format(repr(e)))
        return 1
    return int(completed.returncode)


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, list):
        raise TypeError("Expected dataset list at {}".format(path))
    out = [entry for entry in payload if isinstance(entry, dict)]
    if len(out) == 0:
        raise ValueError("No dict episodes found in dataset: {}".format(path))
    return out


def _collect_task_names(dataset: Sequence[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for item in dataset:
        task_name = item.get("task_name")
        if task_name is None:
            continue
        task_name_s = str(task_name)
        if task_name_s in seen:
            continue
        ordered.append(task_name_s)
        seen.add(task_name_s)
    return ordered


def _selected_dataset_rows(
    dataset: Sequence[Dict[str, Any]],
    task_name: str,
    episodes_per_task: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        if str(item.get("task_name")) != str(task_name):
            continue
        out.append(
            {
                "dataset_episode_index": int(idx),
                "task_name": str(item.get("task_name")),
                "task_id": item.get("task_id"),
                "env_id": item.get("env_id"),
            }
        )
        if len(out) >= int(episodes_per_task):
            break
    return out


def _load_rollout_header(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        return {}
    return {
        "task_name": payload.get("task_name"),
        "task_id": payload.get("task_id"),
        "env_id": payload.get("env_id"),
        "finished": payload.get("finished"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid rollouts in task subsets and render topdown videos for each selected episode."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Hybrid policy checkpoint (.pt)")
    parser.add_argument("--goal-cond-mode", type=str, default="belief", choices=["gt", "belief"])
    parser.add_argument(
        "--belief-vae-checkpoint",
        type=str,
        default="",
        help="Belief VAE checkpoint (.pt). Required when --goal-cond-mode belief.",
    )
    parser.add_argument("--dataset-path", type=str, default="dataset/test_env_set_help.pik")
    parser.add_argument(
        "--task-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional task_name allowlist. Default: all task names in dataset order.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--out-root", type=str, default="outputs/hybrid_belief_taskset_batch")
    parser.add_argument("--run-tag", type=str, default=None)

    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--executable-file", type=str, default="../executable/linux_exec/linux_exec.v2.2.4.x86_64")
    parser.add_argument("--base-port", type=int, default=8681)
    parser.add_argument("--task-port-stride", type=int, default=20)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-cuda", action="store_true", default=False)

    parser.add_argument("--belief-vae-device", type=str, default="cuda")
    parser.add_argument("--belief-context-dim", type=int, default=64)
    parser.add_argument("--belief-use-actions", action="store_true", default=True)
    parser.add_argument("--no-belief-use-actions", action="store_false", dest="belief_use_actions")
    parser.add_argument("--belief-action-source", type=str, default="alice", choices=["alice", "self", "none"])

    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-video", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if int(args.episodes_per_task) <= 0:
        raise ValueError("--episodes-per-task must be positive")
    if int(args.num_processes) != 1:
        raise ValueError("This batch wrapper currently supports --num-processes 1 only.")

    checkpoint_path = Path(args.checkpoint)
    belief_ckpt_path = Path(args.belief_vae_checkpoint) if str(args.belief_vae_checkpoint).strip() else None
    dataset_path = Path(args.dataset_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    if args.goal_cond_mode == "belief":
        if belief_ckpt_path is None:
            raise ValueError("--belief-vae-checkpoint is required when --goal-cond-mode belief.")
        if not belief_ckpt_path.is_file():
            raise FileNotFoundError("Belief checkpoint not found: {}".format(belief_ckpt_path))
    if not dataset_path.is_file():
        raise FileNotFoundError("Dataset not found: {}".format(dataset_path))

    repo_root = Path(__file__).resolve().parents[1]
    test_script = repo_root / "testing_agents" / "test_hybrid_quick.py"
    if not test_script.is_file():
        raise FileNotFoundError("Missing test script: {}".format(test_script))

    dataset = _load_dataset(dataset_path)
    all_task_names = _collect_task_names(dataset)
    if len(all_task_names) == 0:
        raise ValueError("No task_name entries found in dataset: {}".format(dataset_path))

    if args.task_names is None or len(args.task_names) == 0:
        selected_task_names = list(all_task_names)
    else:
        requested = [str(t) for t in args.task_names]
        missing = [t for t in requested if t not in all_task_names]
        if len(missing) > 0:
            raise KeyError("Requested task names not found in dataset: {}".format(missing))
        selected_task_names = requested

    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / _safe_name(run_tag)
    if run_dir.exists() and not bool(args.overwrite):
        raise FileExistsError("Output directory exists: {} (pass --overwrite to reuse).".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)

    rollouts_root = run_dir / "rollouts"
    viz_root = run_dir / "topdown"
    rollouts_root.mkdir(parents=True, exist_ok=True)
    viz_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    batch_summary: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint_path),
        "goal_cond_mode": str(args.goal_cond_mode),
        "belief_vae_checkpoint": None if belief_ckpt_path is None else str(belief_ckpt_path),
        "belief_action_source": str(args.belief_action_source),
        "dataset_path": str(dataset_path),
        "episodes_per_task": int(args.episodes_per_task),
        "selected_tasks": selected_task_names,
        "run_dir": str(run_dir),
        "tasks": [],
    }

    for task_idx, task_name in enumerate(selected_task_names):
        _log("Running task subset: {}".format(task_name))
        selected_rows = _selected_dataset_rows(dataset, task_name, int(args.episodes_per_task))
        task_record_dir = rollouts_root / _safe_name(task_name)
        task_viz_root = viz_root / _safe_name(task_name)
        task_record_dir.mkdir(parents=True, exist_ok=True)
        task_viz_root.mkdir(parents=True, exist_ok=True)

        task_base_port = int(args.base_port) + int(task_idx) * int(args.task_port_stride)

        test_cmd = [
            str(args.python_exe),
            str(test_script),
            "--load-model",
            str(checkpoint_path),
            "--num-processes",
            str(args.num_processes),
            "--base-port",
            str(task_base_port),
            "--executable_file",
            str(args.executable_file),
            "--simulator-type",
            "unity",
            "--obs_type",
            "partial",
            "--goal-cond-mode",
            str(args.goal_cond_mode),
            "--task-set",
            str(task_name),
            "--dataset_path",
            str(dataset_path),
            "--eval-max-episodes",
            str(args.episodes_per_task),
            "--record-dir-override",
            str(task_record_dir),
            "--overwrite-eval-logs",
        ]

        if args.gpu_id is not None:
            test_cmd.extend(["--gpu-id", str(args.gpu_id)])
        if bool(args.no_cuda):
            test_cmd.append("--no-cuda")
        if args.goal_cond_mode == "belief":
            if belief_ckpt_path is None:
                raise ValueError("Internal error: belief checkpoint missing in belief mode.")
            test_cmd.extend(
                [
                    "--belief-vae-checkpoint",
                    str(belief_ckpt_path),
                    "--belief-vae-device",
                    str(args.belief_vae_device),
                    "--belief-context-dim",
                    str(args.belief_context_dim),
                    "--belief-action-source",
                    str(args.belief_action_source),
                ]
            )
            if bool(args.belief_use_actions):
                test_cmd.append("--belief-use-actions")
            else:
                test_cmd.append("--no-belief-use-actions")

        test_rc = _run_subprocess(test_cmd, env=env)

        task_summary: Dict[str, Any] = {
            "task_name": str(task_name),
            "task_base_port": int(task_base_port),
            "selected_dataset_episodes": selected_rows,
            "record_dir": str(task_record_dir),
            "quick_test_cmd": test_cmd,
            "quick_test_return_code": int(test_rc),
            "belief_action_oov_report": str(task_record_dir / "belief_action_oov_report.json"),
            "rollouts": [],
        }

        log_files = sorted(task_record_dir.glob("logs_agent_*.pik"))
        if len(log_files) == 0:
            _log("No rollout logs found for task {} in {}".format(task_name, task_record_dir))

        for log_path in log_files:
            rollout_head = _load_rollout_header(log_path)
            if str(rollout_head.get("task_name", "")) != str(task_name):
                continue

            viz_out_dir = task_viz_root / _safe_name(log_path.stem)
            topdown_cmd = [
                str(args.python_exe),
                "-m",
                "utils.visualize_rl_rollout_episode_topdown",
                "--rollout-log",
                str(log_path),
                "--out-dir",
                str(viz_out_dir),
                "--fps",
                str(args.fps),
                "--dpi",
                str(args.dpi),
            ]
            if bool(args.skip_video):
                topdown_cmd.append("--skip-video")
            if args.max_steps is not None:
                topdown_cmd.extend(["--max-steps", str(args.max_steps)])
            if bool(args.overwrite):
                topdown_cmd.append("--overwrite")
            else:
                topdown_cmd.append("--no-overwrite")

            render_rc = _run_subprocess(topdown_cmd, env=env)

            rollout_summary = {
                "log_path": str(log_path),
                "task_id": rollout_head.get("task_id"),
                "env_id": rollout_head.get("env_id"),
                "finished": rollout_head.get("finished"),
                "topdown_cmd": topdown_cmd,
                "topdown_return_code": int(render_rc),
                "topdown_out_dir": str(viz_out_dir),
                "episode_meta": str(viz_out_dir / "episode_meta.json"),
                "steps_json": str(viz_out_dir / "steps.json"),
                "belief_seq_full": str(viz_out_dir / "belief_seq_full.json"),
                "video": str(viz_out_dir / "episode_topdown.mp4"),
            }
            task_summary["rollouts"].append(rollout_summary)

        batch_summary["tasks"].append(task_summary)

    summary_path = run_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    _log("Batch summary written: {}".format(summary_path))


if __name__ == "__main__":
    main()
