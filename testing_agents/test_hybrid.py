import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from envs.unity_environment import UnityEnvironment
from agents import MCTS_agent, HRL_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from algos.a2c_mp import A2C as A2C_MP
from utils import utils_rl_agent
import ray


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print("[test-hybrid {}] {}".format(ts, msg), flush=True)


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


def _run_subprocess(cmd: list, label: str) -> int:
    _log("Launching {}.".format(label))
    _log("Command: {}".format(" ".join(cmd)))
    try:
        completed = subprocess.run(cmd, check=False)
    except Exception as e:
        _log("{} failed to start: {}".format(label, repr(e)))
        return 1
    if completed.returncode != 0:
        _log("{} failed with return code {}.".format(label, completed.returncode))
    else:
        _log("{} finished successfully.".format(label))
    return int(completed.returncode)


def _maybe_visualize_selected_eval(
    args: Any,
    iter_id: int,
    episode_id: int,
    env_task: Dict[str, Any],
    eval_log_path: Path,
    viz_run_root: Path,
) -> None:
    if not args.viz_eval_enable:
        return
    if int(iter_id) != int(args.viz_try_index):
        return
    if int(episode_id) != int(args.viz_episode_index):
        return

    task_name = env_task.get("task_name", "unknown")
    task_id = env_task.get("task_id", "unknown")
    env_id = env_task.get("env_id", "unknown")
    episode_tag = "try_{:02d}_ep_{:04d}_{}".format(int(iter_id), int(episode_id), _safe_name(task_name))
    out_root = viz_run_root / episode_tag
    unity_out_dir = out_root / "unity"
    topdown_out_dir = out_root / "topdown"
    out_root.mkdir(parents=True, exist_ok=True)

    _log(
        "Selected eval reached for visualization: try={} episode={} task={} (task_id={}, env_id={}).".format(
            int(iter_id), int(episode_id), task_name, task_id, env_id
        )
    )

    unity_rollout_log: Optional[Path] = None
    unity_rc: Optional[int] = None
    topdown_rc: Optional[int] = None

    if args.viz_mode in ("unity", "both"):
        if not str(args.load_model):
            _log("Skipping Unity visualization because --load-model is empty.")
        else:
            viz_base_port = int(args.base_port) + int(args.viz_port_offset)
            unity_cmd = [
                sys.executable,
                "-m",
                "utils.visualize_hybrid_policy_episode_unity",
                "--checkpoint",
                str(args.load_model),
                "--unity-executable",
                str(args.executable_file),
                "--dataset-path",
                str(args.dataset_path),
                "--task-set",
                str(args.task_set),
                "--episode-index",
                str(episode_id),
                "--base-port",
                str(viz_base_port),
                "--seed",
                str(iter_id),
                "--max-episode-length",
                str(args.max_episode_length),
                "--max-number-steps",
                str(args.max_number_steps),
                "--camera-index-within-char",
                str(args.viz_camera_index_within_char),
                "--capture-phase",
                str(args.viz_capture_phase),
                "--processing-time-limit",
                str(args.viz_processing_time_limit),
                "--time-scale",
                str(args.viz_time_scale),
                "--fps",
                str(args.viz_fps),
                "--out-dir",
                str(unity_out_dir),
                "--overwrite",
            ]
            if args.viz_x_display not in (None, ""):
                unity_cmd.extend(["--x-display", str(args.viz_x_display)])
            if args.viz_no_graphics:
                unity_cmd.append("--no-graphics")
            if args.viz_skip_video:
                unity_cmd.append("--skip-video")
            if args.use_alice:
                unity_cmd.append("--use-alice")
            else:
                unity_cmd.append("--no-use-alice")
            if args.viz_deterministic_policy:
                unity_cmd.append("--deterministic-policy")
            else:
                unity_cmd.append("--stochastic-policy")
            if args.viz_capture_char_index is not None:
                unity_cmd.extend(["--capture-char-index", str(args.viz_capture_char_index)])
            if args.viz_max_steps is not None:
                unity_cmd.extend(["--max-steps", str(args.viz_max_steps)])

            unity_rc = _run_subprocess(unity_cmd, label="Unity visualization")
            candidate_rollout = unity_out_dir / "policy_episode_log.pik"
            if unity_rc == 0 and candidate_rollout.is_file():
                unity_rollout_log = candidate_rollout
            else:
                _log("Unity rollout log missing; topdown fallback will use eval log pickle.")

    if args.viz_mode in ("topdown", "both"):
        topdown_source = unity_rollout_log if unity_rollout_log is not None else eval_log_path
        topdown_cmd = [
            sys.executable,
            "-m",
            "utils.visualize_rl_rollout_episode_topdown",
            "--rollout-log",
            str(topdown_source),
            "--out-dir",
            str(topdown_out_dir),
            "--episode-name",
            "logs_agent_{}_{}".format(task_id, task_name),
            "--fps",
            str(args.viz_fps),
            "--dpi",
            str(args.viz_dpi),
            "--overwrite",
        ]
        if args.viz_skip_video:
            topdown_cmd.append("--skip-video")
        if args.viz_max_steps is not None:
            topdown_cmd.extend(["--max-steps", str(args.viz_max_steps)])

        topdown_rc = _run_subprocess(topdown_cmd, label="Topdown visualization")

    summary = {
        "try_index": int(iter_id),
        "episode_index": int(episode_id),
        "task_name": task_name,
        "task_id": task_id,
        "env_id": env_id,
        "mode": str(args.viz_mode),
        "eval_log_path": str(eval_log_path),
        "unity_rollout_log": None if unity_rollout_log is None else str(unity_rollout_log),
        "unity_return_code": unity_rc,
        "topdown_return_code": topdown_rc,
        "out_root": str(out_root),
    }
    with open(out_root / "viz_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log("Visualization summary written: {}".format(out_root / "viz_summary.json"))


if __name__ == "__main__":
    args = get_args()

    args.max_episode_length = 250
    args.num_per_apartment = 20
    args.mode = "hybrid_truegoal"
    args.evaluation = True
    args.use_alice = True
    args.obs_type = "partial"
    if not args.dataset_path:
        args.dataset_path = "./dataset/test_env_set_help.pik"

    with open(args.dataset_path, "rb") as f:
        env_task_set = pickle.load(f)
    args.record_dir = "../test_results/multiBob_env_task_set_{}_{}".format(args.num_per_apartment, args.mode)
    executable_args = {
        "file_name": args.executable_file,
        "x_display": 0,
        "no_graphics": True,
    }

    if args.task_set != "full":
        env_task_set = [env_task for env_task in env_task_set if env_task["task_name"] == args.task_set]
    if len(env_task_set) == 0:
        raise RuntimeError("No episodes found after task-set filtering: {}".format(args.task_set))

    if args.viz_eval_enable:
        if args.viz_mode not in ("unity", "topdown", "both"):
            raise ValueError("Unsupported --viz-mode: {}".format(args.viz_mode))
        if args.viz_episode_index < 0 or args.viz_episode_index >= len(env_task_set):
            raise IndexError(
                "--viz-episode-index {} out of range [0, {}).".format(args.viz_episode_index, len(env_task_set))
            )

    agent_goal = "full"
    args.task_type = "full"
    num_agents = 1
    agent_goals = [agent_goal]
    if args.use_alice:
        num_agents += 1
        observation_types = ["partial", args.obs_type]
        agent_goals.append(agent_goal)
        rl_agent_id = 2
    else:
        rl_agent_id = 1
        observation_types = [args.obs_type]

    episode_ids = sorted(list(range(len(env_task_set))))
    num_tries = 5
    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]

    def env_fn(env_id):
        return UnityEnvironment(
            num_agents=num_agents,
            max_episode_length=args.max_episode_length,
            port_id=env_id,
            env_task_set=env_task_set,
            agent_goals=agent_goals,
            observation_types=observation_types,
            use_editor=args.use_editor,
            executable_args=executable_args,
            base_port=args.base_port,
            seed=None,
        )

    graph_helper = utils_rl_agent.GraphHelper(
        max_num_objects=args.max_num_objects,
        max_num_edges=args.max_num_edges,
        current_task=None,
        simulator_type=args.simulator_type,
    )

    def MCTS_agent_fn(arena_id, env):
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
        )
        args_mcts["agent_id"] = 1
        args_mcts["char_index"] = 0
        return MCTS_agent(**args_mcts)

    def HRL_agent_fn(arena_id, env):
        args_agent2 = {"agent_id": rl_agent_id, "char_index": rl_agent_id - 1, "args": args, "graph_helper": graph_helper}
        args_agent2["seed"] = arena_id
        return HRL_agent(**args_agent2)

    agents = [HRL_agent_fn]
    if args.use_alice:
        agents = [MCTS_agent_fn] + agents

    if args.num_processes > 1:
        ArenaMPRemote = ray.remote(ArenaMP)
        arenas = [ArenaMPRemote.remote(args.max_number_steps, arena_id, env_fn, agents) for arena_id in range(args.num_processes)]
        a2c = A2C_MP(arenas, graph_helper, args)
        raise NotImplementedError("test_hybrid visualization hooks currently support --num-processes 1 only.")
    else:
        arenas = [ArenaMP(args.max_number_steps, arena_id, env_fn, agents) for arena_id in range(args.num_processes)]
        a2c = A2C_MP(arenas, graph_helper, args)

    a2c.load_model(args.load_model)

    viz_run_root = Path(args.viz_out_root) / time.strftime("%Y%m%d_%H%M%S")
    viz_generated = False

    test_results = {}
    for iter_id in range(num_tries):
        seed = iter_id
        _log("Starting eval try {}/{} (seed={}).".format(iter_id + 1, num_tries, seed))

        for episode_id in tqdm(range(len(episode_ids))):
            log_file_name = (
                args.record_dir
                + "/logs_agent_{}_{}_{}.pik".format(
                    env_task_set[episode_id]["task_id"],
                    env_task_set[episode_id]["task_name"],
                    seed,
                )
            )
            eval_log_path = Path(log_file_name)

            if os.path.isfile(log_file_name):
                print("exsits")
                if (
                    args.viz_eval_enable
                    and not viz_generated
                    and int(iter_id) == int(args.viz_try_index)
                    and int(episode_id) == int(args.viz_episode_index)
                ):
                    _maybe_visualize_selected_eval(
                        args=args,
                        iter_id=iter_id,
                        episode_id=episode_id,
                        env_task=env_task_set[episode_id],
                        eval_log_path=eval_log_path,
                        viz_run_root=viz_run_root,
                    )
                    viz_generated = True
                continue

            for agent in arenas[0].agents:
                agent.seed = seed

            res = a2c.eval(episode_id)
            rollout_info = res[1][0]
            finished = bool(rollout_info.get("finished", False))
            action_bundle = rollout_info.get("action", {})
            if isinstance(action_bundle, dict):
                action_len = 0
                for seq in action_bundle.values():
                    if isinstance(seq, list):
                        action_len = max(action_len, len(seq))
                length = int(action_len)
            else:
                length = 0

            graph_list = rollout_info.get("graph", [])
            info_results = {
                "finished": finished,
                "L": length,
                "task_id": arenas[0].env.task_id,
                "env_id": arenas[0].env.env_id,
                "task_name": arenas[0].env.task_name,
                "gt_goals": arenas[0].env.task_goal[0],
                "goals_finished": rollout_info.get("goals_finished", []),
                "goals": arenas[0].env.task_goal,
                "obs": rollout_info.get("obs", []),
                "action": action_bundle,
                "graph": graph_list,
                "graph_seq": graph_list,
            }

            S[episode_id].append(finished)
            L[episode_id].append(length)
            test_results[episode_id] = {"S": S[episode_id], "L": L[episode_id]}
            Path(args.record_dir).mkdir(parents=True, exist_ok=True)
            with open(log_file_name, "wb") as flog:
                pickle.dump(info_results, flog)

            if (
                args.viz_eval_enable
                and not viz_generated
                and int(iter_id) == int(args.viz_try_index)
                and int(episode_id) == int(args.viz_episode_index)
            ):
                _maybe_visualize_selected_eval(
                    args=args,
                    iter_id=iter_id,
                    episode_id=episode_id,
                    env_task=env_task_set[episode_id],
                    eval_log_path=eval_log_path,
                    viz_run_root=viz_run_root,
                )
                viz_generated = True

        with open(args.record_dir + "/results_{}.pik".format(0), "wb") as f:
            pickle.dump(test_results, f)

    if args.viz_eval_enable and not viz_generated:
        _log(
            "Visualization target was not reached (try_index={}, episode_index={}).".format(
                args.viz_try_index, args.viz_episode_index
            )
        )
