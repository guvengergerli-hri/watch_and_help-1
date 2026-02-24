import argparse
import json
import pickle
from glob import glob
from pathlib import Path


def _read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _goal_dict_for_agent(goal_bundle, agent_id=0):
    if isinstance(goal_bundle, dict):
        if agent_id in goal_bundle:
            return goal_bundle[agent_id]
        if str(agent_id) in goal_bundle:
            return goal_bundle[str(agent_id)]
    return {}


def _actions_for_agent(action_bundle, agent_id=0):
    if isinstance(action_bundle, dict):
        if agent_id in action_bundle:
            return action_bundle[agent_id]
        if str(agent_id) in action_bundle:
            return action_bundle[str(agent_id)]
    return []


def export_rollouts(log_dir, output_path, include_failures):
    log_paths = sorted(glob(str(Path(log_dir) / "logs_agent_*.pik")))
    demos = []
    task_name_counts = {}
    success_count = 0

    for log_path in log_paths:
        try:
            data = _read_pickle(log_path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        success = bool(data.get("finished", False))
        if not include_failures and not success:
            continue

        if success:
            success_count += 1

        goal_spec = _goal_dict_for_agent(data.get("goals", {}), agent_id=0)
        if len(goal_spec) == 1:
            single_goal_predicate, single_goal_count = list(goal_spec.items())[0]
        else:
            single_goal_predicate = data.get("single_goal_predicate")
            single_goal_count = data.get("single_goal_count")

        action_seq = _actions_for_agent(data.get("action", {}), agent_id=0)
        graph_seq = data.get("graph_seq", [])
        task_name = data.get("task_name", "unknown")
        task_name_counts[task_name] = task_name_counts.get(task_name, 0) + 1

        demos.append(
            {
                "demo_id": Path(log_path).stem,
                "source_log_path": str(log_path),
                "task_id": data.get("task_id"),
                "task_name": task_name,
                "env_id": data.get("env_id"),
                "success": success,
                "steps": len(action_seq),
                "goal_spec": goal_spec,
                "single_goal_predicate": single_goal_predicate,
                "single_goal_count": single_goal_count,
                "goals_finished_seq": data.get("goals_finished", []),
                "reward_seq": data.get("reward_seq", []),
                "action_seq": action_seq,
                "graph_seq": graph_seq,
                "init_graph": data.get("init_unity_graph"),
                "source_task_id": data.get("source_task_id"),
                "source_task_name": data.get("source_task_name"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    stats = {
        "log_dir": str(log_dir),
        "num_log_files_found": len(log_paths),
        "num_demos_exported": len(demos),
        "num_successful_demos_exported": success_count,
        "include_failures": include_failures,
        "task_name_counts": dict(sorted(task_name_counts.items())),
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Exported {len(demos)} demos to {output_path}")
    print(f"Stats written to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Alice rollout logs to a VAE-ready demo dataset.")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory that contains logs_agent_*.pik files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/alice_single_goal_vae_demos.pik",
        help="Output pickle path for exported demos.",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        default=False,
        help="Include failed demonstrations in the exported dataset.",
    )
    args = parser.parse_args()

    export_rollouts(
        log_dir=Path(args.log_dir),
        output_path=Path(args.output),
        include_failures=args.include_failures,
    )


if __name__ == "__main__":
    main()
