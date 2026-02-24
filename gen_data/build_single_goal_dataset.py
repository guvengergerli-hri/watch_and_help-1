import argparse
import copy
import json
import pickle
from collections import Counter
from pathlib import Path


def _get_agent_goal(task_goal, agent_id=0):
    if agent_id in task_goal:
        return task_goal[agent_id]
    if str(agent_id) in task_goal:
        return task_goal[str(agent_id)]
    return {}


def _predicate_to_class_predicate(predicate, id2class):
    elems = predicate.split("_")
    if len(elems) < 3:
        return predicate

    relation, obj_name, target_token = elems[0], elems[1], elems[2]
    if target_token.isdigit():
        if relation == "holds":
            target_name = "character"
        else:
            target_name = id2class.get(int(target_token), target_token)
    else:
        target_name = target_token

    return f"{relation}_{obj_name}_{target_name}"


def _extract_positive_goals(task_goal, prefixes):
    positives = []
    for predicate, count in task_goal.items():
        if count <= 0:
            continue
        if prefixes and predicate.split("_")[0] not in prefixes:
            continue
        positives.append((predicate, int(count)))
    return positives


def build_single_goal_dataset(
    input_path,
    output_path,
    prefixes,
    force_count_one,
    keep_only_goal_fields,
):
    with open(input_path, "rb") as f:
        env_task_set = pickle.load(f)

    new_env_task_set = []
    task_counter = 0
    source_task_counter = Counter()
    single_goal_counter = Counter()
    goals_per_episode = []

    for env_task in env_task_set:
        source_task_counter[env_task.get("task_name", "unknown")] += 1
        init_graph = env_task["init_graph"]
        id2class = {node["id"]: node["class_name"] for node in init_graph["nodes"]}

        task_goal = _get_agent_goal(env_task["task_goal"], agent_id=0)
        positive_goals = _extract_positive_goals(task_goal, prefixes)
        goals_per_episode.append(len(positive_goals))

        for goal_idx, (predicate, goal_count) in enumerate(positive_goals):
            count_value = 1 if force_count_one else goal_count
            class_predicate = _predicate_to_class_predicate(predicate, id2class)
            single_goal_counter[class_predicate] += 1

            if keep_only_goal_fields:
                new_env_task = {
                    "env_id": env_task["env_id"],
                    "init_graph": copy.deepcopy(env_task["init_graph"]),
                    "init_rooms": copy.deepcopy(env_task["init_rooms"]),
                    "level": env_task.get("level", 0),
                }
            else:
                new_env_task = copy.deepcopy(env_task)

            new_env_task.update(
                {
                    "task_id": task_counter,
                    "task_name": env_task.get("task_name", "task"),
                    "task_goal": {0: {predicate: count_value}, 1: {predicate: count_value}},
                    "goal_class": {class_predicate: count_value},
                    "pred_str": f"{class_predicate}.{count_value}",
                    "source_task_id": env_task.get("task_id", None),
                    "source_task_name": env_task.get("task_name", None),
                    "source_goal_count": goal_count,
                    "single_goal_index": goal_idx,
                    "single_goal_predicate": predicate,
                    "single_goal_class_predicate": class_predicate,
                    "single_goal_count": count_value,
                }
            )

            new_env_task_set.append(new_env_task)
            task_counter += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(new_env_task_set, f)

    stats = {
        "source_dataset_path": str(input_path),
        "output_dataset_path": str(output_path),
        "source_num_episodes": len(env_task_set),
        "output_num_episodes": len(new_env_task_set),
        "force_count_one": force_count_one,
        "goal_prefixes": sorted(prefixes),
        "goals_per_source_episode": {
            "min": min(goals_per_episode) if goals_per_episode else 0,
            "max": max(goals_per_episode) if goals_per_episode else 0,
            "avg": (sum(goals_per_episode) / len(goals_per_episode)) if goals_per_episode else 0.0,
        },
        "source_task_name_counts": dict(sorted(source_task_counter.items())),
        "single_goal_predicate_counts": dict(sorted(single_goal_counter.items())),
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Built single-goal dataset: {output_path}")
    print(f"Episodes: {len(env_task_set)} -> {len(new_env_task_set)}")
    print(f"Stats written to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build a single-goal Alice dataset from an env_task_set pickle.")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/test_env_set_help.pik",
        help="Input env_task_set pickle file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/test_env_set_help_single_goal.pik",
        help="Output single-goal env_task_set pickle file.",
    )
    parser.add_argument(
        "--goal-prefixes",
        type=str,
        default="on,inside,holds,sit",
        help="Comma-separated list of predicate prefixes to keep.",
    )
    parser.add_argument(
        "--force-count-one",
        action="store_true",
        default=False,
        help="Force each single-goal predicate count to 1.",
    )
    parser.add_argument(
        "--keep-only-goal-fields",
        action="store_true",
        default=False,
        help="Write a compact output dataset with only fields needed for rollouts.",
    )
    args = parser.parse_args()

    prefixes = {x.strip() for x in args.goal_prefixes.split(",") if x.strip()}
    build_single_goal_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        prefixes=prefixes,
        force_count_one=args.force_count_one,
        keep_only_goal_fields=args.keep_only_goal_fields,
    )


if __name__ == "__main__":
    main()
