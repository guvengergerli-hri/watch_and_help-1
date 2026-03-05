import json
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .tensorizer import WatchGraphTensorizer


def _load_split(path: str, split_key: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        content = json.load(f)
    if split_key not in content:
        raise KeyError("Split key '{}' not found in {}. Available keys: {}".format(split_key, path, list(content.keys())))
    return content[split_key]


class WatchVAEDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split_key: str,
        tensorizer: WatchGraphTensorizer,
        max_seq_len: Optional[int] = None,
        min_seq_len: int = 1,
        use_actions: bool = False,
        max_demos: Optional[int] = None,
        stable_slots: bool = True,
        strict_action_indexing: bool = False,
    ):
        self.data_path = data_path
        self.split_key = split_key
        self.tensorizer = tensorizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = max(1, int(min_seq_len))
        self.use_actions = use_actions
        self.stable_slots = stable_slots
        self.strict_action_indexing = bool(strict_action_indexing)

        demos = _load_split(data_path, split_key)
        self.demos: List[Dict[str, Any]] = []

        actions_scanned = 0
        unique_canonical = set()
        unknown_counter: Counter = Counter()

        for demo_idx, demo in enumerate(demos):
            graphs = demo.get("graphs", [])
            actions = demo.get("valid_action_with_walk", [])

            if use_actions:
                seq_len = min(len(graphs), len(actions))
            else:
                seq_len = len(graphs)

            if self.max_seq_len is not None:
                seq_len = min(seq_len, self.max_seq_len)

            if seq_len < self.min_seq_len:
                continue

            if self.use_actions:
                demo_name = str(demo.get("name", "unknown"))
                demo_task_name = str(demo.get("task_name", "unknown"))
                for step_idx in range(seq_len):
                    source_context = {
                        "mode": "watch_dataset",
                        "data_path": str(self.data_path),
                        "split_key": str(self.split_key),
                        "demo_name": demo_name,
                        "task_name": demo_task_name,
                        "demo_index": int(demo_idx),
                        "step_idx": int(step_idx),
                    }
                    action_entry = actions[step_idx]
                    action_id, action_meta = self.tensorizer.action_index(
                        action_entry,
                        strict=False,
                        source_context=source_context,
                        return_details=True,
                    )
                    canonical_key = action_meta.get("canonical_key")
                    if canonical_key is not None:
                        canonical_key = str(canonical_key)
                        unique_canonical.add(canonical_key)
                        if canonical_key not in self.tensorizer.action_to_idx:
                            unknown_counter[canonical_key] += 1
                            if self.strict_action_indexing:
                                # Re-run in strict mode to raise structured diagnostics.
                                self.tensorizer.action_index(
                                    action_entry,
                                    strict=True,
                                    source_context=source_context,
                                )
                    elif self.strict_action_indexing and action_entry is not None:
                        # Non-null malformed actions should fail-fast in strict mode.
                        self.tensorizer.action_index(
                            action_entry,
                            strict=True,
                            source_context=source_context,
                        )
                    _ = action_id
                    actions_scanned += 1

            # Precompute stable slot-map once at dataset construction time to avoid
            # mutating per-sample state inside DataLoader worker processes.
            slot_map = (
                self.tensorizer.build_stable_slot_map(graphs[:seq_len])
                if self.stable_slots
                else None
            )

            self.demos.append(
                {
                    "name": demo.get("name", "unknown"),
                    "graphs": graphs,
                    "actions": actions,
                    "seq_len": seq_len,
                    "goal": demo.get("goal", []),
                    "task_name": demo.get("task_name", "unknown"),
                    "slot_map": slot_map,
                }
            )
            if max_demos is not None and len(self.demos) >= int(max_demos):
                break

        self.action_preflight = {
            "actions_scanned": int(actions_scanned),
            "unique_canonical_actions": int(len(unique_canonical)),
            "unknown_count": int(sum(unknown_counter.values())),
            "unknown_by_key": dict(sorted([(str(k), int(v)) for k, v in unknown_counter.items()], key=lambda kv: kv[0])),
        }

    def __len__(self) -> int:
        return len(self.demos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        demo = self.demos[index]
        seq_len = demo["seq_len"]

        class_seq = []
        coords_seq = []
        states_seq = []
        mask_seq = []
        action_seq = []

        slot_map = demo["slot_map"] if self.stable_slots else None

        for step_idx in range(seq_len):
            if self.stable_slots:
                frame = self.tensorizer.encode_nodes_with_slot_map(
                    demo["graphs"][step_idx],
                    slot_map=slot_map,
                    allow_new_ids=False,
                )
            else:
                frame = self.tensorizer.encode_nodes(demo["graphs"][step_idx])
            class_seq.append(frame["class_objects"])
            coords_seq.append(frame["object_coords"])
            states_seq.append(frame["states_objects"])
            mask_seq.append(frame["mask_object"])

            if self.use_actions:
                source_context = {
                    "mode": "watch_dataset",
                    "data_path": str(self.data_path),
                    "split_key": str(self.split_key),
                    "demo_name": str(demo.get("name", "unknown")),
                    "task_name": str(demo.get("task_name", "unknown")),
                    "step_idx": int(step_idx),
                    "dataset_index": int(index),
                }
                action_seq.append(
                    self.tensorizer.action_index(
                        demo["actions"][step_idx],
                        strict=bool(self.strict_action_indexing),
                        source_context=source_context,
                    )
                )

        output: Dict[str, Any] = {
            "dataset_index": int(index),
            "name": demo["name"],
            "task_name": demo["task_name"],
            "goal": demo["goal"],
            "seq_len": seq_len,
            "class_ids": torch.from_numpy(np.stack(class_seq, axis=0)).long(),
            "coords": torch.from_numpy(np.stack(coords_seq, axis=0)).float(),
            "states": torch.from_numpy(np.stack(states_seq, axis=0)).float(),
            "node_mask": torch.from_numpy(np.stack(mask_seq, axis=0)).float(),
        }

        if self.use_actions:
            output["action_ids"] = torch.tensor(action_seq, dtype=torch.long)

        return output

    def get_action_preflight_summary(self) -> Dict[str, Any]:
        return {
            "actions_scanned": int(self.action_preflight.get("actions_scanned", 0)),
            "unique_canonical_actions": int(self.action_preflight.get("unique_canonical_actions", 0)),
            "unknown_count": int(self.action_preflight.get("unknown_count", 0)),
            "unknown_by_key": dict(self.action_preflight.get("unknown_by_key", {})),
            "strict_action_indexing": bool(self.strict_action_indexing),
            "use_actions": bool(self.use_actions),
        }


def collate_watch_vae(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch provided to collate_watch_vae")

    batch_size = len(batch)
    lengths = [sample["seq_len"] for sample in batch]
    max_t = max(lengths)

    max_nodes = int(batch[0]["class_ids"].shape[1])
    num_states = int(batch[0]["states"].shape[2])

    class_ids = torch.zeros((batch_size, max_t, max_nodes), dtype=torch.long)
    coords = torch.zeros((batch_size, max_t, max_nodes, 6), dtype=torch.float32)
    states = torch.zeros((batch_size, max_t, max_nodes, num_states), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_t, max_nodes), dtype=torch.float32)
    time_mask = torch.zeros((batch_size, max_t), dtype=torch.float32)

    use_actions = "action_ids" in batch[0]
    if use_actions:
        action_ids = torch.zeros((batch_size, max_t), dtype=torch.long)
    else:
        action_ids = None

    names: List[str] = []
    task_names: List[str] = []
    goals: List[List[str]] = []
    dataset_indices = torch.zeros((batch_size,), dtype=torch.long)

    for batch_idx, sample in enumerate(batch):
        seq_len = sample["seq_len"]
        dataset_indices[batch_idx] = int(sample.get("dataset_index", batch_idx))
        class_ids[batch_idx, :seq_len] = sample["class_ids"]
        coords[batch_idx, :seq_len] = sample["coords"]
        states[batch_idx, :seq_len] = sample["states"]
        node_mask[batch_idx, :seq_len] = sample["node_mask"]
        time_mask[batch_idx, :seq_len] = 1.0

        if use_actions:
            action_ids[batch_idx, :seq_len] = sample["action_ids"]

        names.append(sample["name"])
        task_names.append(sample["task_name"])
        goals.append(sample["goal"])

    output = {
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "dataset_indices": dataset_indices,
        "time_mask": time_mask,
        "class_ids": class_ids,
        "coords": coords,
        "states": states,
        "node_mask": node_mask,
        "names": names,
        "task_names": task_names,
        "goals": goals,
    }

    if use_actions and action_ids is not None:
        output["action_ids"] = action_ids

    return output
