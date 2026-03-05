import copy
import difflib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


ActionEntry = Union[str, Sequence[Any]]
NodesInput = Union[Dict[str, Any], List[Dict[str, Any]]]
SlotMap = Dict[int, int]


DEFAULT_ACTION_OBJECT_ALIASES: Dict[str, str] = {
    "kitchencabinet": "kitchencabinets",
}
DEFAULT_ACTION_VERB_ALIASES: Dict[str, str] = {
    "[walktowards]": "[walk]",
}


class ActionCanonicalizationError(ValueError):
    """Structured action parsing error shared by Watch train/inference."""

    def __init__(
        self,
        message: str,
        *,
        raw_action: Optional[str],
        normalized_action: Optional[str],
        canonical_key: Optional[str],
        source_context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.raw_action = raw_action
        self.normalized_action = normalized_action
        self.canonical_key = canonical_key
        self.source_context = dict(source_context or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "raw_action": self.raw_action,
            "normalized_action": self.normalized_action,
            "canonical_key": self.canonical_key,
            "source_context": dict(self.source_context),
        }


class UnknownActionKeyError(ActionCanonicalizationError):
    """Raised when canonical action key is absent from metadata vocab in strict mode."""

    def __init__(
        self,
        *,
        raw_action: Optional[str],
        normalized_action: Optional[str],
        canonical_key: Optional[str],
        known_vocab_size: int,
        nearest_candidates: Sequence[str],
        source_context: Optional[Dict[str, Any]] = None,
    ):
        message = (
            "Unknown action key '{}' (vocab size {}).".format(
                canonical_key,
                int(known_vocab_size),
            )
        )
        super().__init__(
            message,
            raw_action=raw_action,
            normalized_action=normalized_action,
            canonical_key=canonical_key,
            source_context=source_context,
        )
        self.known_vocab_size = int(known_vocab_size)
        self.nearest_candidates = [str(item) for item in nearest_candidates]

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "known_vocab_size": int(self.known_vocab_size),
                "nearest_candidates": list(self.nearest_candidates),
            }
        )
        return payload


class WatchGraphTensorizer:
    """Converts watch/help graph observations into fixed-size node tensors.

    The representation intentionally mirrors the existing watch graph encoder input:
    - class_objects: [max_nodes]
    - object_coords: [max_nodes, 6]
    - states_objects: [max_nodes, num_states]
    - mask_object: [max_nodes]
    """

    def __init__(
        self,
        metadata_path: Optional[str] = None,
        max_nodes: Optional[int] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        state_to_idx: Optional[Dict[str, int]] = None,
        action_to_idx: Optional[Dict[str, int]] = None,
        action_object_aliases: Optional[Dict[str, str]] = None,
        action_verb_aliases: Optional[Dict[str, str]] = None,
    ):
        if metadata_path is not None:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            class_to_idx = metadata["graph_class_names"]
            state_to_idx = metadata["graph_node_states"]
            action_to_idx = metadata.get("action_predicates", {})
            if max_nodes is None:
                max_nodes = metadata["max_node_length"]

        if class_to_idx is None or state_to_idx is None:
            raise ValueError("class_to_idx and state_to_idx must be provided (directly or through metadata_path).")

        if max_nodes is None:
            raise ValueError("max_nodes must be provided (directly or through metadata_path).")

        self.class_to_idx = copy.deepcopy(class_to_idx)
        self.state_to_idx = copy.deepcopy(state_to_idx)
        self.action_to_idx = copy.deepcopy(action_to_idx or {})
        self.action_object_aliases = copy.deepcopy(action_object_aliases or DEFAULT_ACTION_OBJECT_ALIASES)
        self.action_verb_aliases = copy.deepcopy(action_verb_aliases or DEFAULT_ACTION_VERB_ALIASES)
        self.max_nodes = int(max_nodes)

        self.unknown_class_idx = self.class_to_idx.get("None", 0)

        self.num_classes = (max(self.class_to_idx.values()) + 1) if len(self.class_to_idx) > 0 else 1
        self.num_states = (max(self.state_to_idx.values()) + 1) if len(self.state_to_idx) > 0 else 1
        self.num_actions = (max(self.action_to_idx.values()) + 1) if len(self.action_to_idx) > 0 else 1

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_to_idx": copy.deepcopy(self.class_to_idx),
            "state_to_idx": copy.deepcopy(self.state_to_idx),
            "action_to_idx": copy.deepcopy(self.action_to_idx),
            "action_object_aliases": copy.deepcopy(self.action_object_aliases),
            "action_verb_aliases": copy.deepcopy(self.action_verb_aliases),
            "max_nodes": self.max_nodes,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WatchGraphTensorizer":
        return cls(
            metadata_path=None,
            max_nodes=config["max_nodes"],
            class_to_idx=config["class_to_idx"],
            state_to_idx=config["state_to_idx"],
            action_to_idx=config.get("action_to_idx", {}),
            action_object_aliases=config.get("action_object_aliases"),
            action_verb_aliases=config.get("action_verb_aliases"),
        )

    def _extract_nodes(self, graph_or_nodes: NodesInput) -> List[Dict[str, Any]]:
        if isinstance(graph_or_nodes, dict):
            return list(graph_or_nodes.get("nodes", []))
        return list(graph_or_nodes)

    @staticmethod
    def _is_character(node: Dict[str, Any]) -> bool:
        return "character" in str(node.get("class_name", "")).lower()

    @staticmethod
    def _node_id(node: Dict[str, Any]) -> Optional[int]:
        if "id" not in node or node["id"] is None:
            return None
        try:
            return int(node["id"])
        except (TypeError, ValueError):
            return None

    def _sorted_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Keep the original watch convention where character is placed first.
        character_nodes = [node for node in nodes if self._is_character(node)]
        other_nodes = [node for node in nodes if not self._is_character(node)]

        # Sort non-character nodes by id for deterministic online/offline behavior.
        other_nodes = sorted(other_nodes, key=lambda node: (int(node.get("id", -1)), str(node.get("class_name", ""))))

        # Keep only one character as canonical first slot if multiple are present.
        if len(character_nodes) > 0:
            character_nodes = sorted(character_nodes, key=lambda node: int(node.get("id", -1)))
            return [character_nodes[0]] + other_nodes + character_nodes[1:]
        return other_nodes

    def _char_center(self, nodes: List[Dict[str, Any]]) -> np.ndarray:
        if len(nodes) == 0:
            return np.zeros((3,), dtype=np.float32)

        # Use first node if it is a character; otherwise fallback to first node with bbox.
        candidates = []
        if self._is_character(nodes[0]):
            candidates.append(nodes[0])
        candidates.extend(nodes)

        for node in candidates:
            bbox = node.get("bounding_box")
            if bbox is None:
                continue
            center = bbox.get("center")
            if center is None or len(center) != 3:
                continue
            return np.array(center, dtype=np.float32)

        return np.zeros((3,), dtype=np.float32)

    def _next_free_slot(self, slot_map: SlotMap, prefer_zero: bool) -> Optional[int]:
        used_slots = set(slot_map.values())
        if prefer_zero and 0 not in used_slots:
            return 0

        for slot in range(self.max_nodes):
            if slot not in used_slots:
                return slot
        return None

    def extend_slot_map(self, graph_or_nodes: NodesInput, slot_map: SlotMap) -> None:
        """Assign stable slots to newly-seen object IDs.

        Slot assignment follows first-seen order under `_sorted_nodes`, which keeps
        online and offline behavior aligned.
        """
        nodes = self._extract_nodes(graph_or_nodes)
        nodes = self._sorted_nodes(nodes)

        for node in nodes:
            node_id = self._node_id(node)
            if node_id is None or node_id in slot_map:
                continue
            slot = self._next_free_slot(slot_map, prefer_zero=self._is_character(node))
            if slot is None:
                break
            slot_map[node_id] = slot

    def build_stable_slot_map(self, graph_sequence: Sequence[NodesInput]) -> SlotMap:
        slot_map: SlotMap = {}
        for graph_or_nodes in graph_sequence:
            self.extend_slot_map(graph_or_nodes, slot_map)
            if len(slot_map) >= self.max_nodes:
                break
        return slot_map

    def _empty_encoded(self) -> Dict[str, np.ndarray]:
        return {
            "class_objects": np.full((self.max_nodes,), self.unknown_class_idx, dtype=np.int64),
            "object_coords": np.zeros((self.max_nodes, 6), dtype=np.float32),
            "states_objects": np.zeros((self.max_nodes, self.num_states), dtype=np.float32),
            "mask_object": np.zeros((self.max_nodes,), dtype=np.float32),
        }

    def _write_node_features(
        self,
        slot: int,
        node: Dict[str, Any],
        char_center: np.ndarray,
        class_objects: np.ndarray,
        object_coords: np.ndarray,
        states_objects: np.ndarray,
        mask_object: np.ndarray,
    ) -> None:
        class_name = str(node.get("class_name", "None"))
        class_objects[slot] = self.class_to_idx.get(class_name, self.unknown_class_idx)

        for state in node.get("states", []):
            if state in self.state_to_idx:
                states_objects[slot, self.state_to_idx[state]] = 1.0

        bbox = node.get("bounding_box")
        if bbox is not None:
            center = bbox.get("center")
            size = bbox.get("size")
            if center is not None and len(center) == 3:
                object_coords[slot, :3] = np.array(center, dtype=np.float32) - char_center
            if size is not None and len(size) == 3:
                object_coords[slot, 3:] = np.array(size, dtype=np.float32)

        mask_object[slot] = 1.0

    def encode_nodes(self, graph_or_nodes: NodesInput) -> Dict[str, np.ndarray]:
        nodes = self._extract_nodes(graph_or_nodes)
        nodes = self._sorted_nodes(nodes)
        nodes = nodes[: self.max_nodes]

        encoded = self._empty_encoded()
        class_objects = encoded["class_objects"]
        object_coords = encoded["object_coords"]
        states_objects = encoded["states_objects"]
        mask_object = encoded["mask_object"]

        char_center = self._char_center(nodes)

        for idx, node in enumerate(nodes):
            self._write_node_features(
                slot=idx,
                node=node,
                char_center=char_center,
                class_objects=class_objects,
                object_coords=object_coords,
                states_objects=states_objects,
                mask_object=mask_object,
            )

        return encoded

    def encode_nodes_with_slot_map(
        self,
        graph_or_nodes: NodesInput,
        slot_map: SlotMap,
        allow_new_ids: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Encode a frame using a stable object-id to slot mapping.

        If `allow_new_ids=True`, unseen IDs are assigned to the next free slot.
        """
        nodes = self._extract_nodes(graph_or_nodes)
        nodes = self._sorted_nodes(nodes)

        if allow_new_ids:
            self.extend_slot_map(nodes, slot_map)

        encoded = self._empty_encoded()
        class_objects = encoded["class_objects"]
        object_coords = encoded["object_coords"]
        states_objects = encoded["states_objects"]
        mask_object = encoded["mask_object"]

        char_center = self._char_center(nodes)
        used_slots = set()

        for node in nodes:
            node_id = self._node_id(node)
            if node_id is None:
                continue

            slot = slot_map.get(node_id)
            if slot is None or slot < 0 or slot >= self.max_nodes or slot in used_slots:
                continue

            self._write_node_features(
                slot=slot,
                node=node,
                char_center=char_center,
                class_objects=class_objects,
                object_coords=object_coords,
                states_objects=states_objects,
                mask_object=mask_object,
            )
            used_slots.add(slot)

        return encoded

    def _extract_action_script(self, action_entry: Optional[ActionEntry]) -> Optional[Any]:
        if action_entry is None:
            return None
        if isinstance(action_entry, (list, tuple)):
            if len(action_entry) == 0:
                return None
            return action_entry[0]
        return action_entry

    def _canonicalize_action_object_token(self, obj_token: str) -> str:
        token = str(obj_token)
        if token.startswith("<") and token.endswith(">") and len(token) >= 3:
            raw_name = token[1:-1]
            alias_name = self.action_object_aliases.get(raw_name, raw_name)
            return "<{}>".format(alias_name)
        return str(self.action_object_aliases.get(token, token))

    def canonicalize_action_entry(
        self,
        action_entry: Optional[ActionEntry],
        *,
        strict: bool = False,
        source_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[str]]:
        """Normalize action text to a canonical key: "[verb] <object>"."""
        source_ctx = dict(source_context or {})
        script = self._extract_action_script(action_entry)
        if script is None:
            return {"raw_action": None, "normalized_action": None, "canonical_key": None}
        if not isinstance(script, str):
            raw_action = str(script)
            if strict:
                raise ActionCanonicalizationError(
                    "Action entry must contain a string script.",
                    raw_action=raw_action,
                    normalized_action=None,
                    canonical_key=None,
                    source_context=source_ctx,
                )
            return {"raw_action": raw_action, "normalized_action": None, "canonical_key": None}

        raw_action = script
        normalized_tokens = raw_action.strip().split()
        normalized_action = " ".join(normalized_tokens) if len(normalized_tokens) > 0 else ""
        if len(normalized_tokens) < 2:
            if strict:
                raise ActionCanonicalizationError(
                    "Action script must include verb and object tokens.",
                    raw_action=raw_action,
                    normalized_action=normalized_action,
                    canonical_key=None,
                    source_context=source_ctx,
                )
            return {"raw_action": raw_action, "normalized_action": normalized_action, "canonical_key": None}

        verb = str(self.action_verb_aliases.get(normalized_tokens[0], normalized_tokens[0]))
        obj = self._canonicalize_action_object_token(normalized_tokens[1])
        normalized_tokens[0] = verb
        normalized_tokens[1] = obj
        normalized_action = " ".join(normalized_tokens)
        canonical_key = "{} {}".format(verb, obj)
        return {
            "raw_action": raw_action,
            "normalized_action": normalized_action,
            "canonical_key": canonical_key,
        }

    def action_index(
        self,
        action_entry: Optional[ActionEntry],
        *,
        strict: bool = False,
        source_context: Optional[Dict[str, Any]] = None,
        return_details: bool = False,
    ) -> Union[int, Tuple[int, Dict[str, Any]]]:
        meta = self.canonicalize_action_entry(
            action_entry,
            strict=bool(strict),
            source_context=source_context,
        )
        canonical_key = meta.get("canonical_key")
        action_idx = 0
        if canonical_key is not None:
            lookup = self.action_to_idx.get(str(canonical_key))
            if lookup is None:
                if strict:
                    all_keys = sorted([str(key) for key in self.action_to_idx.keys()])
                    near = difflib.get_close_matches(str(canonical_key), all_keys, n=5, cutoff=0.0)
                    raise UnknownActionKeyError(
                        raw_action=meta.get("raw_action"),
                        normalized_action=meta.get("normalized_action"),
                        canonical_key=str(canonical_key),
                        known_vocab_size=len(all_keys),
                        nearest_candidates=near,
                        source_context=source_context,
                    )
                action_idx = 0
            else:
                action_idx = int(lookup)

        details = {
            "raw_action": meta.get("raw_action"),
            "normalized_action": meta.get("normalized_action"),
            "canonical_key": meta.get("canonical_key"),
            "action_index": int(action_idx),
        }
        if return_details:
            return int(action_idx), details
        return int(action_idx)
