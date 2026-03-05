from typing import Any, Dict, Hashable, Optional, Tuple, Union

import torch

from .model import GraphSequenceVAE
from .tensorizer import WatchGraphTensorizer


class OnlineVAEInference:
    """Stateful per-timestep VAE inference for online multi-agent use.

    Each agent key gets its own recurrent state.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        use_actions: Optional[bool] = None,
        strict_action_indexing: bool = False,
    ):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.tensorizer = WatchGraphTensorizer.from_config(checkpoint["tensorizer_config"])
        self.model = GraphSequenceVAE.from_config(checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        raw_goal_names = checkpoint.get("goal_predicate_names")
        if isinstance(raw_goal_names, list):
            self.goal_predicate_names = [str(name) for name in raw_goal_names]
        else:
            self.goal_predicate_names = None
        if use_actions is None:
            self.use_actions = bool(self.model.use_actions)
        else:
            # Optional override for runtime action usage; never enable actions if model was trained without them.
            self.use_actions = bool(use_actions) and bool(self.model.use_actions)
        self.strict_action_indexing = bool(strict_action_indexing)

        self.hidden_by_agent: Dict[Hashable, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.slot_map_by_agent: Dict[Hashable, Dict[int, int]] = {}

    def _encode_action_ids(
        self,
        action: Optional[Any],
        *,
        strict_actions: Optional[bool] = None,
        source_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if not self.use_actions:
            return None, {
                "raw_action": None,
                "normalized_action": None,
                "canonical_key": None,
                "action_index": None,
                "strict_action_indexing": bool(self.strict_action_indexing if strict_actions is None else strict_actions),
            }
        strict_flag = self.strict_action_indexing if strict_actions is None else bool(strict_actions)
        action_id, action_meta = self.tensorizer.action_index(
            action,
            strict=bool(strict_flag),
            source_context=source_context,
            return_details=True,
        )
        action_meta["strict_action_indexing"] = bool(strict_flag)
        action_ids = torch.tensor([[int(action_id)]], dtype=torch.long, device=self.device)
        return action_ids, action_meta

    def reset(self, agent_key: Optional[Hashable] = None) -> None:
        if agent_key is None:
            self.hidden_by_agent = {}
            self.slot_map_by_agent = {}
        else:
            self.hidden_by_agent.pop(agent_key, None)
            self.slot_map_by_agent.pop(agent_key, None)

    def _prepare_step_inputs(self, agent_key: Hashable, nodes_or_graph: Any) -> Dict[str, torch.Tensor]:
        slot_map = self.slot_map_by_agent.setdefault(agent_key, {})
        frame = self.tensorizer.encode_nodes_with_slot_map(
            nodes_or_graph,
            slot_map=slot_map,
            allow_new_ids=True,
        )
        return {
            "class_ids": torch.tensor(frame["class_objects"], dtype=torch.long, device=self.device).view(1, 1, -1),
            "coords": torch.tensor(frame["object_coords"], dtype=torch.float32, device=self.device).view(1, 1, self.tensorizer.max_nodes, 6),
            "states": torch.tensor(frame["states_objects"], dtype=torch.float32, device=self.device).view(1, 1, self.tensorizer.max_nodes, -1),
            "node_mask": torch.tensor(frame["mask_object"], dtype=torch.float32, device=self.device).view(1, 1, -1),
        }

    def step(
        self,
        agent_key: Hashable,
        nodes_or_graph: Any,
        action: Optional[Any] = None,
        sample: bool = True,
        strict_actions: Optional[bool] = None,
        source_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hidden = self.hidden_by_agent.get(agent_key)
        inputs = self._prepare_step_inputs(agent_key, nodes_or_graph)

        action_ids, action_meta = self._encode_action_ids(
            action,
            strict_actions=strict_actions,
            source_context=source_context,
        )

        with torch.no_grad():
            z, mu, logvar, next_hidden = self.model.encode_step(
                class_ids=inputs["class_ids"],
                coords=inputs["coords"],
                states=inputs["states"],
                node_mask=inputs["node_mask"],
                hidden=hidden,
                action_ids=action_ids,
                sample=sample,
            )

        # Detach recurrent state from graph for long online runs.
        self.hidden_by_agent[agent_key] = (next_hidden[0].detach(), next_hidden[1].detach())

        return {
            "z": z[:, 0, :].detach().cpu(),
            "mu": mu[:, 0, :].detach().cpu(),
            "logvar": logvar[:, 0, :].detach().cpu(),
            "action_raw": action_meta.get("raw_action"),
            "action_normalized": action_meta.get("normalized_action"),
            "action_canonical": action_meta.get("canonical_key"),
            "action_index": action_meta.get("action_index"),
        }

    def step_with_beliefs(
        self,
        agent_key: Hashable,
        nodes_or_graph: Any,
        action: Optional[Any] = None,
        sample: bool = False,
        strict_actions: Optional[bool] = None,
        source_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hidden = self.hidden_by_agent.get(agent_key)
        inputs = self._prepare_step_inputs(agent_key, nodes_or_graph)

        action_ids, action_meta = self._encode_action_ids(
            action,
            strict_actions=strict_actions,
            source_context=source_context,
        )

        with torch.no_grad():
            _, mu, logvar, next_hidden = self.model.encode_step(
                class_ids=inputs["class_ids"],
                coords=inputs["coords"],
                states=inputs["states"],
                node_mask=inputs["node_mask"],
                hidden=hidden,
                action_ids=action_ids,
                sample=sample,
            )
            mu_last = mu[:, 0, :]
            belief_logits = self.model.predicate_logits_from_mu(mu_last)
            belief_probs = torch.sigmoid(belief_logits)

        self.hidden_by_agent[agent_key] = (next_hidden[0].detach(), next_hidden[1].detach())

        return {
            "mu": mu_last.detach().cpu(),
            "logvar": logvar[:, 0, :].detach().cpu(),
            "belief_logits": belief_logits.detach().cpu(),
            "belief_probs": belief_probs.detach().cpu(),
            "action_raw": action_meta.get("raw_action"),
            "action_normalized": action_meta.get("normalized_action"),
            "action_canonical": action_meta.get("canonical_key"),
            "action_index": action_meta.get("action_index"),
        }
