from typing import Any, Dict, Hashable, Optional, Tuple, Union

import torch

from .model import GraphSequenceVAE
from .tensorizer import WatchGraphTensorizer


class OnlineVAEInference:
    """Stateful per-timestep VAE inference for online multi-agent use.

    Each agent key gets its own recurrent state.
    """

    def __init__(self, checkpoint_path: str, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.tensorizer = WatchGraphTensorizer.from_config(checkpoint["tensorizer_config"])
        self.model = GraphSequenceVAE.from_config(checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.hidden_by_agent: Dict[Hashable, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.slot_map_by_agent: Dict[Hashable, Dict[int, int]] = {}

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
    ) -> Dict[str, torch.Tensor]:
        hidden = self.hidden_by_agent.get(agent_key)
        inputs = self._prepare_step_inputs(agent_key, nodes_or_graph)

        if self.model.use_actions:
            action_id = self.tensorizer.action_index(action)
            action_ids = torch.tensor([[action_id]], dtype=torch.long, device=self.device)
        else:
            action_ids = None

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
        }
