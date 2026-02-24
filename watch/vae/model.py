from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeFeatureEncoder(nn.Module):
    def __init__(self, num_classes: int, num_states: int, hidden_size: int):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even")

        half = hidden_size // 2
        self.class_embedding = nn.Sequential(
            nn.Embedding(num_classes, half),
            nn.ReLU(),
            nn.Linear(half, half),
        )
        self.state_embedding = nn.Sequential(
            nn.Linear(num_states, half),
            nn.ReLU(),
            nn.Linear(half, half),
        )
        self.coord_embedding = nn.Sequential(
            nn.Linear(6, half),
            nn.ReLU(),
            nn.Linear(half, half),
        )

        self.combine = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size + half, hidden_size),
        )

    def forward(self, class_ids: torch.Tensor, coords: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # class_ids: [B, N], coords: [B, N, 6], states: [B, N, S]
        class_feat = self.class_embedding(class_ids)
        coord_feat = self.coord_embedding(coords)
        state_feat = self.state_embedding(states)
        node_feat = torch.cat([class_feat, coord_feat, state_feat], dim=-1)
        return self.combine(node_feat)


class FrameEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_states: int,
        hidden_size: int,
        transformer_nhead: int,
    ):
        super().__init__()
        self.node_encoder = NodeFeatureEncoder(num_classes=num_classes, num_states=num_states, hidden_size=hidden_size)

        encoder_layer = nn.modules.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_nhead,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
        )
        self.transformer = nn.modules.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(
        self,
        class_ids: torch.Tensor,
        coords: torch.Tensor,
        states: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of frame sequences into per-timestep embeddings.

        Inputs:
            class_ids: [B, T, N]
            coords: [B, T, N, 6]
            states: [B, T, N, S]
            node_mask: [B, T, N]

        Returns:
            frame_embeddings: [B, T, H]
        """
        batch_size, time_steps, max_nodes = class_ids.shape

        flat_class = class_ids.reshape(batch_size * time_steps, max_nodes)
        flat_coords = coords.reshape(batch_size * time_steps, max_nodes, 6)
        flat_states = states.reshape(batch_size * time_steps, max_nodes, states.shape[-1])
        flat_mask = node_mask.reshape(batch_size * time_steps, max_nodes)

        node_feat = self.node_encoder(flat_class, flat_coords, flat_states)

        # Avoid all-masked rows in Transformer (can produce NaNs on padded timesteps).
        safe_mask = flat_mask.clone()
        empty_rows = safe_mask.sum(dim=1) <= 0.0
        if empty_rows.any():
            safe_mask[empty_rows, 0] = 1.0

        key_padding_mask = (safe_mask <= 0.0)
        trans_out = self.transformer(node_feat.transpose(0, 1), src_key_padding_mask=key_padding_mask)
        trans_out = trans_out.transpose(0, 1)

        denom = flat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (trans_out * flat_mask.unsqueeze(-1)).sum(dim=1) / denom

        return pooled.reshape(batch_size, time_steps, -1)


class GraphSequenceVAE(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_states: int,
        max_nodes: int,
        hidden_size: int = 128,
        latent_size: int = 64,
        transformer_nhead: int = 2,
        use_actions: bool = False,
        num_actions: int = 1,
        kl_weight: float = 1.0,
        class_weight: float = 1.0,
        state_weight: float = 1.0,
        coord_weight: float = 1.0,
        mask_weight: float = 0.2,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_states = num_states
        self.max_nodes = max_nodes
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.transformer_nhead = transformer_nhead
        self.use_actions = use_actions
        self.num_actions = num_actions

        self.kl_weight = kl_weight
        self.class_weight = class_weight
        self.state_weight = state_weight
        self.coord_weight = coord_weight
        self.mask_weight = mask_weight
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.frame_encoder = FrameEncoder(
            num_classes=num_classes,
            num_states=num_states,
            hidden_size=hidden_size,
            transformer_nhead=transformer_nhead,
        )

        if self.use_actions:
            self.action_embedding = nn.Embedding(num_actions, hidden_size)
            self.action_fuse = nn.Linear(hidden_size * 2, hidden_size)

        self.temporal_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.posterior_mu = nn.Linear(hidden_size, latent_size)
        self.posterior_logvar = nn.Linear(hidden_size, latent_size)

        slot_dim = max(16, latent_size // 2)
        self.node_slot_embedding = nn.Embedding(max_nodes, slot_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + slot_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(hidden_size, num_classes)
        self.state_head = nn.Linear(hidden_size, num_states)
        self.coord_head = nn.Linear(hidden_size, 6)
        self.mask_head = nn.Linear(hidden_size, 1)

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "num_states": self.num_states,
            "max_nodes": self.max_nodes,
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "transformer_nhead": self.transformer_nhead,
            "use_actions": self.use_actions,
            "num_actions": self.num_actions,
            "kl_weight": self.kl_weight,
            "class_weight": self.class_weight,
            "state_weight": self.state_weight,
            "coord_weight": self.coord_weight,
            "mask_weight": self.mask_weight,
            "logvar_min": self.logvar_min,
            "logvar_max": self.logvar_max,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphSequenceVAE":
        return cls(**config)

    def get_teacher_prefixes(self, scope: str = "backbone") -> list:
        if scope == "backbone":
            prefixes = [
                "frame_encoder.",
                "temporal_encoder.",
            ]
            if self.use_actions:
                prefixes.extend(["action_embedding.", "action_fuse."])
            return prefixes
        if scope == "full_encoder":
            prefixes = [
                "frame_encoder.",
                "temporal_encoder.",
                "posterior_mu.",
                "posterior_logvar.",
            ]
            if self.use_actions:
                prefixes.extend(["action_embedding.", "action_fuse."])
            return prefixes
        raise ValueError("Unknown teacher scope: {}. Expected one of: backbone, full_encoder".format(scope))

    def get_teacher_state_dict(self, scope: str = "backbone") -> Dict[str, torch.Tensor]:
        prefixes = self.get_teacher_prefixes(scope=scope)
        state = self.state_dict()
        return {k: v for k, v in state.items() if any(k.startswith(prefix) for prefix in prefixes)}

    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Backward-compatible alias: previous behavior exported full encoder."""
        return self.get_teacher_state_dict(scope="full_encoder")

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _fuse_actions(self, frame_emb: torch.Tensor, action_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_actions:
            return frame_emb
        if action_ids is None:
            raise ValueError("Model was configured with use_actions=True but action_ids is missing")
        action_emb = self.action_embedding(action_ids)
        return self.action_fuse(torch.cat([frame_emb, action_emb], dim=-1))

    def encode_sequence(
        self,
        class_ids: torch.Tensor,
        coords: torch.Tensor,
        states: torch.Tensor,
        node_mask: torch.Tensor,
        lengths: torch.Tensor,
        action_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_emb = self.frame_encoder(class_ids, coords, states, node_mask)
        frame_emb = self._fuse_actions(frame_emb, action_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            frame_emb,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.temporal_encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=frame_emb.shape[1],
        )

        mu = self.posterior_mu(encoded)
        logvar = self.posterior_logvar(encoded).clamp(min=self.logvar_min, max=self.logvar_max)
        return encoded, mu, logvar

    def encode_step(
        self,
        class_ids: torch.Tensor,
        coords: torch.Tensor,
        states: torch.Tensor,
        node_mask: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_ids: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Single-step online encoding API. Inputs are [B, 1, ...]
        frame_emb = self.frame_encoder(class_ids, coords, states, node_mask)
        frame_emb = self._fuse_actions(frame_emb, action_ids)
        out, hidden_next = self.temporal_encoder(frame_emb, hidden)
        mu = self.posterior_mu(out)
        logvar = self.posterior_logvar(out).clamp(min=self.logvar_min, max=self.logvar_max)
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar, hidden_next

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = z.shape

        slot_ids = torch.arange(self.max_nodes, device=z.device)
        slot_emb = self.node_slot_embedding(slot_ids)
        slot_emb = slot_emb.view(1, 1, self.max_nodes, -1).expand(batch_size, time_steps, self.max_nodes, -1)

        z_expanded = z.unsqueeze(2).expand(batch_size, time_steps, self.max_nodes, self.latent_size)
        decoder_input = torch.cat([z_expanded, slot_emb], dim=-1)
        dec_feat = self.decoder(decoder_input)

        class_logits = self.class_head(dec_feat)
        state_logits = self.state_head(dec_feat)
        coord_pred = self.coord_head(dec_feat)
        mask_logits = self.mask_head(dec_feat).squeeze(-1)

        return {
            "class_logits": class_logits,
            "state_logits": state_logits,
            "coord_pred": coord_pred,
            "mask_logits": mask_logits,
        }

    def compute_losses(
        self,
        recon: Dict[str, torch.Tensor],
        class_ids: torch.Tensor,
        states: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
        time_mask: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        valid_nodes = node_mask * time_mask.unsqueeze(-1)
        valid_node_count = valid_nodes.sum().clamp(min=1.0)

        valid_index = valid_nodes > 0
        if valid_index.any():
            class_loss = F.cross_entropy(recon["class_logits"][valid_index], class_ids[valid_index])
        else:
            class_loss = recon["class_logits"].sum() * 0.0

        state_bce = F.binary_cross_entropy_with_logits(
            recon["state_logits"],
            states,
            reduction="none",
        ).mean(dim=-1)
        state_loss = (state_bce * valid_nodes).sum() / valid_node_count

        coord_l1 = F.smooth_l1_loss(
            recon["coord_pred"],
            coords,
            reduction="none",
        ).mean(dim=-1)
        coord_loss = (coord_l1 * valid_nodes).sum() / valid_node_count

        mask_bce = F.binary_cross_entropy_with_logits(
            recon["mask_logits"],
            node_mask,
            reduction="none",
        )
        valid_time = time_mask.unsqueeze(-1).expand_as(mask_bce)
        mask_loss = (mask_bce * valid_time).sum() / valid_time.sum().clamp(min=1.0)

        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        kl_loss = (kl * time_mask).sum() / time_mask.sum().clamp(min=1.0)

        total = (
            self.class_weight * class_loss
            + self.state_weight * state_loss
            + self.coord_weight * coord_loss
            + self.mask_weight * mask_loss
            + self.kl_weight * kl_loss
        )

        return {
            "loss": total,
            "class_loss": class_loss,
            "state_loss": state_loss,
            "coord_loss": coord_loss,
            "mask_loss": mask_loss,
            "kl_loss": kl_loss,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded, mu, logvar = self.encode_sequence(
            class_ids=batch["class_ids"],
            coords=batch["coords"],
            states=batch["states"],
            node_mask=batch["node_mask"],
            lengths=batch["lengths"],
            action_ids=batch.get("action_ids"),
        )
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        losses = self.compute_losses(
            recon=recon,
            class_ids=batch["class_ids"],
            states=batch["states"],
            coords=batch["coords"],
            node_mask=batch["node_mask"],
            time_mask=batch["time_mask"],
            mu=mu,
            logvar=logvar,
        )

        losses["z_mean_abs"] = mu.abs().mean()
        losses["z_std_mean"] = torch.exp(0.5 * logvar).mean()
        losses["encoded_norm"] = encoded.norm(dim=-1).mean()
        return losses
