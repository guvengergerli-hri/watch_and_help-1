from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GraphSequenceVAE


class GraphSequenceSemiSupVAE(GraphSequenceVAE):
    """Semi-supervised sequence VAE with continuous z and multi-label Bernoulli y.

    y is modeled as a factorized multivariate Bernoulli q(y|x). During training we can
    sample a Binary Concrete relaxation for differentiable conditioning.
    """

    def __init__(
        self,
        num_classes: int,
        num_states: int,
        max_nodes: int,
        num_goal_predicates: int,
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
        y_bce_weight: float = 1.0,
        y_kl_weight: float = 0.0,
        y_dyn_kl_weight: float = 0.0,
        y_temperature: float = 0.67,
        y_prior_prob: float = 0.1,
        y_dyn_prior_init_prob: float = 0.5,
        condition_z_on_y: bool = True,
        condition_decoder_on_y: bool = True,
        use_labeled_y_for_recon: bool = True,
    ):
        super().__init__(
            num_classes=num_classes,
            num_states=num_states,
            max_nodes=max_nodes,
            hidden_size=hidden_size,
            latent_size=latent_size,
            transformer_nhead=transformer_nhead,
            use_actions=use_actions,
            num_actions=num_actions,
            kl_weight=kl_weight,
            class_weight=class_weight,
            state_weight=state_weight,
            coord_weight=coord_weight,
            mask_weight=mask_weight,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        if num_goal_predicates <= 0:
            raise ValueError("num_goal_predicates must be positive")

        self.num_goal_predicates = int(num_goal_predicates)
        self.y_bce_weight = float(y_bce_weight)
        self.y_kl_weight = float(y_kl_weight)
        self.y_dyn_kl_weight = float(y_dyn_kl_weight)
        self.y_temperature = float(y_temperature)
        self.y_prior_prob = float(y_prior_prob)
        self.y_dyn_prior_init_prob = float(y_dyn_prior_init_prob)
        self.condition_z_on_y = bool(condition_z_on_y)
        self.condition_decoder_on_y = bool(condition_decoder_on_y)
        self.use_labeled_y_for_recon = bool(use_labeled_y_for_recon)

        self.goal_head = nn.Linear(self.hidden_size, self.num_goal_predicates)

        if self.condition_z_on_y:
            self.posterior_condition = nn.Sequential(
                nn.Linear(self.hidden_size + self.num_goal_predicates, self.hidden_size),
                nn.ReLU(),
            )
        else:
            self.posterior_condition = None

        # Replace decoder if y conditions the generative path.
        slot_dim = self.node_slot_embedding.embedding_dim
        decoder_in_dim = self.latent_size + slot_dim + (self.num_goal_predicates if self.condition_decoder_on_y else 0)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(self.hidden_size, self.num_classes)
        self.state_head = nn.Linear(self.hidden_size, self.num_states)
        self.coord_head = nn.Linear(self.hidden_size, 6)
        self.mask_head = nn.Linear(self.hidden_size, 1)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_goal_predicates": self.num_goal_predicates,
                "y_bce_weight": self.y_bce_weight,
                "y_kl_weight": self.y_kl_weight,
                "y_dyn_kl_weight": self.y_dyn_kl_weight,
                "y_temperature": self.y_temperature,
                "y_prior_prob": self.y_prior_prob,
                "y_dyn_prior_init_prob": self.y_dyn_prior_init_prob,
                "condition_z_on_y": self.condition_z_on_y,
                "condition_decoder_on_y": self.condition_decoder_on_y,
                "use_labeled_y_for_recon": self.use_labeled_y_for_recon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphSequenceSemiSupVAE":
        return cls(**config)

    @staticmethod
    def _binary_concrete_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        u = torch.rand_like(logits).clamp(min=1e-6, max=1.0 - 1e-6)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        return torch.sigmoid((logits + logistic_noise) / max(1e-6, float(temperature)))

    @staticmethod
    def _pool_last(encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(encoded.shape[0], device=encoded.device)
        return encoded[batch_idx, idx]

    @staticmethod
    def _analytic_bernoulli_kl_probs(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q = q.clamp(min=1e-6, max=1.0 - 1e-6)
        p = p.clamp(min=1e-6, max=1.0 - 1e-6)
        kl = q * (torch.log(q) - torch.log(p)) + (1.0 - q) * (torch.log1p(-q) - torch.log1p(-p))
        return kl.sum(dim=-1)

    def _analytic_bernoulli_kl(self, logits: torch.Tensor) -> torch.Tensor:
        q = torch.sigmoid(logits)
        p = torch.full_like(q, fill_value=min(max(self.y_prior_prob, 1e-6), 1.0 - 1e-6))
        return self._analytic_bernoulli_kl_probs(q, p)

    def _temporal_belief_prior_kl(self, y_probs_seq: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        """KL regularizer for q(y_t|x_{1:t}) against a belief-persistence prior.

        Prior is factorized Bernoulli:
          p(y_0) = 0.5 (or configured init prob)
          p(y_t) = q(y_{t-1})  for t >= 1
        with stop-gradient through q(y_{t-1}) so the prior regularizes the current belief.
        """
        prior = torch.full_like(
            y_probs_seq,
            fill_value=min(max(self.y_dyn_prior_init_prob, 1e-6), 1.0 - 1e-6),
        )
        if y_probs_seq.shape[1] > 1:
            prior[:, 1:] = y_probs_seq[:, :-1].detach()

        kl_per_t = self._analytic_bernoulli_kl_probs(y_probs_seq, prior)  # [B, T]
        mask = time_mask.float()
        denom = mask.sum().clamp(min=1.0)
        return (kl_per_t * mask).sum() / denom

    def _build_y_condition(
        self,
        y_logits: torch.Tensor,
        goal_labels: Optional[torch.Tensor],
        goal_label_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        y_probs = torch.sigmoid(y_logits)
        if self.training:
            y_relaxed = self._binary_concrete_sample(y_logits, temperature=self.y_temperature)
        else:
            y_relaxed = y_probs

        y_for_cond = y_relaxed
        if self.use_labeled_y_for_recon and goal_labels is not None:
            if goal_label_mask is None:
                y_for_cond = goal_labels
            else:
                mask = goal_label_mask.view(-1, 1)
                y_for_cond = torch.where(mask > 0.5, goal_labels, y_relaxed)

        return {
            "y_probs": y_probs,
            "y_relaxed": y_relaxed,
            "y_for_cond": y_for_cond,
        }

    def decode(self, z: torch.Tensor, y_demo: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = z.shape

        slot_ids = torch.arange(self.max_nodes, device=z.device)
        slot_emb = self.node_slot_embedding(slot_ids)
        slot_emb = slot_emb.view(1, 1, self.max_nodes, -1).expand(batch_size, time_steps, self.max_nodes, -1)

        z_expanded = z.unsqueeze(2).expand(batch_size, time_steps, self.max_nodes, self.latent_size)
        decoder_input_parts = [z_expanded, slot_emb]

        if self.condition_decoder_on_y:
            if y_demo is None:
                raise ValueError("Decoder conditioning on y is enabled, but y_demo is missing")
            y_seq = y_demo.unsqueeze(1).expand(batch_size, time_steps, self.num_goal_predicates)
            y_expanded = y_seq.unsqueeze(2).expand(batch_size, time_steps, self.max_nodes, self.num_goal_predicates)
            decoder_input_parts.append(y_expanded)

        decoder_input = torch.cat(decoder_input_parts, dim=-1)
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

    def infer_y_probs_seq(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the encoder backbone and return timestep-wise y belief probabilities."""
        frame_emb = self.frame_encoder(batch["class_ids"], batch["coords"], batch["states"], batch["node_mask"])
        frame_emb = self._fuse_actions(frame_emb, batch.get("action_ids"))

        packed = nn.utils.rnn.pack_padded_sequence(
            frame_emb,
            lengths=batch["lengths"].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.temporal_encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=frame_emb.shape[1],
        )
        y_logits_seq = self.goal_head(encoded)
        return {
            "encoded": encoded,
            "y_logits_seq": y_logits_seq,
            "y_probs_seq": torch.sigmoid(y_logits_seq),
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Encode x with the base encoder backbone first (without z posterior heads).
        y_seq_info = self.infer_y_probs_seq(batch)
        encoded = y_seq_info["encoded"]
        y_logits_seq = y_seq_info["y_logits_seq"]
        y_logits = self._pool_last(y_logits_seq, batch["lengths"])
        y_info = self._build_y_condition(
            y_logits=y_logits,
            goal_labels=batch.get("goal_labels"),
            goal_label_mask=batch.get("goal_label_mask"),
        )
        y_for_cond = y_info["y_for_cond"]

        if self.condition_z_on_y:
            y_seq = y_for_cond.unsqueeze(1).expand(-1, encoded.shape[1], -1)
            posterior_in = self.posterior_condition(torch.cat([encoded, y_seq], dim=-1))
        else:
            posterior_in = encoded

        mu = self.posterior_mu(posterior_in)
        logvar = self.posterior_logvar(posterior_in).clamp(min=self.logvar_min, max=self.logvar_max)
        z = self.reparameterize(mu, logvar)

        recon = self.decode(z, y_demo=y_for_cond if self.condition_decoder_on_y else None)
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

        # Multi-label supervision on y (only where labels are marked available).
        goal_labels = batch.get("goal_labels")
        goal_label_mask = batch.get("goal_label_mask")
        if goal_labels is not None:
            y_bce_vec = F.binary_cross_entropy_with_logits(y_logits, goal_labels, reduction="none").mean(dim=-1)
            if goal_label_mask is None:
                y_bce_loss = y_bce_vec.mean()
                labeled_fraction = torch.ones((), device=y_logits.device)
                labeled_mask_bool = torch.ones((y_logits.shape[0],), device=y_logits.device, dtype=torch.bool)
                goal_label_mask_vec = torch.ones((y_logits.shape[0],), device=y_logits.device)
            else:
                denom = goal_label_mask.sum().clamp(min=1.0)
                y_bce_loss = (y_bce_vec * goal_label_mask).sum() / denom
                labeled_fraction = goal_label_mask.mean()
                labeled_mask_bool = goal_label_mask > 0.5
                goal_label_mask_vec = goal_label_mask
        else:
            y_bce_loss = y_logits.sum() * 0.0
            labeled_fraction = torch.zeros((), device=y_logits.device)
            labeled_mask_bool = torch.zeros((y_logits.shape[0],), device=y_logits.device, dtype=torch.bool)
            goal_label_mask_vec = torch.zeros((y_logits.shape[0],), device=y_logits.device)

        y_kl_per = self._analytic_bernoulli_kl(y_logits)
        y_kl_loss = y_kl_per.mean()
        y_probs_seq = y_seq_info["y_probs_seq"]
        y_dyn_kl_loss = self._temporal_belief_prior_kl(y_probs_seq, batch["time_mask"])

        losses["loss"] = (
            losses["loss"]
            + self.y_bce_weight * y_bce_loss
            + self.y_kl_weight * y_kl_loss
            + self.y_dyn_kl_weight * y_dyn_kl_loss
        )
        losses["goal_bce_loss"] = y_bce_loss
        losses["goal_kl_loss"] = y_kl_loss
        losses["goal_dyn_kl_loss"] = y_dyn_kl_loss
        losses["goal_labeled_fraction"] = labeled_fraction

        y_probs = y_info["y_probs"]
        y_probs_last = self._pool_last(y_probs_seq, batch["lengths"])
        losses["goal_prob_mean"] = y_probs.mean()
        losses["goal_prob_std"] = y_probs.std(unbiased=False)
        losses["goal_relaxed_mean"] = y_info["y_relaxed"].mean()
        losses["goal_dyn_prior_init_prob"] = y_logits.new_tensor(self.y_dyn_prior_init_prob)
        eps = 1e-6
        y_entropy_last = -(
            y_probs_last.clamp(min=eps, max=1.0 - eps) * torch.log(y_probs_last.clamp(min=eps, max=1.0 - eps))
            + (1.0 - y_probs_last).clamp(min=eps, max=1.0 - eps) * torch.log((1.0 - y_probs_last).clamp(min=eps, max=1.0 - eps))
        )
        y_entropy_seq = -(
            y_probs_seq.clamp(min=eps, max=1.0 - eps) * torch.log(y_probs_seq.clamp(min=eps, max=1.0 - eps))
            + (1.0 - y_probs_seq).clamp(min=eps, max=1.0 - eps) * torch.log((1.0 - y_probs_seq).clamp(min=eps, max=1.0 - eps))
        )
        losses["goal_entropy_last_mean"] = y_entropy_last.mean()
        time_mask_exp = batch["time_mask"].float().unsqueeze(-1)
        seq_entropy_denom = time_mask_exp.sum().clamp(min=1.0) * y_probs_seq.shape[-1]
        losses["goal_entropy_seq_mean"] = (y_entropy_seq * time_mask_exp).sum() / seq_entropy_denom

        if y_probs_seq.shape[1] > 1:
            pair_mask = (batch["time_mask"][:, 1:] * batch["time_mask"][:, :-1]).float()
            pair_denom = pair_mask.sum().clamp(min=1.0)
            prob_delta = (y_probs_seq[:, 1:] - y_probs_seq[:, :-1].detach()).abs().mean(dim=-1)
            losses["goal_prob_delta_abs_mean"] = (prob_delta * pair_mask).sum() / pair_denom
        else:
            losses["goal_prob_delta_abs_mean"] = y_logits.sum() * 0.0

        if goal_labels is not None and labeled_mask_bool.any():
            pred = (y_probs[labeled_mask_bool] >= 0.5).float()
            tar = goal_labels[labeled_mask_bool]
            tp = (pred * tar).sum()
            fp = (pred * (1.0 - tar)).sum()
            fn = ((1.0 - pred) * tar).sum()
            precision = tp / (tp + fp).clamp(min=1.0)
            recall = tp / (tp + fn).clamp(min=1.0)
            f1 = (2.0 * precision * recall) / (precision + recall).clamp(min=1e-6)
            exact = (pred == tar).all(dim=-1).float().mean()
            losses["goal_precision_0p5"] = precision
            losses["goal_recall_0p5"] = recall
            losses["goal_f1_0p5"] = f1
            losses["goal_precision_0p5_last"] = precision
            losses["goal_recall_0p5_last"] = recall
            losses["goal_f1_0p5_last"] = f1
            losses["goal_exact_match_0p5"] = exact
            losses["goal_exact_match_0p5_last"] = exact
            losses["goal_target_pos_rate"] = tar.mean()

            tar_seq = goal_labels.unsqueeze(1).expand(-1, y_probs_seq.shape[1], -1)
            pred_seq = (y_probs_seq >= 0.5).float()
            step_mask = batch["time_mask"].float().unsqueeze(-1) * goal_label_mask_vec.view(-1, 1, 1).float()
            tp_seq = (pred_seq * tar_seq * step_mask).sum()
            fp_seq = (pred_seq * (1.0 - tar_seq) * step_mask).sum()
            fn_seq = ((1.0 - pred_seq) * tar_seq * step_mask).sum()
            precision_seq = tp_seq / (tp_seq + fp_seq).clamp(min=1.0)
            recall_seq = tp_seq / (tp_seq + fn_seq).clamp(min=1.0)
            f1_seq = (2.0 * precision_seq * recall_seq) / (precision_seq + recall_seq).clamp(min=1e-6)
            exact_seq = ((pred_seq == tar_seq).all(dim=-1).float() * step_mask.squeeze(-1)).sum() / step_mask.squeeze(-1).sum().clamp(min=1.0)
            losses["goal_precision_0p5_allsteps"] = precision_seq
            losses["goal_recall_0p5_allsteps"] = recall_seq
            losses["goal_f1_0p5_allsteps"] = f1_seq
            losses["goal_exact_match_0p5_allsteps"] = exact_seq
        else:
            zero = y_logits.sum() * 0.0
            losses["goal_precision_0p5"] = zero
            losses["goal_recall_0p5"] = zero
            losses["goal_f1_0p5"] = zero
            losses["goal_precision_0p5_last"] = zero
            losses["goal_recall_0p5_last"] = zero
            losses["goal_f1_0p5_last"] = zero
            losses["goal_exact_match_0p5"] = zero
            losses["goal_exact_match_0p5_last"] = zero
            losses["goal_target_pos_rate"] = zero
            losses["goal_precision_0p5_allsteps"] = zero
            losses["goal_recall_0p5_allsteps"] = zero
            losses["goal_f1_0p5_allsteps"] = zero
            losses["goal_exact_match_0p5_allsteps"] = zero

        losses["goal_pred_pos_rate_0p5"] = (y_probs >= 0.5).float().mean()
        losses["goal_pred_pos_rate_0p5_last"] = (y_probs >= 0.5).float().mean()
        losses["goal_pred_pos_rate_0p5_allsteps"] = (
            ((y_probs_seq >= 0.5).float() * batch["time_mask"].float().unsqueeze(-1)).sum()
            / (batch["time_mask"].float().sum().clamp(min=1.0) * y_probs_seq.shape[-1])
        )
        losses["z_mean_abs"] = mu.abs().mean()
        losses["z_std_mean"] = torch.exp(0.5 * logvar).mean()
        losses["encoded_norm"] = encoded.norm(dim=-1).mean()
        return losses
