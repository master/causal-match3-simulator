"""Continuous hierarchical latent model and predictive training objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..retention import MasteryConfig
from .action_policy import ActionPolicyConfig, ContinuousActionPolicy
from .encoder import CausalPrefixEncoder, PrefixEncoderConfig
from .heads import (
    AssignmentHead,
    ChurnHead,
    CorrelatedSkillTransform,
    EvidenceHead,
    WinHead,
)


@dataclass(frozen=True)
class PredictiveTarget:
    """One target episode per player plus all of its observed action rows."""

    episode_player: torch.Tensor
    served_difficulty: torch.Tensor
    levels: torch.Tensor
    tiers: torch.Tensor
    evidence: torch.Tensor
    outcomes: torch.Tensor
    mastery_before: torch.Tensor
    completion_margin: torch.Tensor
    churn: torch.Tensor
    churn_mask: torch.Tensor
    churn_scale: torch.Tensor
    action_player: torch.Tensor
    boards: torch.Tensor
    goal_colours: torch.Tensor
    moves_left: torch.Tensor
    goals_left: torch.Tensor
    legal_actions: torch.Tensor
    actions: torch.Tensor


class ContinuousCausalVAE(nn.Module):
    """Strict-prefix continuous latent model with structural decoder heads."""

    def __init__(
        self,
        encoder_config: PrefixEncoderConfig = PrefixEncoderConfig(),
        action_config: ActionPolicyConfig = ActionPolicyConfig(),
        mastery_config: MasteryConfig = MasteryConfig(),
    ):
        super().__init__()
        if encoder_config.skill_dimensions != action_config.skill_dimensions:
            raise ValueError("encoder and action policy skill dimensions must match")
        self.encoder = CausalPrefixEncoder(encoder_config)
        self.skill_transform = CorrelatedSkillTransform()
        self.action_policy = ContinuousActionPolicy(action_config)
        self.assignment_head = AssignmentHead(
            n_levels=encoder_config.n_levels,
            n_tiers=encoder_config.n_tiers,
            skill_dimensions=encoder_config.skill_dimensions,
        )
        self.evidence_head = EvidenceHead(
            skill_dimensions=encoder_config.skill_dimensions,
            evidence_dimensions=encoder_config.proxy_dimensions,
            n_levels=encoder_config.n_levels,
        )
        self.win_head = WinHead(
            n_levels=encoder_config.n_levels,
            n_tiers=encoder_config.n_tiers,
            skill_dimensions=encoder_config.skill_dimensions,
        )
        self.churn_head = ChurnHead(n_levels=encoder_config.n_levels)
        self.mastery_update_rate = mastery_config.update_rate

    def posterior(
        self, prefix: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(**prefix)

    def expected_churn_probability(
        self,
        win_probability: torch.Tensor,
        mastery_before: torch.Tensor,
        level: torch.Tensor,
        margin_after_win: torch.Tensor | None = None,
        margin_after_loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate churn over outcomes with optional conditional margins."""
        mastery = mastery_before.to(win_probability.dtype)
        after_win = mastery + self.mastery_update_rate * (1.0 - mastery)
        after_loss = mastery - self.mastery_update_rate * mastery
        churn_after_win = self.churn_head.probabilities(
            after_win, level, completion_margin=margin_after_win
        )
        churn_after_loss = self.churn_head.probabilities(
            after_loss, level, completion_margin=margin_after_loss
        )
        return (
            win_probability * churn_after_win
            + (1.0 - win_probability) * churn_after_loss
        )

    def predictive_objective(
        self,
        prefix: dict[str, torch.Tensor],
        target: PredictiveTarget,
        *,
        kl_weight: float = 1.0,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Score episode j using q(K_i | completed episodes before j)."""
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        mean, log_scale = self.posterior(prefix)
        whitened_skill = (
            self.encoder.rsample(mean, log_scale) if sample_posterior else mean
        )
        skill = self.skill_transform(whitened_skill)
        batch_size = skill.shape[0]
        if torch.any((target.episode_player < 0) | (target.episode_player >= batch_size)):
            raise ValueError("episode_player index outside posterior batch")
        if torch.any((target.action_player < 0) | (target.action_player >= batch_size)):
            raise ValueError("action_player index outside posterior batch")

        episode_skill = skill[target.episode_player.long()]
        assignment_nll = -self.assignment_head.log_prob(
            target.served_difficulty,
            episode_skill,
            target.levels.long(),
            target.tiers.long(),
        ).mean()
        evidence_nll = -self.evidence_head.log_prob(
            target.evidence, episode_skill, target.levels.long()
        ).mean()
        win_logits = self.win_head.logits(
            target.served_difficulty,
            episode_skill,
            target.levels.long(),
            target.tiers.long(),
        )
        win_nll = F.binary_cross_entropy_with_logits(
            win_logits, target.outcomes.to(win_logits.dtype)
        )
        mastery_before = target.mastery_before.to(win_logits.dtype)
        mastery_after = mastery_before + self.mastery_update_rate * (
            target.outcomes.to(win_logits.dtype) - mastery_before
        )
        churn_probability = target.churn_scale * torch.sigmoid(
            self.churn_head.logits(
                mastery_after,
                target.levels.long(),
                completion_margin=target.completion_margin,
            )
        )
        churn_losses = F.binary_cross_entropy(
            churn_probability,
            target.churn.to(churn_probability.dtype),
            reduction="none",
        )
        churn_mask = target.churn_mask.to(churn_losses.dtype)
        churn_nll = (churn_losses * churn_mask).sum() / churn_mask.sum().clamp_min(1.0)
        if target.actions.numel():
            action_skill = skill[target.action_player.long()]
            action_nll = self.action_policy.negative_log_likelihood(
                action=target.actions,
                board=target.boards,
                goal_colour=target.goal_colours,
                moves_left=target.moves_left,
                goals_left=target.goals_left,
                skill=action_skill,
                legal_actions=target.legal_actions,
            )
        else:
            action_nll = skill.sum() * 0.0
        kl = self.encoder.kl_standard_normal(mean, log_scale).mean()
        total = (
            assignment_nll
            + evidence_nll
            + win_nll
            + churn_nll
            + action_nll
            + kl_weight * kl
        )
        return {
            "loss": total,
            "assignment_nll": assignment_nll,
            "evidence_nll": evidence_nll,
            "win_nll": win_nll,
            "churn_nll": churn_nll,
            "action_nll": action_nll,
            "kl": kl,
            "posterior_mean": mean,
            "posterior_log_scale": log_scale,
            "skill_sample": skill,
            "mastery_after": mastery_after,
        }


__all__ = ["ContinuousCausalVAE", "PredictiveTarget"]