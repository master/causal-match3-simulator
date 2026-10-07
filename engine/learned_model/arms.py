"""Matched context pathways around the shared generative gameplay core."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F

from ..retention import MasteryConfig
from .action_policy import ActionPolicyConfig, ContinuousActionPolicy
from .encoder import CausalPrefixEncoder, PrefixEncoderConfig
from .generative import GameplayRSSM, GameplayRSSMConfig
from .heads import (
    AssignmentHead,
    ChurnHead,
    CorrelatedSkillTransform,
    EvidenceHead,
    WinHead,
)
from .model import PredictiveTarget


class ModelArm(str, Enum):
    POOLED = "pooled"
    MATCHED_NO_K = "matched_no_k"
    CAUSAL = "causal"
    ORACLE = "oracle"


@dataclass(frozen=True)
class GenerativeModelConfig:
    dynamics: GameplayRSSMConfig = field(default_factory=GameplayRSSMConfig)
    prefix: PrefixEncoderConfig = field(default_factory=PrefixEncoderConfig)
    behavior: ActionPolicyConfig = field(default_factory=ActionPolicyConfig)
    mastery: MasteryConfig = field(default_factory=MasteryConfig)

    def __post_init__(self) -> None:
        if self.dynamics.n_colours != self.prefix.n_colours:
            raise ValueError("dynamics and prefix colour vocabularies must match")
        if self.dynamics.n_colours != self.behavior.n_colours:
            raise ValueError("dynamics and behavior colour vocabularies must match")
        if self.dynamics.n_levels != self.prefix.n_levels:
            raise ValueError("dynamics and prefix level vocabularies must match")
        if self.dynamics.n_tiers != self.prefix.n_tiers:
            raise ValueError("dynamics and prefix tier vocabularies must match")
        if self.prefix.skill_dimensions != self.behavior.skill_dimensions:
            raise ValueError("prefix and behavior skill dimensions must match")
        if self.prefix.skill_dimensions != 4:
            raise ValueError("player context must match the four simulator skills")


@dataclass(frozen=True)
class PlayerContext:
    skill: torch.Tensor
    kl: torch.Tensor
    posterior_mean: torch.Tensor | None
    posterior_log_scale: torch.Tensor | None


class GenerativeWorldModel(nn.Module):
    """Compose one player-context arm with shared dynamics and behavior heads."""

    def __init__(
        self,
        arm: ModelArm,
        config: GenerativeModelConfig = GenerativeModelConfig(),
    ):
        super().__init__()
        self.arm = ModelArm(arm)
        self.config = config
        self.dynamics = GameplayRSSM(config.dynamics)
        self.behavior = ContinuousActionPolicy(config.behavior)
        self.prefix_encoder = (
            None
            if self.arm is ModelArm.POOLED
            else CausalPrefixEncoder(config.prefix)
        )
        self.skill_transform = CorrelatedSkillTransform()
        self.assignment_head = AssignmentHead(
            n_levels=config.prefix.n_levels,
            n_tiers=config.prefix.n_tiers,
            skill_dimensions=config.prefix.skill_dimensions,
        )
        self.evidence_head = EvidenceHead(
            skill_dimensions=config.prefix.skill_dimensions,
            evidence_dimensions=config.prefix.proxy_dimensions,
            n_levels=config.prefix.n_levels,
        )
        self.win_head = WinHead(
            n_levels=config.prefix.n_levels,
            n_tiers=config.prefix.n_tiers,
            skill_dimensions=config.prefix.skill_dimensions,
        )
        self.churn_head = ChurnHead(n_levels=config.prefix.n_levels)

    def player_context(
        self,
        *,
        prefix: dict[str, torch.Tensor] | None = None,
        oracle_skill: torch.Tensor | None = None,
        sample: bool = True,
    ) -> PlayerContext:
        """Resolve the arm's stable player context without target information."""
        if self.arm is not ModelArm.ORACLE and oracle_skill is not None:
            raise ValueError("oracle_skill is restricted to the oracle arm")
        if self.arm is ModelArm.ORACLE and oracle_skill is None:
            raise ValueError("oracle arm requires oracle_skill")
        if prefix is None and oracle_skill is None:
            raise ValueError("prefix is required to determine the player batch")

        if prefix is not None:
            if "episode_mask" not in prefix:
                raise ValueError("prefix must contain episode_mask")
            reference = prefix["episode_mask"]
            batch_size = reference.shape[0]
        else:
            assert oracle_skill is not None
            reference = oracle_skill
            batch_size = oracle_skill.shape[0]

        if self.arm is ModelArm.CAUSAL:
            assert self.prefix_encoder is not None
            assert prefix is not None
            mean, log_scale = self.prefix_encoder(**prefix)
            whitened = (
                self.prefix_encoder.rsample(mean, log_scale) if sample else mean
            )
            return PlayerContext(
                skill=self.skill_transform(whitened),
                kl=self.prefix_encoder.kl_standard_normal(mean, log_scale),
                posterior_mean=mean,
                posterior_log_scale=log_scale,
            )

        if self.arm is ModelArm.ORACLE:
            assert oracle_skill is not None
            expected = (batch_size, self.config.prefix.skill_dimensions)
            if oracle_skill.shape != expected:
                raise ValueError(f"oracle_skill must have shape {expected}")
            if prefix is not None and oracle_skill.shape[0] != batch_size:
                raise ValueError("oracle_skill must align with the prefix batch")
            skill = oracle_skill.to(
                device=reference.device,
                dtype=self.behavior.skill.weight.dtype,
            )
            return PlayerContext(
                skill=skill,
                kl=skill.new_zeros(batch_size),
                posterior_mean=None,
                posterior_log_scale=None,
            )

        skill = torch.zeros(
            (batch_size, self.config.prefix.skill_dimensions),
            device=reference.device,
            dtype=self.behavior.skill.weight.dtype,
        )
        return PlayerContext(
            skill=skill,
            kl=skill.new_zeros(batch_size),
            posterior_mean=None,
            posterior_log_scale=None,
        )

    def transition_objective(
        self,
        batch: dict[str, torch.Tensor],
        *,
        kl_weight: float = 1.0,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Score mechanics without admitting stable player context."""
        return self.dynamics.objective(
            **batch,
            kl_weight=kl_weight,
            sample_posterior=sample_posterior,
        )

    def structural_objective(
        self,
        *,
        prefix: dict[str, torch.Tensor],
        target: PredictiveTarget,
        oracle_skill: torch.Tensor | None = None,
        context_kl_weight: float = 1.0,
        sample_context: bool = True,
    ) -> dict[str, torch.Tensor | PlayerContext]:
        """Score selection, evidence, behavior, completion, and churn."""
        if context_kl_weight < 0:
            raise ValueError("context_kl_weight must be non-negative")
        context = self.player_context(
            prefix=prefix,
            oracle_skill=oracle_skill,
            sample=sample_context,
        )
        batch_size = context.skill.shape[0]
        if torch.any(
            (target.episode_player < 0) | (target.episode_player >= batch_size)
        ):
            raise ValueError("episode_player index outside context batch")
        if torch.any(
            (target.action_player < 0) | (target.action_player >= batch_size)
        ):
            raise ValueError("action_player index outside context batch")

        episode_skill = context.skill[target.episode_player.long()]
        assignment_nll = -self.assignment_head.log_prob(
            target.served_difficulty,
            episode_skill,
            target.levels.long(),
            target.tiers.long(),
        ).mean()
        evidence_nll = -self.evidence_head.log_prob(
            target.evidence,
            episode_skill,
            target.levels.long(),
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
        mastery_after = mastery_before + self.config.mastery.update_rate * (
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
        churn_nll = (churn_losses * churn_mask).sum() / churn_mask.sum().clamp_min(
            1.0
        )
        if target.actions.numel():
            action_nll = self.behavior.negative_log_likelihood(
                action=target.actions,
                board=target.boards,
                goal_colour=target.goal_colours,
                moves_left=target.moves_left,
                goals_left=target.goals_left,
                skill=context.skill[target.action_player.long()],
                legal_actions=target.legal_actions,
            )
        else:
            action_nll = context.skill.sum() * 0.0
        context_kl = context.kl.mean()
        loss = (
            assignment_nll
            + evidence_nll
            + win_nll
            + churn_nll
            + action_nll
            + context_kl_weight * context_kl
        )
        return {
            "loss": loss,
            "assignment_nll": assignment_nll,
            "evidence_nll": evidence_nll,
            "win_nll": win_nll,
            "churn_nll": churn_nll,
            "action_nll": action_nll,
            "context_kl": context_kl,
            "context": context,
        }

    def objective(
        self,
        *,
        prefix: dict[str, torch.Tensor],
        target: PredictiveTarget,
        transitions: dict[str, torch.Tensor],
        oracle_skill: torch.Tensor | None = None,
        context_kl_weight: float = 1.0,
        dynamics_kl_weight: float = 1.0,
        sample_context: bool = True,
        sample_dynamics: bool = True,
    ) -> dict[str, torch.Tensor | PlayerContext]:
        """Combine the matched structural and generative training terms."""
        structural = self.structural_objective(
            prefix=prefix,
            target=target,
            oracle_skill=oracle_skill,
            context_kl_weight=context_kl_weight,
            sample_context=sample_context,
        )
        dynamics = self.transition_objective(
            transitions,
            kl_weight=dynamics_kl_weight,
            sample_posterior=sample_dynamics,
        )
        return {
            "loss": structural["loss"] + dynamics["loss"],
            "assignment_nll": structural["assignment_nll"],
            "evidence_nll": structural["evidence_nll"],
            "win_nll": structural["win_nll"],
            "churn_nll": structural["churn_nll"],
            "action_nll": structural["action_nll"],
            "context_kl": structural["context_kl"],
            "board_nll": dynamics["board_nll"],
            "counter_mse": dynamics["counter_mse"],
            "dynamics_kl": dynamics["kl"],
            "context": structural["context"],
        }

    def behavior_log_probabilities(
        self,
        *,
        boards: torch.Tensor,
        goal_colours: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        legal_actions: torch.Tensor,
        player_context: torch.Tensor,
        action_players: torch.Tensor,
    ) -> torch.Tensor:
        """Score action rows using their resolved stable player contexts."""
        if action_players.shape != (boards.shape[0],):
            raise ValueError("action_players must align with action rows")
        if player_context.ndim != 2 or player_context.shape[1] != (
            self.config.prefix.skill_dimensions
        ):
            raise ValueError("player_context has the wrong shape")
        if torch.any(
            (action_players < 0) | (action_players >= player_context.shape[0])
        ):
            raise ValueError("action_players contains an out-of-range index")
        return self.behavior(
            board=boards,
            goal_colour=goal_colours,
            moves_left=moves_left,
            goals_left=goals_left,
            skill=player_context[action_players.long()],
            legal_actions=legal_actions,
        )

    def parameter_counts(self) -> dict[str, int]:
        """Report shared and slow-context trainable capacity by component."""

        def count(module: nn.Module | None) -> int:
            return (
                0
                if module is None
                else sum(
                    parameter.numel()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
            )

        counts = {
            "dynamics": count(self.dynamics),
            "behavior": count(self.behavior),
            "slow_context": count(self.prefix_encoder),
            "structural_heads": sum(
                count(module)
                for module in (
                    self.assignment_head,
                    self.evidence_head,
                    self.win_head,
                    self.churn_head,
                )
            ),
        }
        counts["total"] = sum(counts.values())
        return counts


def build_generative_arms(
    config: GenerativeModelConfig = GenerativeModelConfig(),
) -> dict[ModelArm, GenerativeWorldModel]:
    return {arm: GenerativeWorldModel(arm, config) for arm in ModelArm}


__all__ = [
    "GenerativeModelConfig",
    "GenerativeWorldModel",
    "ModelArm",
    "PlayerContext",
    "build_generative_arms",
]