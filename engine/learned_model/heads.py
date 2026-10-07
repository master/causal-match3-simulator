"""Structural decoder heads for the continuous latent-skill model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, LogNormal, Normal, Poisson
from torch.nn import functional as F

from ..retention import CHURN_SCHEDULE
from ..spec import SKILL_COVARIANCE
from ..evidence import EVIDENCE_Q_MATRIX, EVIDENCE_SPECS


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


class CorrelatedSkillTransform(nn.Module):
    """Map whitened U~N(0,I) into K~N(0,Sigma_K)."""

    def __init__(self):
        super().__init__()
        cholesky = np.linalg.cholesky(SKILL_COVARIANCE)
        self.register_buffer(
            "cholesky", torch.as_tensor(cholesky, dtype=torch.float32)
        )

    def forward(self, whitened_skill: torch.Tensor) -> torch.Tensor:
        if whitened_skill.shape[-1] != self.cholesky.shape[0]:
            raise ValueError("whitened skill has the wrong final dimension")
        return whitened_skill @ self.cholesky.T


class AssignmentHead(nn.Module):
    """Gaussian p(E | D, L, K), with no model for p(D | K)."""

    def __init__(self, n_levels: int = 3, n_tiers: int = 3, skill_dimensions: int = 4):
        super().__init__()
        self.level_intercept = nn.Embedding(n_levels, 1)
        self.tier_intercept = nn.Embedding(n_tiers, 1)
        self.raw_skill = nn.Parameter(
            torch.full(
                (n_levels, skill_dimensions), _inverse_softplus(0.20)
            )
        )
        self.raw_scale = nn.Parameter(
            torch.full((n_levels,), _inverse_softplus(0.65))
        )

    @property
    def skill_coefficients(self) -> torch.Tensor:
        return F.softplus(self.raw_skill)

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + 1e-4

    def mean(
        self, skill: torch.Tensor, level: torch.Tensor, tier: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.level_intercept(level).squeeze(-1)
            + self.tier_intercept(tier).squeeze(-1)
            + (self.skill_coefficients[level] * skill).sum(-1)
        )

    def log_prob(
        self,
        served_difficulty: torch.Tensor,
        skill: torch.Tensor,
        level: torch.Tensor,
        tier: torch.Tensor,
    ) -> torch.Tensor:
        return Normal(self.mean(skill, level, tier), self.scale[level]).log_prob(
            served_difficulty
        )


class EvidenceHead(nn.Module):
    """Support-aware p(X | K, L) matching the declared evidence schema."""

    def __init__(
        self,
        skill_dimensions: int = 4,
        evidence_dimensions: int = 12,
        n_levels: int = 3,
    ):
        super().__init__()
        if evidence_dimensions != len(EVIDENCE_SPECS):
            raise ValueError(
                "evidence_dimensions must match the declared evidence schema"
            )
        if skill_dimensions != EVIDENCE_Q_MATRIX.shape[1]:
            raise ValueError("skill_dimensions must match the evidence Q-matrix")
        q_matrix = np.asarray(EVIDENCE_Q_MATRIX, dtype=np.float32)
        loading_mask = q_matrix != 0
        raw_loading = np.zeros_like(q_matrix)
        raw_loading[loading_mask] = np.log(
            np.expm1(np.abs(q_matrix[loading_mask]))
        )
        self.raw_loading = nn.Parameter(torch.as_tensor(raw_loading))
        self.register_buffer(
            "loading_sign", torch.as_tensor(np.sign(q_matrix))
        )
        self.register_buffer(
            "loading_mask", torch.as_tensor(loading_mask)
        )
        self.intercept = nn.Parameter(torch.zeros(evidence_dimensions))
        self.level_offset = nn.Embedding(n_levels, evidence_dimensions)
        nn.init.zeros_(self.level_offset.weight)
        initial_dispersion = []
        for spec in EVIDENCE_SPECS:
            value = spec.dispersion - 2.0 if spec.distribution == "beta" else spec.dispersion
            initial_dispersion.append(max(value, 0.1))
        self.raw_dispersion = nn.Parameter(
            torch.tensor(
                [_inverse_softplus(value) for value in initial_dispersion],
                dtype=torch.float32,
            )
        )

    @property
    def loadings(self) -> torch.Tensor:
        return (
            self.loading_sign
            * F.softplus(self.raw_loading)
            * self.loading_mask
        )

    @property
    def dispersion(self) -> torch.Tensor:
        return F.softplus(self.raw_dispersion) + 1e-4

    def log_prob(
        self,
        evidence: torch.Tensor,
        skill: torch.Tensor,
        level: torch.Tensor,
    ) -> torch.Tensor:
        if evidence.shape[-1] != len(EVIDENCE_SPECS):
            raise ValueError("evidence has the wrong final dimension")
        predictor = (
            skill @ self.loadings.T
            + self.intercept
            + self.level_offset(level.long())
        )
        terms = []
        for index, spec in enumerate(EVIDENCE_SPECS):
            observed = evidence[..., index]
            if spec.distribution == "beta":
                mean = torch.sigmoid(predictor[..., index])
                concentration = self.dispersion[index] + 2.0
                distribution = Beta(
                    mean * concentration,
                    (1.0 - mean) * concentration,
                )
            elif spec.distribution == "log_normal":
                distribution = LogNormal(
                    predictor[..., index], self.dispersion[index]
                )
            elif spec.distribution == "poisson":
                distribution = Poisson(torch.exp(predictor[..., index]).clamp_max(1e4))
            elif spec.distribution == "normal":
                distribution = Normal(
                    predictor[..., index], self.dispersion[index]
                )
            else:
                raise ValueError(
                    f"unsupported evidence distribution {spec.distribution}"
                )
            terms.append(distribution.log_prob(observed))
        return torch.stack(terms, dim=-1).sum(-1)


class WinHead(nn.Module):
    """Constrained p(R=1 | E, D, L, K)."""

    def __init__(self, n_levels: int = 3, n_tiers: int = 3, skill_dimensions: int = 4):
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(n_levels, n_tiers))
        self.raw_skill = nn.Parameter(
            torch.full(
                (n_levels, skill_dimensions), _inverse_softplus(0.20)
            )
        )
        self.raw_difficulty = nn.Parameter(
            torch.full((n_levels,), _inverse_softplus(1.0))
        )

    @property
    def skill_coefficients(self) -> torch.Tensor:
        return F.softplus(self.raw_skill)

    @property
    def difficulty_coefficients(self) -> torch.Tensor:
        return F.softplus(self.raw_difficulty) + 1e-4

    def logits(
        self,
        served_difficulty: torch.Tensor,
        skill: torch.Tensor,
        level: torch.Tensor,
        tier: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.intercept[level, tier]
            + (self.skill_coefficients[level] * skill).sum(-1)
            - self.difficulty_coefficients[level] * served_difficulty
        )

    def probabilities(
        self,
        served_difficulty: torch.Tensor,
        skill: torch.Tensor,
        level: torch.Tensor,
        tier: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.logits(served_difficulty, skill, level, tier))


@dataclass(frozen=True)
class ChurnHeadConfig:
    mastery_target: float = CHURN_SCHEDULE.mastery_target

    def __post_init__(self) -> None:
        if not 0.0 < self.mastery_target < 1.0:
            raise ValueError("mastery_target must lie in (0, 1)")


class ChurnHead(nn.Module):
    """Constrained U-shaped p(C=1 | M_after, L)."""

    def __init__(
        self,
        n_levels: int = 3,
        config: ChurnHeadConfig = ChurnHeadConfig(),
    ):
        super().__init__()
        self.config = config
        if n_levels != len(CHURN_SCHEDULE.level_names):
            raise ValueError("n_levels must match the ground-truth churn schedule")
        self.intercept = nn.Parameter(torch.tensor(CHURN_SCHEDULE.intercepts))
        self.raw_deviation = nn.Parameter(
            torch.tensor(
                [
                    _inverse_softplus(value)
                    for value in CHURN_SCHEDULE.deviation_coefficients
                ]
            )
        )
        overchallenge_deviation = (
            CHURN_SCHEDULE.deviation_coefficients
            if CHURN_SCHEDULE.overchallenge_deviation_coefficients is None
            else CHURN_SCHEDULE.overchallenge_deviation_coefficients
        )
        self.raw_overchallenge_deviation = nn.Parameter(
            torch.tensor(
                [_inverse_softplus(value) for value in overchallenge_deviation]
            )
        )
        self.raw_margin_deviation = nn.Parameter(
            torch.tensor(
                [
                    _inverse_softplus(max(value, 0.1))
                    for value in CHURN_SCHEDULE.margin_deviation_coefficients
                ]
            )
        )
        margin_overchallenge_deviation = (
            CHURN_SCHEDULE.margin_deviation_coefficients
            if CHURN_SCHEDULE.margin_overchallenge_deviation_coefficients is None
            else CHURN_SCHEDULE.margin_overchallenge_deviation_coefficients
        )
        self.raw_margin_overchallenge_deviation = nn.Parameter(
            torch.tensor(
                [
                    _inverse_softplus(max(value, 0.1))
                    for value in margin_overchallenge_deviation
                ]
            )
        )
        self.raw_margin_target = nn.Parameter(
            torch.atanh(
                torch.tensor(CHURN_SCHEDULE.margin_targets).clamp(-0.999, 0.999)
            )
        )

    @property
    def deviation_coefficient(self) -> torch.Tensor:
        return F.softplus(self.raw_deviation)

    @property
    def overchallenge_deviation_coefficient(self) -> torch.Tensor:
        return F.softplus(self.raw_overchallenge_deviation)

    @property
    def margin_deviation_coefficient(self) -> torch.Tensor:
        return F.softplus(self.raw_margin_deviation)

    @property
    def margin_overchallenge_deviation_coefficient(self) -> torch.Tensor:
        return F.softplus(self.raw_margin_overchallenge_deviation)

    @property
    def margin_target(self) -> torch.Tensor:
        return torch.tanh(self.raw_margin_target)

    def logits(
        self,
        mastery_after: torch.Tensor,
        level: torch.Tensor,
        *,
        completion_margin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        level_index = level.long()
        margin = (
            self.margin_target[level_index]
            if completion_margin is None
            else completion_margin.to(mastery_after.dtype)
        )
        mastery_deviation = mastery_after - self.config.mastery_target
        mastery_coefficient = torch.where(
            mastery_deviation < 0.0,
            self.overchallenge_deviation_coefficient[level_index],
            self.deviation_coefficient[level_index],
        )
        margin_deviation = margin - self.margin_target[level_index]
        margin_coefficient = torch.where(
            margin_deviation < 0.0,
            self.margin_overchallenge_deviation_coefficient[level_index],
            self.margin_deviation_coefficient[level_index],
        )
        return (
            self.intercept[level_index]
            + mastery_coefficient * mastery_deviation.square()
            + margin_coefficient * margin_deviation.square()
        )

    def probabilities(
        self,
        mastery_after: torch.Tensor,
        level: torch.Tensor,
        *,
        completion_margin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.logits(
                mastery_after,
                level,
                completion_margin=completion_margin,
            )
        )


__all__ = [
    "AssignmentHead",
    "ChurnHead",
    "ChurnHeadConfig",
    "CorrelatedSkillTransform",
    "EvidenceHead",
    "WinHead",
]