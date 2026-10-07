from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from match3_simulator.learned_model.heads import EvidenceHead
from match3_simulator.retention import ChurnSchedule, MasteryConfig
from match3_simulator.spec import BENCHMARK_CONFIG
from match3_simulator.experiments.data_base import LANDMARK


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(max(value, 1e-6))))


class RetentionHead(nn.Module):
    """Learned retention head (see module docstring)."""

    def __init__(self, context_width: int, *, churn: ChurnSchedule, mastery: MasteryConfig, warmup_scale: float = BENCHMARK_CONFIG.warmup_churn_scale, landmark: int = LANDMARK):
        super().__init__()
        n = len(churn.level_names)
        self.context_width = int(context_width)
        self.warmup_scale = float(warmup_scale)
        self.landmark = int(landmark)
        self.intercept = nn.Parameter(torch.tensor(churn.intercepts, dtype=torch.float32))
        self.raw_deviation = nn.Parameter(torch.tensor([_inverse_softplus(v) for v in churn.deviation_coefficients]))
        over = churn.overchallenge_deviation_coefficients or churn.deviation_coefficients
        self.raw_over = nn.Parameter(torch.tensor([_inverse_softplus(v) for v in over]))
        self.raw_margin = nn.Parameter(torch.tensor([_inverse_softplus(max(v, 0.1)) for v in churn.margin_deviation_coefficients]))
        m_over = churn.margin_overchallenge_deviation_coefficients or churn.margin_deviation_coefficients
        self.raw_margin_over = nn.Parameter(torch.tensor([_inverse_softplus(max(v, 0.1)) for v in m_over]))
        self.raw_margin_target = nn.Parameter(torch.atanh(torch.tensor(churn.margin_targets, dtype=torch.float32).clamp(-0.999, 0.999)))
        self.raw_mastery_target = nn.Parameter(torch.logit(torch.tensor(float(churn.mastery_target)).clamp(1e-3, 1 - 1e-3)))
        self.completion = nn.Parameter(torch.zeros(n))
        self.context = nn.Linear(self.context_width, 1)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)
        self.n_levels = n
        self.mastery_config = mastery

    @property
    def mastery_target(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_mastery_target)

    @property
    def margin_target(self) -> torch.Tensor:
        return torch.tanh(self.raw_margin_target)

    def logits(self, *, completion: torch.Tensor, margin: torch.Tensor, mastery_after: torch.Tensor, level: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inputs: completion / margin / mastery_after (..., ), level (...,) long, context (..., width). Outputs: logits (...,)."""
        li = level.long()
        md = mastery_after.to(torch.float32) - self.mastery_target
        m_coef = torch.where(md < 0, F.softplus(self.raw_over)[li], F.softplus(self.raw_deviation)[li])
        qd = margin.to(torch.float32).clamp(-1.0, 1.0) - self.margin_target[li]
        q_coef = torch.where(qd < 0, F.softplus(self.raw_margin_over)[li], F.softplus(self.raw_margin)[li])
        return self.intercept[li] + m_coef * md.square() + q_coef * qd.square() + self.completion[li] * completion.to(torch.float32) + self.context(context.to(torch.float32)).squeeze(-1)

    def scale(self, attempt: torch.Tensor) -> torch.Tensor:
        return torch.where(attempt.long() < self.landmark, torch.full_like(attempt, self.warmup_scale, dtype=torch.float32), torch.ones_like(attempt, dtype=torch.float32))

    def probabilities(self, *, completion, margin, mastery_after, level, context, attempt) -> torch.Tensor:
        return self.scale(attempt) * torch.sigmoid(self.logits(completion=completion, margin=margin, mastery_after=mastery_after, level=level, context=context))

    def nll(self, *, churn: torch.Tensor, **inputs) -> torch.Tensor:
        p = self.probabilities(**inputs).clamp(1e-6, 1 - 1e-6)
        return F.binary_cross_entropy(p, churn.to(p.dtype), reduction="none")


class AssignmentHead(nn.Module):
    """Gaussian anchoring head p(E | D, L, context) with an unconstrained linear context path and a per-level scale."""

    def __init__(self, context_width: int, n_levels: int = 3, n_tiers: int = 3):
        super().__init__()
        self.level_intercept = nn.Embedding(n_levels, 1)
        self.tier_intercept = nn.Embedding(n_tiers, 1)
        self.context = nn.Linear(context_width, 1)
        self.raw_scale = nn.Parameter(torch.full((n_levels,), _inverse_softplus(0.8)))

    def mean(self, context, level, tier):
        return self.level_intercept(level.long()).squeeze(-1) + self.tier_intercept(tier.long()).squeeze(-1) + self.context(context).squeeze(-1)

    def log_prob(self, served_difficulty, context, level, tier):
        return Normal(self.mean(context, level, tier), F.softplus(self.raw_scale)[level.long()] + 1e-4).log_prob(served_difficulty.to(torch.float32))


class CompletionHead(nn.Module):
    """Bernoulli anchoring head p(R = 1 | E, D, L, context)."""

    def __init__(self, context_width: int, n_levels: int = 3, n_tiers: int = 3):
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(n_levels, n_tiers))
        self.difficulty = nn.Parameter(torch.full((n_levels,), -1.0))
        self.context = nn.Linear(context_width, 1)

    def logits(self, served_difficulty, context, level, tier):
        return self.intercept[level.long(), tier.long()] + self.difficulty[level.long()] * served_difficulty.to(torch.float32) + self.context(context).squeeze(-1)

    def log_prob(self, completion, served_difficulty, context, level, tier):
        return -F.binary_cross_entropy_with_logits(self.logits(served_difficulty, context, level, tier), completion.to(torch.float32), reduction="none")


class MarginHead(nn.Module):
    """Gaussian anchoring head p(Q | E, D, L, context)."""

    def __init__(self, context_width: int, n_levels: int = 3, n_tiers: int = 3):
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(n_levels, n_tiers))
        self.difficulty = nn.Parameter(torch.full((n_levels,), -0.1))
        self.context = nn.Linear(context_width, 1)
        self.raw_scale = nn.Parameter(torch.full((n_levels,), _inverse_softplus(0.3)))

    def mean(self, served_difficulty, context, level, tier):
        return self.intercept[level.long(), tier.long()] + self.difficulty[level.long()] * served_difficulty.to(torch.float32) + self.context(context).squeeze(-1)

    def log_prob(self, margin, served_difficulty, context, level, tier):
        return Normal(self.mean(served_difficulty, context, level, tier), F.softplus(self.raw_scale)[level.long()] + 1e-4).log_prob(margin.to(torch.float32))


class ActionSummaryHead(nn.Module):
    """Gaussian anchoring head for the per-attempt action-quality summaries (mean legal-move count / 10, mean goal clears of the chosen swap)."""

    def __init__(self, context_width: int, n_levels: int = 3, n_summaries: int = 2):
        super().__init__()
        self.intercept = nn.Embedding(n_levels, n_summaries)
        self.context = nn.Linear(context_width, n_summaries)
        self.raw_scale = nn.Parameter(torch.full((n_summaries,), _inverse_softplus(1.0)))

    def log_prob(self, summaries, context, level):
        mean = self.intercept(level.long()) + self.context(context)
        return Normal(mean, F.softplus(self.raw_scale) + 1e-4).log_prob(summaries.to(torch.float32)).sum(-1)


class EvidenceAdapter(nn.Module):
    """Linear map from the context to the 4-d skill coordinates expected by the schema-aware EvidenceHead (support clamped for the Beta / LogNormal terms)."""

    def __init__(self, context_width: int):
        super().__init__()
        self.project = nn.Linear(context_width, 4)
        self.head = EvidenceHead(skill_dimensions=4)

    def log_prob(self, evidence, context, level):
        from match3_simulator.evidence import EVIDENCE_SPECS

        x = evidence.to(torch.float32).clone()
        for i, spec in enumerate(EVIDENCE_SPECS):
            if spec.distribution == "beta":
                x[..., i] = x[..., i].clamp(1e-4, 1 - 1e-4)
            elif spec.distribution == "log_normal":
                x[..., i] = x[..., i].clamp_min(1e-4)
            elif spec.distribution == "poisson":
                x[..., i] = x[..., i].clamp_min(0.0).round()
        return self.head.log_prob(x, self.project(context), level.long())


__all__ = ["ActionSummaryHead", "AssignmentHead", "CompletionHead", "EvidenceAdapter", "MarginHead", "RetentionHead"]
