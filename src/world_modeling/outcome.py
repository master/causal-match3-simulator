"""Outcome channel shared by every WM-1 family: cleared-cell Bernoulli head with the round-0 support clamp, the Poisson-binomial derived
goal count, the signed residual head (goal tiles that arrive by refill and clear in the same move, minus over-counts of the settled
reconstruction), and the terminal head. Pure functions over tensors so the transformer and the latent models score and sample identically.

Definitions (per transition row, pre-move cell indexing):
  p_c            = P(cell c cleared) = 1 on the round-0 clear set (deterministic given board, specials, action) else sigmoid(logit_c)
  derived count  ~ PoissonBinomial({p_c : board[c] == goal colour})           (sampled, never its mean, in rollouts)
  residual r     ~ Categorical over [residual_min, residual_min + classes)     target = logged goal delta - derived count of the target mask
  goal delta     = derived count + r, goals_left' = max(0, goals_left - delta)
  terminal logit -> BCE against 1{goals_left' == 0 or moves_left' == 0}
Win rows (goals_left' == 0) censor the residual: the logged delta is a lower bound on the tiles cleared, so the residual loss is the tail
mass -log P(r >= r_logged) instead of the point mass.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from match3_simulator.world_modeling.cleared_mask import poisson_binomial_pmf


class ResidualSpec:
    """Signed residual vocabulary."""

    def __init__(self, minimum: int, classes: int):
        if classes < 1:
            raise ValueError("residual classes must be positive")
        self.minimum = int(minimum)
        self.classes = int(classes)

    def index_of(self, residual: torch.Tensor) -> torch.Tensor:
        """Inputs: signed residual (rows,). Outputs: class index clamped into the vocabulary."""
        return (residual.long() - self.minimum).clamp(0, self.classes - 1)

    def clamped(self, residual: torch.Tensor) -> torch.Tensor:
        return (residual.long() < self.minimum) | (residual.long() >= self.minimum + self.classes)

    def values(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.arange(self.minimum, self.minimum + self.classes, device=device, dtype=dtype)

    def expected(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.softmax(dim=-1) * self.values(logits.device, logits.dtype)).sum(dim=-1)

    def sample(self, logits: torch.Tensor, *, stochastic: bool, generator: torch.Generator | None) -> torch.Tensor:
        if stochastic:
            index = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(1)
        else:
            index = logits.argmax(dim=-1)
        return index + self.minimum

    def nll(self, logits: torch.Tensor, residual: torch.Tensor, censored: torch.Tensor | None = None) -> torch.Tensor:
        """Point-mass NLL, or tail-mass NLL -log P(r >= target) where censored is set.

        Inputs: logits (rows, classes); residual (rows,) signed; censored (rows,) bool or None. Outputs: NLL (rows,).
        """
        target = self.index_of(residual)
        point = F.cross_entropy(logits, target, reduction="none")
        if censored is None:
            return point
        log_probs = logits.log_softmax(dim=-1)
        classes = torch.arange(self.classes, device=logits.device)
        tail = torch.logsumexp(log_probs.masked_fill(classes.unsqueeze(0) < target.unsqueeze(1), float("-inf")), dim=-1)
        return torch.where(censored, -tail, point)


def cleared_probabilities(mask_logits: torch.Tensor, round0_pre: torch.Tensor | None) -> torch.Tensor:
    """Per-cell clear probabilities with the round-0 support clamp.

    Inputs: mask_logits (rows, 64); round0_pre (rows, 64) bool or None. Outputs: probabilities (rows, 64).
    """
    probabilities = torch.sigmoid(mask_logits)
    if round0_pre is None:
        return probabilities
    return torch.where(round0_pre, torch.ones_like(probabilities), probabilities)


def mask_bce(mask_logits: torch.Tensor, target: torch.Tensor, round0_pre: torch.Tensor | None) -> torch.Tensor:
    """Binary cross-entropy of the cleared mask averaged over the cells the model must actually predict (round-0 cells excluded).

    Inputs: mask_logits (rows, 64); target (rows, 64) {0,1}; round0_pre (rows, 64) bool or None. Outputs: per-row mean BCE (rows,).
    """
    losses = F.binary_cross_entropy_with_logits(mask_logits, target.to(mask_logits.dtype), reduction="none")
    if round0_pre is None:
        return losses.mean(dim=-1)
    free = (~round0_pre).to(losses.dtype)
    return (losses * free).sum(dim=-1) / free.sum(dim=-1).clamp_min(1.0)


def derived_goal_nll(probabilities: torch.Tensor, goal_cells: torch.Tensor, target_count: torch.Tensor, max_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poisson-binomial NLL of the target derived count (tail folded at max_count).

    Inputs: probabilities (rows, 64); goal_cells (rows, 64) bool; target_count (rows,) long; max_count.
    Outputs: NLL (rows,), log-pmf (rows, max_count + 1), clamped flag (rows,).
    """
    pmf = poisson_binomial_pmf(probabilities, goal_cells, max_count=max_count)
    clamped = target_count > max_count
    nll = -torch.log(pmf.gather(1, target_count.clamp(0, max_count).unsqueeze(1)).squeeze(1).clamp_min(1e-9))
    return nll, torch.log(pmf.clamp_min(1e-9)), clamped


def sample_goal_delta(probabilities: torch.Tensor, goal_cells: torch.Tensor, residual_logits: torch.Tensor, residual: ResidualSpec, *, stochastic: bool,
                      generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the cleared mask and the goal delta (derived count + residual, floored at zero).

    Inputs: probabilities (rows, 64); goal_cells (rows, 64) bool; residual logits (rows, classes); spec; stochastic flag; generator.
    Outputs: cleared (rows, 64) bool and goal delta (rows,) long >= 0.
    """
    if stochastic:
        draws = torch.rand(probabilities.shape, generator=generator, device=probabilities.device, dtype=probabilities.dtype)
        cleared = draws < probabilities
    else:
        cleared = probabilities >= 0.5
    delta = (cleared & goal_cells).sum(dim=-1) + residual.sample(residual_logits, stochastic=stochastic, generator=generator)
    return cleared, delta.clamp_min(0)


class TerminalHead(nn.Module):
    """Logit of 'the attempt ends after this move' from a global feature, the normalised counters and the expected goal delta."""

    def __init__(self, feature_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_size + 3, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))

    def forward(self, features: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor, expected_delta: torch.Tensor, *, max_moves: int, goals_scale: float) -> torch.Tensor:
        """Inputs: features (rows, F); moves_left, goals_left, expected_delta (rows,). Outputs: logit (rows,)."""
        extra = torch.stack((moves_left.to(features.dtype) / max_moves, goals_left.to(features.dtype) / goals_scale, expected_delta.to(features.dtype) / goals_scale), dim=-1)
        return self.network(torch.cat((features, extra), dim=-1)).squeeze(-1)


def terminal_target(next_moves_left: torch.Tensor, next_goals_left: torch.Tensor) -> torch.Tensor:
    """Inputs: next counters (rows,). Outputs: bool (rows,) terminal flag."""
    return (next_goals_left.long() <= 0) | (next_moves_left.long() <= 0)


__all__ = ["ResidualSpec", "TerminalHead", "cleared_probabilities", "derived_goal_nll", "mask_bce", "sample_goal_delta", "terminal_target"]
