"""Hierarchical causal-prefix encoder for continuous player skill."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..scm import TIER_MOVE_BUDGETS
from .tokens import ACTION_SLOTS, N_CELLS


@dataclass(frozen=True)
class PrefixEncoderConfig:
    n_colours: int = 6
    n_levels: int = 3
    n_tiers: int = 3
    proxy_dimensions: int = 12
    skill_dimensions: int = 4
    hidden_size: int = 64
    max_moves_left: int = max(TIER_MOVE_BUDGETS)


class CausalPrefixEncoder(nn.Module):
    """Encode only completed episodes before a decision into q(U_i)."""

    def __init__(self, config: PrefixEncoderConfig = PrefixEncoderConfig()):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.colour = nn.Embedding(config.n_colours, width)
        self.action = nn.Embedding(ACTION_SLOTS, width)
        self.level = nn.Embedding(config.n_levels, width)
        self.tier = nn.Embedding(config.n_tiers, width)
        self.step_projection = nn.Sequential(
            nn.Linear(2 * width + 2, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.episode_projection = nn.Sequential(
            nn.Linear(3 * width + config.proxy_dimensions + 2, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.episode_gru = nn.GRU(width, width, batch_first=True)
        self.baseline_projection = nn.Sequential(
            nn.Linear(config.proxy_dimensions, width),
            nn.GELU(),
        )
        self.no_history = nn.Parameter(torch.zeros(width))
        self.posterior = nn.Linear(2 * width, 2 * config.skill_dimensions)

    def forward(
        self,
        *,
        boards: torch.Tensor,
        actions: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        step_mask: torch.Tensor,
        levels: torch.Tensor,
        tiers: torch.Tensor,
        served_difficulty: torch.Tensor,
        outcomes: torch.Tensor,
        proxies: torch.Tensor,
        episode_mask: torch.Tensor,
        baseline_evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log scale in whitened skill coordinates."""
        if boards.ndim != 4 or boards.shape[-1] != N_CELLS:
            raise ValueError("boards must have shape (batch, episodes, steps, 64)")
        batch_size, n_episodes, n_steps, _ = boards.shape
        expected_steps = (batch_size, n_episodes, n_steps)
        for name, values in {
            "actions": actions,
            "moves_left": moves_left,
            "goals_left": goals_left,
            "step_mask": step_mask,
        }.items():
            if values.shape != expected_steps:
                raise ValueError(f"{name} must have shape {expected_steps}")
        expected_episodes = (batch_size, n_episodes)
        for name, values in {
            "levels": levels,
            "tiers": tiers,
            "served_difficulty": served_difficulty,
            "outcomes": outcomes,
            "episode_mask": episode_mask,
        }.items():
            if values.shape != expected_episodes:
                raise ValueError(f"{name} must have shape {expected_episodes}")
        if proxies.shape != (
            batch_size,
            n_episodes,
            self.config.proxy_dimensions,
        ):
            raise ValueError("proxies have the wrong shape")
        if baseline_evidence.shape != (
            batch_size,
            self.config.proxy_dimensions,
        ):
            raise ValueError("baseline_evidence has the wrong shape")

        episode_mask = episode_mask.bool()
        lengths = episode_mask.sum(dim=1)
        expected_mask = (
            torch.arange(n_episodes, device=episode_mask.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not torch.equal(episode_mask, expected_mask):
            raise ValueError("episode_mask must describe a contiguous prefix")

        board_embedding = self.colour(boards).mean(dim=-2)
        action_embedding = self.action(actions)
        counters = torch.stack(
            (
                moves_left.to(board_embedding.dtype)
                / self.config.max_moves_left,
                goals_left.to(board_embedding.dtype) / 32.0,
            ),
            dim=-1,
        )
        step_embedding = self.step_projection(
            torch.cat((board_embedding, action_embedding, counters), dim=-1)
        )
        valid_steps = step_mask.to(step_embedding.dtype).unsqueeze(-1)
        step_sum = (step_embedding * valid_steps).sum(dim=2)
        step_count = valid_steps.sum(dim=2).clamp_min(1.0)
        trajectory_embedding = step_sum / step_count

        context = torch.cat(
            (
                trajectory_embedding,
                self.level(levels),
                self.tier(tiers),
                served_difficulty.to(trajectory_embedding.dtype).unsqueeze(-1),
                outcomes.to(trajectory_embedding.dtype).unsqueeze(-1),
                proxies.to(trajectory_embedding.dtype),
            ),
            dim=-1,
        )
        episode_embedding = self.episode_projection(context)
        episode_embedding = episode_embedding * episode_mask.unsqueeze(-1)

        packed = pack_padded_sequence(
            episode_embedding,
            lengths.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.episode_gru(packed)
        history = hidden[-1]
        history = torch.where(
            (lengths == 0).unsqueeze(1),
            self.no_history.unsqueeze(0).expand(batch_size, -1),
            history,
        )
        baseline = self.baseline_projection(baseline_evidence.to(history.dtype))
        mean, log_scale = self.posterior(
            torch.cat((history, baseline), dim=-1)
        ).chunk(2, dim=-1)
        return mean, log_scale.clamp(-6.0, 3.0)

    @staticmethod
    def rsample(mean: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * log_scale.exp()

    @staticmethod
    def kl_standard_normal(
        mean: torch.Tensor, log_scale: torch.Tensor
    ) -> torch.Tensor:
        variance = torch.exp(2.0 * log_scale)
        return 0.5 * torch.sum(
            mean.square() + variance - 1.0 - 2.0 * log_scale,
            dim=-1,
        )


__all__ = ["CausalPrefixEncoder", "PrefixEncoderConfig"]