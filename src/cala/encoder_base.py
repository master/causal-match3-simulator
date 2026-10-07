"""Strict-prefix player encoder for CaLA: a port of ``learned_model.encoder.CausalPrefixEncoder`` with the completion margin Q added to the
per-attempt features and a learned constant in place of the (unshipped) baseline evidence X0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from match3_simulator.learned_model.tokens import ACTION_SLOTS, N_CELLS
from match3_simulator.scm import TIER_MOVE_BUDGETS


@dataclass(frozen=True)
class PrefixEncoderConfig:
    n_colours: int = 6
    n_levels: int = 3
    n_tiers: int = 3
    proxy_dimensions: int = 12
    latent_dimensions: int = 4
    hidden_size: int = 64
    max_moves_left: int = max(TIER_MOVE_BUDGETS)

    def __post_init__(self) -> None:
        if self.latent_dimensions < 1 or self.hidden_size < 8:
            raise ValueError("latent_dimensions must be positive and hidden_size >= 8")


class PrefixEncoder(nn.Module):
    """q_eta(Z | H_<a, B) over completed attempts strictly before the target."""

    def __init__(self, config: PrefixEncoderConfig = PrefixEncoderConfig()):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.colour = nn.Embedding(config.n_colours, width)
        self.action = nn.Embedding(ACTION_SLOTS, width)
        self.level = nn.Embedding(config.n_levels, width)
        self.tier = nn.Embedding(config.n_tiers, width)
        self.step_projection = nn.Sequential(nn.Linear(2 * width + 2, width), nn.GELU(), nn.Linear(width, width))
        self.episode_projection = nn.Sequential(nn.Linear(3 * width + config.proxy_dimensions + 3, width), nn.GELU(), nn.Linear(width, width))
        self.episode_gru = nn.GRU(width, width, batch_first=True)
        self.baseline = nn.Parameter(torch.zeros(width))  # learned constant standing in for the projected baseline evidence X0 (not shipped)
        self.no_history = nn.Parameter(torch.zeros(width))
        self.posterior = nn.Linear(2 * width, 2 * config.latent_dimensions)

    def forward(self, *, boards: torch.Tensor, actions: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor, step_mask: torch.Tensor, levels: torch.Tensor, tiers: torch.Tensor,
                served_difficulty: torch.Tensor, outcomes: torch.Tensor, margins: torch.Tensor, proxies: torch.Tensor, episode_mask: torch.Tensor, **_ignored) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs: boards (B, A, T, 64); actions / moves_left / goals_left / step_mask (B, A, T); levels / tiers / served_difficulty / outcomes / margins / episode_mask (B, A);
        proxies (B, A, 12). Outputs: posterior mean and clamped log-scale, each (B, latent_dimensions)."""
        if boards.ndim != 4 or boards.shape[-1] != N_CELLS:
            raise ValueError("boards must have shape (batch, episodes, steps, 64)")
        batch_size, n_episodes, n_steps, _ = boards.shape
        for name, values in {"actions": actions, "moves_left": moves_left, "goals_left": goals_left, "step_mask": step_mask}.items():
            if values.shape != (batch_size, n_episodes, n_steps):
                raise ValueError(f"{name} must have shape {(batch_size, n_episodes, n_steps)}")
        for name, values in {"levels": levels, "tiers": tiers, "served_difficulty": served_difficulty, "outcomes": outcomes, "margins": margins, "episode_mask": episode_mask}.items():
            if values.shape != (batch_size, n_episodes):
                raise ValueError(f"{name} must have shape {(batch_size, n_episodes)}")
        if proxies.shape != (batch_size, n_episodes, self.config.proxy_dimensions):
            raise ValueError("proxies have the wrong shape")
        episode_mask = episode_mask.bool()
        lengths = episode_mask.sum(dim=1)
        expected = torch.arange(n_episodes, device=episode_mask.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(episode_mask, expected):
            raise ValueError("episode_mask must describe a contiguous prefix")
        board_embedding = self.colour(boards.long()).mean(dim=-2)
        counters = torch.stack((moves_left.to(board_embedding.dtype) / self.config.max_moves_left, goals_left.to(board_embedding.dtype) / 32.0), dim=-1)
        step_embedding = self.step_projection(torch.cat((board_embedding, self.action(actions.long()), counters), dim=-1))
        valid = step_mask.to(step_embedding.dtype).unsqueeze(-1)
        trajectory = (step_embedding * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        context = torch.cat((trajectory, self.level(levels.long()), self.tier(tiers.long()), served_difficulty.to(trajectory.dtype).unsqueeze(-1), outcomes.to(trajectory.dtype).unsqueeze(-1),
                             margins.to(trajectory.dtype).unsqueeze(-1), proxies.to(trajectory.dtype)), dim=-1)
        episode_embedding = self.episode_projection(context) * episode_mask.unsqueeze(-1)
        packed = pack_padded_sequence(episode_embedding, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.episode_gru(packed)
        history = torch.where((lengths == 0).unsqueeze(1), self.no_history.unsqueeze(0).expand(batch_size, -1), hidden[-1])
        mean, log_scale = self.posterior(torch.cat((history, self.baseline.unsqueeze(0).expand(batch_size, -1)), dim=-1)).chunk(2, dim=-1)
        return mean, log_scale.clamp(-6.0, 3.0)

    @staticmethod
    def kl_standard_normal(mean: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        variance = torch.exp(2.0 * log_scale)
        return 0.5 * torch.sum(mean.square() + variance - 1.0 - 2.0 * log_scale, dim=-1)


__all__ = ["PrefixEncoder", "PrefixEncoderConfig"]
