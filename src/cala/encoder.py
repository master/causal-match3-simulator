"""
This is an implementation of strict-prefix encoders that share the across-attempt machinery of the frozen ``PrefixEncoder`` (level / tier embeddings, attempt MLP, GRU over
attempts 1..a-1, learned baseline, posterior head, latent width 4, hidden 64) and differ only in how one logged *step* is summarised.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from match3_simulator.learned_model.tokens import ACTION_SLOTS, BOARD_HEIGHT, BOARD_WIDTH, CELL1_TOKEN, CELL2_TOKEN, N_CELLS
from match3_simulator.experiments.encoder_base import PrefixEncoder, PrefixEncoderConfig
from match3_simulator.world_modeling.wm1_mechanics import N_SPECIAL_KINDS

ENCODERS: tuple[str, ...] = ("current", "matched", "spatial")
GOAL_DELTA_SCALE = 20.0
SPECIAL_COUNT_SCALE = 8.0


@dataclass(frozen=True)
class SpatialEncoderConfig:
    channels: int = 32
    layers: int = 3


class _PrefixEncoder4(PrefixEncoder):
    """Shared across-attempt path; subclasses implement ``step_features``. Forward signature = frozen encoder + specials / goal_delta / specials_after / goal_colours."""

    def _check(self, boards, actions, moves_left, goals_left, step_mask, levels, tiers, served_difficulty, outcomes, margins, proxies, episode_mask, specials, goal_delta, specials_after, goal_colours):
        if boards.ndim != 4 or boards.shape[-1] != N_CELLS:
            raise ValueError("boards must have shape (batch, episodes, steps, 64)")
        b, a, t, _ = boards.shape
        for name, values in {"actions": actions, "moves_left": moves_left, "goals_left": goals_left, "step_mask": step_mask, "goal_delta": goal_delta, "specials_after": specials_after}.items():
            if values.shape != (b, a, t):
                raise ValueError(f"{name} must have shape {(b, a, t)}")
        if specials.shape != boards.shape:
            raise ValueError("specials must align with boards")
        for name, values in {"levels": levels, "tiers": tiers, "served_difficulty": served_difficulty, "outcomes": outcomes, "margins": margins, "episode_mask": episode_mask, "goal_colours": goal_colours}.items():
            if values.shape != (b, a):
                raise ValueError(f"{name} must have shape {(b, a)}")
        if proxies.shape != (b, a, self.config.proxy_dimensions):
            raise ValueError("proxies have the wrong shape")
        episode_mask = episode_mask.bool()
        lengths = episode_mask.sum(dim=1)
        expected = torch.arange(a, device=episode_mask.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(episode_mask, expected):
            raise ValueError("episode_mask must describe a contiguous prefix")
        return episode_mask, lengths

    def step_features(self, *, boards, actions, moves_left, goals_left, specials, goal_delta, specials_after, goal_colours) -> torch.Tensor:  # (B, A, T, width)
        raise NotImplementedError

    def forward(self, *, boards, actions, moves_left, goals_left, step_mask, levels, tiers, served_difficulty, outcomes, margins, proxies, episode_mask, specials, goal_delta, specials_after, goal_colours, **_ignored):
        episode_mask, lengths = self._check(boards, actions, moves_left, goals_left, step_mask, levels, tiers, served_difficulty, outcomes, margins, proxies, episode_mask, specials, goal_delta, specials_after, goal_colours)
        batch_size = boards.shape[0]
        step_embedding = self.step_features(boards=boards, actions=actions, moves_left=moves_left, goals_left=goals_left, specials=specials, goal_delta=goal_delta, specials_after=specials_after, goal_colours=goal_colours)
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

    def _counters(self, moves_left, goals_left, goal_delta, specials_after, dtype):
        return torch.stack((moves_left.to(dtype) / self.config.max_moves_left, goals_left.to(dtype) / 32.0, goal_delta.to(dtype) / GOAL_DELTA_SCALE, specials_after.to(dtype) / SPECIAL_COUNT_SCALE), dim=-1)


class MatchedInputPrefixEncoder(_PrefixEncoder4):
    """Mean-pool architecture of the frozen encoder + special-kind bag + goal-colour share + transition scalars (matched-input control)."""

    def __init__(self, config: PrefixEncoderConfig = PrefixEncoderConfig()):
        super().__init__(config)
        width = config.hidden_size
        self.special_bag = nn.Embedding(N_SPECIAL_KINDS, width)
        # colour bag (w) + swap embedding (w) + special bag (w) + moves, goals, goal_delta, specials_after (4) + goal share (1)
        self.step_projection = nn.Sequential(nn.Linear(3 * width + 5, width), nn.GELU(), nn.Linear(width, width))

    def step_features(self, *, boards, actions, moves_left, goals_left, specials, goal_delta, specials_after, goal_colours):
        colour_bag = self.colour(boards.long()).mean(dim=-2)
        special_bag = self.special_bag(specials.long()).mean(dim=-2)
        goal_share = (boards.long() == goal_colours.long().unsqueeze(-1).unsqueeze(-1)).float().mean(dim=-1, keepdim=True)
        counters = self._counters(moves_left, goals_left, goal_delta, specials_after, colour_bag.dtype)
        return self.step_projection(torch.cat((colour_bag, self.action(actions.long()), special_bag, counters, goal_share), dim=-1))


class SpatialPrefixEncoder(_PrefixEncoder4):
    """CNN step encoder over colours, positions, special kinds, goal indicator and the recorded swap (see module docstring)."""

    def __init__(self, config: PrefixEncoderConfig = PrefixEncoderConfig(), spatial: SpatialEncoderConfig = SpatialEncoderConfig()):
        super().__init__(config)
        width = config.hidden_size
        del self.colour, self.action  # the frozen bag / slot embeddings are not used by the spatial step encoder (parameter counts stay honest)
        self.spatial = spatial
        in_channels = config.n_colours + N_SPECIAL_KINDS + 1 + 2 + 2  # colour one-hot, special one-hot, is-goal, swap cell 1 / cell 2, row / col coordinates
        layers: list[nn.Module] = []
        c_in = in_channels
        for _ in range(spatial.layers):
            layers += [nn.Conv2d(c_in, spatial.channels, kernel_size=3, padding=1), nn.GELU()]
            c_in = spatial.channels
        self.cnn = nn.Sequential(*layers)
        self.pool_projection = nn.Linear(2 * spatial.channels, width)
        self.step_projection = nn.Sequential(nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, width))
        rows = torch.arange(BOARD_HEIGHT).view(1, 1, BOARD_HEIGHT, 1).expand(1, 1, BOARD_HEIGHT, BOARD_WIDTH).float() / (BOARD_HEIGHT - 1)
        cols = torch.arange(BOARD_WIDTH).view(1, 1, 1, BOARD_WIDTH).expand(1, 1, BOARD_HEIGHT, BOARD_WIDTH).float() / (BOARD_WIDTH - 1)
        self.register_buffer("coords", torch.cat((rows, cols), dim=1), persistent=False)  # (1, 2, 8, 8)
        self.register_buffer("cell1", torch.as_tensor(CELL1_TOKEN), persistent=False)
        self.register_buffer("cell2", torch.as_tensor(CELL2_TOKEN), persistent=False)

    def grid_channels(self, boards: torch.Tensor, specials: torch.Tensor, goal_colours: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Inputs: boards / specials (N, 64), goal_colours / actions (N,). Outputs: (N, C, 8, 8) float."""
        n = boards.shape[0]
        colour = F.one_hot(boards.long().clamp(0, self.config.n_colours - 1), self.config.n_colours).float()
        special = F.one_hot(specials.long().clamp(0, N_SPECIAL_KINDS - 1), N_SPECIAL_KINDS).float()
        is_goal = (boards.long() == goal_colours.long().unsqueeze(-1)).float().unsqueeze(-1)
        a = actions.long().clamp(0, ACTION_SLOTS - 1)
        swap = torch.zeros((n, N_CELLS, 2), dtype=torch.float32, device=boards.device)
        swap[torch.arange(n, device=boards.device), self.cell1[a], 0] = 1.0
        swap[torch.arange(n, device=boards.device), self.cell2[a], 1] = 1.0
        cells = torch.cat((colour, special, is_goal, swap), dim=-1)  # (N, 64, C-2)
        grid = cells.transpose(1, 2).reshape(n, -1, BOARD_HEIGHT, BOARD_WIDTH)
        return torch.cat((grid, self.coords.expand(n, -1, -1, -1)), dim=1)

    def step_features(self, *, boards, actions, moves_left, goals_left, specials, goal_delta, specials_after, goal_colours):
        b, a, t, _ = boards.shape
        flat_boards = boards.reshape(-1, N_CELLS)
        flat_specials = specials.reshape(-1, N_CELLS)
        flat_goal = goal_colours.long().unsqueeze(-1).expand(-1, -1, t).reshape(-1)
        grid = self.grid_channels(flat_boards, flat_specials, flat_goal, actions.reshape(-1))
        h = self.cnn(grid)
        pooled = torch.cat((h.mean(dim=(2, 3)), h.amax(dim=(2, 3))), dim=-1)
        spatial = F.gelu(self.pool_projection(pooled)).view(b, a, t, -1)
        counters = self._counters(moves_left, goals_left, goal_delta, specials_after, spatial.dtype)
        return self.step_projection(torch.cat((spatial, counters), dim=-1))


def make_encoder(kind: str, config: PrefixEncoderConfig, *, spatial: SpatialEncoderConfig = SpatialEncoderConfig()) -> PrefixEncoder:
    if kind == "current":
        return PrefixEncoder(config)
    if kind == "matched":
        return MatchedInputPrefixEncoder(config)
    if kind == "spatial":
        return SpatialPrefixEncoder(config, spatial)
    raise ValueError(f"unknown encoder {kind!r}; expected one of {ENCODERS}")


__all__ = ["ENCODERS", "MatchedInputPrefixEncoder", "SpatialEncoderConfig", "SpatialPrefixEncoder", "make_encoder"]
