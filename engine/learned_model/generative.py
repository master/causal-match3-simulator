"""Typed gameplay encoders and generative RSSM training objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..scm import TIER_MOVE_BUDGETS
from .rssm import FastRSSM, RSSMConfig
from .tokens import BOARD_HEIGHT, BOARD_WIDTH, N_CELLS


@dataclass(frozen=True)
class GameplayRSSMConfig:
    n_colours: int = 6
    n_levels: int = 3
    n_tiers: int = 3
    embedding_size: int = 32
    observation_size: int = 128
    task_context_size: int = 32
    hidden_size: int = 128
    stochastic_size: int = 32
    max_moves_left: int = max(TIER_MOVE_BUDGETS)
    goals_scale: float = 32.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


class GameplayRSSM(nn.Module):
    """Encode match-3 states and train a shared fast generative model."""

    def __init__(self, config: GameplayRSSMConfig = GameplayRSSMConfig()):
        super().__init__()
        self.config = config
        embedding = config.embedding_size
        self.colour = nn.Embedding(config.n_colours, embedding)
        self.row = nn.Embedding(BOARD_HEIGHT, embedding)
        self.col = nn.Embedding(BOARD_WIDTH, embedding)
        self.is_goal = nn.Embedding(2, embedding)
        self.observation_projection = nn.Sequential(
            nn.Linear(embedding + 2, config.observation_size),
            nn.GELU(),
            nn.Linear(config.observation_size, config.observation_size),
        )
        self.level = nn.Embedding(config.n_levels, embedding)
        self.tier = nn.Embedding(config.n_tiers, embedding)
        self.task_projection = nn.Sequential(
            nn.Linear(2 * embedding + 1, config.task_context_size),
            nn.GELU(),
            nn.Linear(config.task_context_size, config.task_context_size),
        )
        self.initial_board_decoder = nn.Sequential(
            nn.Linear(config.task_context_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(
                config.hidden_size, N_CELLS * config.n_colours
            ),
        )
        self.initial_counter_decoder = nn.Sequential(
            nn.Linear(config.task_context_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),
        )
        self.rssm = FastRSSM(
            RSSMConfig(
                n_colours=config.n_colours,
                hidden_size=config.hidden_size,
                stochastic_size=config.stochastic_size,
                observation_size=config.observation_size,
                context_size=config.task_context_size,
            )
        )
        positions = torch.arange(N_CELLS)
        self.register_buffer(
            "cell_rows", positions // BOARD_WIDTH, persistent=False
        )
        self.register_buffer(
            "cell_cols", positions % BOARD_WIDTH, persistent=False
        )

    def encode_observation(
        self,
        boards: torch.Tensor,
        goal_colours: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
    ) -> torch.Tensor:
        """Encode categorical board state and normalized counters."""
        if boards.ndim != 3 or boards.shape[-1] != N_CELLS:
            raise ValueError("boards must have shape (batch, steps, 64)")
        batch_size, steps, _ = boards.shape
        expected = (batch_size, steps)
        for name, values in {
            "goal_colours": goal_colours,
            "moves_left": moves_left,
            "goals_left": goals_left,
        }.items():
            if values.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        tiles = (
            self.colour(boards.long())
            + self.row(self.cell_rows).view(1, 1, N_CELLS, -1)
            + self.col(self.cell_cols).view(1, 1, N_CELLS, -1)
            + self.is_goal(
                (boards == goal_colours.unsqueeze(-1)).long()
            )
        )
        board_embedding = tiles.mean(dim=-2)
        counters = torch.stack(
            (
                moves_left.to(board_embedding.dtype)
                / self.config.max_moves_left,
                goals_left.to(board_embedding.dtype) / self.config.goals_scale,
            ),
            dim=-1,
        )
        return self.observation_projection(
            torch.cat((board_embedding, counters), dim=-1)
        )

    def encode_task(
        self,
        levels: torch.Tensor,
        tiers: torch.Tensor,
        served_difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observed level, baseline tier, and served difficulty."""
        if levels.ndim != 1 or tiers.shape != levels.shape:
            raise ValueError("levels and tiers must be aligned vectors")
        if served_difficulty.shape != levels.shape:
            raise ValueError("served_difficulty must align with levels")
        return self.task_projection(
            torch.cat(
                (
                    self.level(levels.long()),
                    self.tier(tiers.long()),
                    served_difficulty.to(self.level.weight.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
        )

    def initial_state_predictions(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameterize the opening state distribution from observed task data."""
        if context.ndim != 2 or context.shape[1] != self.config.task_context_size:
            raise ValueError("context has the wrong shape")
        board_logits = self.initial_board_decoder(context).view(
            context.shape[0], N_CELLS, self.config.n_colours
        )
        return board_logits, self.initial_counter_decoder(context)

    def objective(
        self,
        *,
        boards: torch.Tensor,
        next_boards: torch.Tensor,
        actions: torch.Tensor,
        goal_colours: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        next_moves_left: torch.Tensor,
        next_goals_left: torch.Tensor,
        levels: torch.Tensor,
        tiers: torch.Tensor,
        served_difficulty: torch.Tensor,
        step_mask: torch.Tensor,
        kl_weight: float = 1.0,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Score next gameplay states under an observed RSSM trajectory."""
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        if next_boards.shape != boards.shape:
            raise ValueError("next_boards must align with boards")
        batch_size, steps, _ = boards.shape
        expected = (batch_size, steps)
        for name, values in {
            "actions": actions,
            "next_moves_left": next_moves_left,
            "next_goals_left": next_goals_left,
            "step_mask": step_mask,
        }.items():
            if values.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        observations = self.encode_observation(
            boards, goal_colours, moves_left, goals_left
        )
        context = self.encode_task(levels, tiers, served_difficulty)
        initial_board_logits, initial_counter_mean = (
            self.initial_state_predictions(context)
        )
        result = self.rssm.observe(
            observations=observations,
            actions=actions,
            context=context,
            sample_posterior=sample_posterior,
        )
        mask = step_mask.to(result["board_logits"].dtype)
        transition_count = mask.sum().clamp_min(1.0)
        state_count = transition_count + batch_size
        board_losses = F.cross_entropy(
            result["board_logits"].reshape(-1, self.config.n_colours),
            next_boards.long().reshape(-1),
            reduction="none",
        ).view(batch_size, steps, N_CELLS).mean(dim=-1)
        initial_board_losses = F.cross_entropy(
            initial_board_logits.reshape(-1, self.config.n_colours),
            boards[:, 0].long().reshape(-1),
            reduction="none",
        ).view(batch_size, N_CELLS).mean(dim=-1)
        board_nll = (
            (board_losses * mask).sum() + initial_board_losses.sum()
        ) / state_count
        counter_targets = torch.stack(
            (
                next_moves_left.to(result["counter_mean"].dtype)
                / self.config.max_moves_left,
                next_goals_left.to(result["counter_mean"].dtype)
                / self.config.goals_scale,
            ),
            dim=-1,
        )
        counter_losses = (
            result["counter_mean"] - counter_targets
        ).square().mean(dim=-1)
        initial_counter_targets = torch.stack(
            (
                moves_left[:, 0].to(initial_counter_mean.dtype)
                / self.config.max_moves_left,
                goals_left[:, 0].to(initial_counter_mean.dtype)
                / self.config.goals_scale,
            ),
            dim=-1,
        )
        initial_counter_losses = (
            initial_counter_mean - initial_counter_targets
        ).square().mean(dim=-1)
        counter_mse = (
            (counter_losses * mask).sum() + initial_counter_losses.sum()
        ) / state_count
        kl = (result["kl"] * mask).sum() / transition_count
        loss = board_nll + counter_mse + kl_weight * kl
        return {
            "loss": loss,
            "board_nll": board_nll,
            "counter_mse": counter_mse,
            "kl": kl,
            "board_logits": result["board_logits"],
            "counter_mean": result["counter_mean"],
            "initial_board_logits": initial_board_logits,
            "initial_counter_mean": initial_counter_mean,
            "hidden": result["hidden"],
            "stochastic": result["stochastic"],
        }


__all__ = ["GameplayRSSM", "GameplayRSSMConfig"]