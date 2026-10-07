"""Spatial transformer decoder for p(A_t | S_t, K_i)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..scm import TIER_MOVE_BUDGETS
from .tokens import (
    ACTION_SLOTS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CELL1_TOKEN,
    CELL2_TOKEN,
    IN_BOUNDS,
    N_CELLS,
)


@dataclass(frozen=True)
class ActionPolicyConfig:
    n_colours: int = 6
    skill_dimensions: int = 4
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.0
    max_moves_left: int = max(TIER_MOVE_BUDGETS)
    mask_fill: float = -1e9
    use_immediate_features: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")


class ContinuousActionPolicy(nn.Module):
    """Predict a legal action from current Markov state and continuous skill."""

    def __init__(self, config: ActionPolicyConfig = ActionPolicyConfig()):
        super().__init__()
        self.config = config
        width = config.d_model
        self.colour = nn.Embedding(config.n_colours, width)
        self.row = nn.Embedding(BOARD_HEIGHT, width)
        self.col = nn.Embedding(BOARD_WIDTH, width)
        self.is_goal = nn.Embedding(2, width)
        self.skill = nn.Linear(config.skill_dimensions, width)
        self.moves = nn.Embedding(config.max_moves_left + 1, width)
        self.goals = nn.Linear(1, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.n_heads,
            dim_feedforward=4 * width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        action_input_width = 3 * width + (2 if config.use_immediate_features else 0)
        self.action_head = nn.Sequential(
            nn.Linear(action_input_width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

        positions = torch.arange(N_CELLS)
        self.register_buffer("cell_rows", positions // BOARD_WIDTH, persistent=False)
        self.register_buffer("cell_cols", positions % BOARD_WIDTH, persistent=False)
        self.register_buffer("cell1", torch.as_tensor(CELL1_TOKEN), persistent=False)
        self.register_buffer("cell2", torch.as_tensor(CELL2_TOKEN), persistent=False)
        self.register_buffer("in_bounds", torch.as_tensor(IN_BOUNDS), persistent=False)

    def immediate_action_features(
        self, board: torch.Tensor, goal_colour: torch.Tensor
    ) -> torch.Tensor:
        """Return total and goal clears for every fixed-vocabulary swap."""
        batch_size = board.shape[0]
        swapped = board.unsqueeze(1).expand(-1, ACTION_SLOTS, -1).clone()
        first_index = self.cell1.view(1, -1, 1).expand(batch_size, -1, -1)
        second_index = self.cell2.view(1, -1, 1).expand(batch_size, -1, -1)
        first_value = swapped.gather(2, first_index)
        second_value = swapped.gather(2, second_index)
        swapped.scatter_(2, first_index, second_value)
        swapped.scatter_(2, second_index, first_value)
        grid = swapped.view(
            batch_size, ACTION_SLOTS, BOARD_HEIGHT, BOARD_WIDTH
        )
        matched = torch.zeros_like(grid, dtype=torch.bool)
        horizontal = (
            (grid[..., :, :-2] == grid[..., :, 1:-1])
            & (grid[..., :, 1:-1] == grid[..., :, 2:])
        )
        matched[..., :, :-2] |= horizontal
        matched[..., :, 1:-1] |= horizontal
        matched[..., :, 2:] |= horizontal
        vertical = (
            (grid[..., :-2, :] == grid[..., 1:-1, :])
            & (grid[..., 1:-1, :] == grid[..., 2:, :])
        )
        matched[..., :-2, :] |= vertical
        matched[..., 1:-1, :] |= vertical
        matched[..., 2:, :] |= vertical
        total_cleared = matched.sum(dim=(-1, -2)).to(torch.float32)
        goal_cleared = (
            matched & (grid == goal_colour.view(-1, 1, 1, 1))
        ).sum(dim=(-1, -2)).to(torch.float32)
        return torch.stack((total_cleared, goal_cleared), dim=-1)

    @staticmethod
    def _standardize_action_features(
        features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        weights = mask.to(features.dtype).unsqueeze(-1)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (features * weights).sum(dim=1, keepdim=True) / count
        variance = (
            (features - mean).square() * weights
        ).sum(dim=1, keepdim=True) / count
        scale = variance.sqrt()
        standardized = torch.where(
            scale > 1e-6,
            (features - mean) / scale.clamp_min(1e-6),
            torch.zeros_like(features),
        )
        return standardized * weights

    def forward(
        self,
        board: torch.Tensor,
        goal_colour: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        skill: torch.Tensor,
        legal_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized log probabilities over the fixed action vocabulary."""
        if board.ndim != 2 or board.shape[1] != N_CELLS:
            raise ValueError("board must have shape (batch, 64)")
        batch_size = board.shape[0]
        if skill.shape != (batch_size, self.config.skill_dimensions):
            raise ValueError("skill must have shape (batch, skill_dimensions)")
        if legal_actions.shape != (batch_size, ACTION_SLOTS):
            raise ValueError("legal_actions must have shape (batch, 128)")

        tiles = (
            self.colour(board)
            + self.row(self.cell_rows).unsqueeze(0)
            + self.col(self.cell_cols).unsqueeze(0)
            + self.is_goal((board == goal_colour.unsqueeze(1)).long())
        )
        skill_token = self.skill(skill).unsqueeze(1)
        move_token = self.moves(
            moves_left.clamp(0, self.config.max_moves_left)
        ).unsqueeze(1)
        goal_token = self.goals(
            (goals_left.to(tiles.dtype) / 32.0).unsqueeze(1)
        ).unsqueeze(1)
        hidden = self.trunk(
            torch.cat((tiles, skill_token, move_token, goal_token), dim=1)
        )
        first = hidden[:, self.cell1]
        second = hidden[:, self.cell2]
        expanded_skill = hidden[:, N_CELLS].unsqueeze(1).expand_as(first)
        mask = legal_actions.bool() & self.in_bounds.unsqueeze(0)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every state must have at least one legal action")
        action_inputs = [first, second, expanded_skill]
        if self.config.use_immediate_features:
            features = self.immediate_action_features(board, goal_colour)
            action_inputs.append(
                self._standardize_action_features(features, mask)
            )
        logits = self.action_head(torch.cat(action_inputs, dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, self.config.mask_fill)
        return torch.log_softmax(logits, dim=-1)

    def negative_log_likelihood(
        self,
        action: torch.Tensor,
        **inputs: torch.Tensor,
    ) -> torch.Tensor:
        log_probabilities = self(**inputs)
        return -log_probabilities.gather(1, action.long().unsqueeze(1)).mean()


__all__ = ["ActionPolicyConfig", "ContinuousActionPolicy"]