from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from match3_simulator.learned_model.action_policy import ActionPolicyConfig, ContinuousActionPolicy
from match3_simulator.learned_model.tokens import N_CELLS
from match3_simulator.world_modeling.wm1_mechanics import N_SPECIAL_KINDS
from match3_simulator.experiments.mechanics import N_SPECIAL_FEATURES, special_aware_action_features


@dataclass(frozen=True)
class SpecialAwarePolicyConfig(ActionPolicyConfig):
    use_specials: bool = False
    special_features: bool = False


class SpecialAwareActionPolicy(ContinuousActionPolicy):
    """See module docstring. ``forward(..., specials=None)``; specials default to none (frozen-policy behaviour)."""

    def __init__(self, config: SpecialAwarePolicyConfig = SpecialAwarePolicyConfig()):
        super().__init__(config)
        width = config.d_model
        self.special: nn.Embedding | None = nn.Embedding(N_SPECIAL_KINDS, width) if config.use_specials else None
        if config.special_features:
            if not config.use_immediate_features:
                raise ValueError("special_features needs use_immediate_features")
            self.action_head = nn.Sequential(nn.Linear(3 * width + N_SPECIAL_FEATURES, width), nn.GELU(), nn.Linear(width, 1))

    @property
    def is_frozen_equivalent(self) -> bool:
        return not (self.config.use_specials or self.config.special_features)

    def forward(self, board: torch.Tensor, goal_colour: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor, skill: torch.Tensor, legal_actions: torch.Tensor,
                specials: torch.Tensor | None = None) -> torch.Tensor:
        if board.ndim != 2 or board.shape[1] != N_CELLS:
            raise ValueError("board must have shape (batch, 64)")
        batch_size = board.shape[0]
        if skill.shape != (batch_size, self.config.skill_dimensions):
            raise ValueError("skill must have shape (batch, skill_dimensions)")
        if legal_actions.shape != (batch_size, int(self.in_bounds.shape[0])):
            raise ValueError("legal_actions must have shape (batch, 128)")
        if specials is not None and specials.shape != board.shape:
            raise ValueError("specials must align with board")
        tiles = self.colour(board) + self.row(self.cell_rows).unsqueeze(0) + self.col(self.cell_cols).unsqueeze(0) + self.is_goal((board == goal_colour.unsqueeze(1)).long())
        if self.special is not None:
            tiles = tiles + self.special(torch.zeros_like(board) if specials is None else specials.long())
        skill_token = self.skill(skill).unsqueeze(1)
        move_token = self.moves(moves_left.clamp(0, self.config.max_moves_left)).unsqueeze(1)
        goal_token = self.goals((goals_left.to(tiles.dtype) / 32.0).unsqueeze(1)).unsqueeze(1)
        hidden = self.trunk(torch.cat((tiles, skill_token, move_token, goal_token), dim=1))
        first = hidden[:, self.cell1]
        second = hidden[:, self.cell2]
        expanded_skill = hidden[:, N_CELLS].unsqueeze(1).expand_as(first)
        mask = legal_actions.bool() & self.in_bounds.unsqueeze(0)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every state must have at least one legal action")
        action_inputs = [first, second, expanded_skill]
        if self.config.use_immediate_features:
            if self.config.special_features:
                features = special_aware_action_features(board, torch.zeros_like(board) if specials is None else specials, goal_colour)
            else:
                features = self.immediate_action_features(board, goal_colour)
            action_inputs.append(self._standardize_action_features(features, mask))
        logits = self.action_head(torch.cat(action_inputs, dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, self.config.mask_fill)
        return torch.log_softmax(logits, dim=-1)


__all__ = ["SpecialAwareActionPolicy", "SpecialAwarePolicyConfig"]
