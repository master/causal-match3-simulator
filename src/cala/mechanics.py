from __future__ import annotations

import torch

from match3_simulator.learned_model.tokens import ACTION_SLOTS, N_CELLS
from match3_simulator.world_modeling.wm1_mechanics import round0

N_SPECIAL_FEATURES = 4


@torch.no_grad()
def special_aware_action_features(board: torch.Tensor, specials: torch.Tensor, goal_colour: torch.Tensor, *, chunk: int = 256) -> torch.Tensor:
    """Inputs: board (M, 64) long; specials (M, 64) long in {0, 1, 2}; goal_colour (M,) long. Outputs: (M, 128, 4) float32."""
    if board.ndim != 2 or board.shape[1] != N_CELLS or specials.shape != board.shape or goal_colour.shape != (board.shape[0],):
        raise ValueError("board / specials must be (M, 64) and goal_colour (M,)")
    m = board.shape[0]
    device = board.device
    actions = torch.arange(ACTION_SLOTS, device=device)
    out = torch.empty((m, ACTION_SLOTS, N_SPECIAL_FEATURES), dtype=torch.float32, device=device)
    for start in range(0, m, chunk):
        b = board[start : start + chunk].long()
        s = specials[start : start + chunk].long()
        g = goal_colour[start : start + chunk].long()
        k = b.shape[0]
        rep_b = b.unsqueeze(1).expand(-1, ACTION_SLOTS, -1).reshape(k * ACTION_SLOTS, N_CELLS)
        rep_s = s.unsqueeze(1).expand(-1, ACTION_SLOTS, -1).reshape(k * ACTION_SLOTS, N_CELLS)
        rep_a = actions.unsqueeze(0).expand(k, -1).reshape(-1)
        r0 = round0(rep_b, rep_s, rep_a)
        cleared = r0["cleared_swapped"]
        goal_hit = cleared & (r0["swapped"] == g.repeat_interleave(ACTION_SLOTS).unsqueeze(1))
        feats = torch.stack((cleared.sum(dim=1).float(), goal_hit.sum(dim=1).float(), (r0["created_cell"] >= 0).float(), r0["activated"].sum(dim=1).float()), dim=-1)
        out[start : start + k] = feats.view(k, ACTION_SLOTS, N_SPECIAL_FEATURES)
    return out


__all__ = ["N_SPECIAL_FEATURES", "special_aware_action_features"]
