"""Per-cell cleared masks reconstructed from logged transitions by replaying the cascade on tile identities.

The swap moves two identities, every cascade step clears the identities at its matched cells, gravity compacts the survivors exactly as
``board.collapse`` does and refilled cells receive fresh identities. The mask marks the cells of the pre-move board whose tile was cleared
at any point of the cascade; goal tiles cleared among refilled tiles have no pre-move cell and are counted separately, so that

    goal_cleared == sum(mask[c] * (board_before[c] == goal_colour)) + refill_goal_cleared

holds exactly for every logged transition.
"""

from __future__ import annotations

import numpy as np

from match3_simulator.spec import Transition

REFILL_ID = 10_000


def cleared_mask_from_transition(transition: Transition, goal_colour: int) -> tuple[np.ndarray, int, int, int]:
    """Inputs: the Transition (action, board_before, cascade steps with matched / fall / spawned); the goal colour.
    Outputs: mask (64,) int8 over the pre-move board; goal tiles cleared from the pre-move board; goal tiles cleared among refilled tiles; refilled tiles cleared (any colour)."""
    board_before = np.asarray(transition.board_before)
    height, width = board_before.shape
    ids = np.arange(height * width).reshape(height, width)
    if transition.action is not None:
        (r1, c1), (r2, c2) = transition.action.cells
        ids[r1, c1], ids[r2, c2] = ids[r2, c2], ids[r1, c1]
    mask = np.zeros(height * width, dtype=np.int8)
    refill_goal = 0
    refill_total = 0
    next_refill = REFILL_ID
    for step in transition.steps:
        step_board = np.asarray(step.board_before)
        for row, col in step.matched:
            tile = int(ids[row, col])
            if tile < height * width:
                mask[tile] = 1
            else:
                refill_total += 1
                if int(step_board[row, col]) == int(goal_colour):
                    refill_goal += 1
            ids[row, col] = -1
        for col in range(width):  # gravity exactly as board.collapse: survivors keep order and land at the bottom, refills fill the top
            survivors = [ids[row, col] for row in range(height) if ids[row, col] != -1]
            offset = height - len(survivors)
            column = np.full(height, -1, dtype=np.int64)
            for index, tile in enumerate(survivors):
                column[offset + index] = tile
            for row in range(offset):
                column[row] = next_refill
                next_refill += 1
            ids[:, col] = column
    s_goal = int(((mask.reshape(height, width) == 1) & (board_before == int(goal_colour))).sum())
    return mask, s_goal, refill_goal, refill_total


def check_transition(transition: Transition, goal_colour: int) -> dict[str, int | bool]:
    """Reconstruct the mask and compare its derived counts with the logged ones."""
    mask, s_goal, refill_goal, refill_total = cleared_mask_from_transition(transition, goal_colour)
    return {"mask_sum": int(mask.sum()), "s_goal": s_goal, "refill_goal": refill_goal, "refill_total": refill_total,
            "goal_cleared_logged": int(transition.goal_cleared), "tiles_cleared_logged": int(transition.tiles_cleared),
            "goal_ok": s_goal + refill_goal == int(transition.goal_cleared), "tiles_ok": int(mask.sum()) + refill_total == int(transition.tiles_cleared)}


__all__ = ["REFILL_ID", "check_transition", "cleared_mask_from_transition"]
