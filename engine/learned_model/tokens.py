"""Action indexing and exact legal masks for learned policies."""

from __future__ import annotations

import numpy as np

from ..board import legal_moves
from ..spec import Action

BOARD_HEIGHT = 8
BOARD_WIDTH = 8
N_CELLS = BOARD_HEIGHT * BOARD_WIDTH
ACTION_SLOTS = N_CELLS * 2


def action_index(row: int, col: int, drow: int) -> int:
    """Map a right/down swap to its fixed vocabulary slot."""
    return (row * BOARD_WIDTH + col) * 2 + drow


def action_to_index(action: Action) -> int:
    if (action.drow, action.dcol) not in ((0, 1), (1, 0)):
        raise ValueError(f"unsupported swap direction {action}")
    return action_index(action.row, action.col, action.drow)


def index_to_action(index: int) -> Action:
    if not 0 <= index < ACTION_SLOTS:
        raise ValueError(f"action index {index} outside [0, {ACTION_SLOTS})")
    cell, drow = divmod(int(index), 2)
    row, col = divmod(cell, BOARD_WIDTH)
    return Action(row, col, drow, 1 - drow)


def _slot_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_bounds = np.zeros(ACTION_SLOTS, dtype=bool)
    first = np.zeros(ACTION_SLOTS, dtype=np.int64)
    second = np.zeros(ACTION_SLOTS, dtype=np.int64)
    for index in range(ACTION_SLOTS):
        action = index_to_action(index)
        next_row = action.row + action.drow
        next_col = action.col + action.dcol
        first[index] = action.row * BOARD_WIDTH + action.col
        if 0 <= next_row < BOARD_HEIGHT and 0 <= next_col < BOARD_WIDTH:
            in_bounds[index] = True
            second[index] = next_row * BOARD_WIDTH + next_col
        else:
            second[index] = first[index]
    return in_bounds, first, second


IN_BOUNDS, CELL1_TOKEN, CELL2_TOKEN = _slot_tables()


def legal_mask(board: np.ndarray) -> np.ndarray:
    """Return a 128-vector whose true entries are exactly legal swaps."""
    board_array = np.asarray(board)
    if board_array.shape != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError("learned action policy requires an 8x8 board")
    mask = np.zeros(ACTION_SLOTS, dtype=bool)
    for action in legal_moves(board_array):
        mask[action_to_index(action)] = True
    return mask


__all__ = [
    "ACTION_SLOTS",
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "CELL1_TOKEN",
    "CELL2_TOKEN",
    "IN_BOUNDS",
    "N_CELLS",
    "action_index",
    "action_to_index",
    "index_to_action",
    "legal_mask",
]