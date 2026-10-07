"""Deterministic match-3 mechanics.

Nothing here samples. Every function is a pure function of the board plus a
``draw`` callback that supplies refill colours. That separation is deliberate:
it lets :mod:`match3.scm` put Pyro sample sites at exactly the points where the
process is genuinely random, and keep the cascade fixpoint deterministic.

A ``draw`` has the signature ``draw(n, tag) -> ndarray[int]`` and returns ``n``
colour indices. ``tag`` is a unique string used to name the sample site.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .spec import (
    EMPTY,
    HORIZONTAL_STRIPE,
    NO_SPECIAL,
    VERTICAL_STRIPE,
    Action,
    CascadeStep,
    LevelContext,
    State,
    Transition,
)

MAX_DEAL_ATTEMPTS = 200


class Draw(Protocol):
    def __call__(self, n: int, tag: str) -> np.ndarray: ...


# ---------------------------------------------------------------- matching ---


def match_mask(board: np.ndarray) -> np.ndarray:
    """Boolean mask of every cell in a horizontal or vertical run of >= 3.

    Runs are detected simultaneously, then cleared together. Walsh's NP-hardness
    proof instead deletes chains bottom-to-top, which Guala et al. show changes
    the game; we follow the simultaneous rule that real match-3 games use.
    """
    height, width = board.shape
    mask = np.zeros(board.shape, dtype=bool)

    for row in range(height):
        col = 0
        while col < width:
            value = board[row, col]
            if value == EMPTY:
                col += 1
                continue
            end = col
            while end + 1 < width and board[row, end + 1] == value:
                end += 1
            if end - col + 1 >= 3:
                mask[row, col : end + 1] = True
            col = end + 1

    for col in range(width):
        row = 0
        while row < height:
            value = board[row, col]
            if value == EMPTY:
                row += 1
                continue
            end = row
            while end + 1 < height and board[end + 1, col] == value:
                end += 1
            if end - row + 1 >= 3:
                mask[row : end + 1, col] = True
            row = end + 1

    return mask


def has_match(board: np.ndarray) -> bool:
    return bool(match_mask(board).any())


def _run_length(board: np.ndarray, row: int, col: int, drow: int, dcol: int) -> int:
    """Length of the run of equal colours through ``(row, col)`` along one axis."""
    height, width = board.shape
    value = board[row, col]
    if value == EMPTY:
        return 0
    total = 1
    for sign in (1, -1):
        r, c = row + sign * drow, col + sign * dcol
        while 0 <= r < height and 0 <= c < width and board[r, c] == value:
            total += 1
            r += sign * drow
            c += sign * dcol
    return total


def creates_match(board: np.ndarray, action: Action) -> bool:
    """Whether a swap produces a run of three, checked locally.

    A swap can only create a run through one of the two cells it moves, so the
    four lines through those cells are sufficient. Equivalent to scanning the
    whole board, and roughly an order of magnitude cheaper -- which matters
    because this is the inner loop of both the policy and dataset generation.
    """
    (r1, c1), (r2, c2) = action.cells
    swapped = apply_swap(board, action)
    for row, col in ((r1, c1), (r2, c2)):
        if _run_length(swapped, row, col, 0, 1) >= 3:
            return True
        if _run_length(swapped, row, col, 1, 0) >= 3:
            return True
    return False


def apply_swap(board: np.ndarray, action: Action) -> np.ndarray:
    """Return a copy of ``board`` with the action's two cells exchanged."""
    out = board.copy()
    (r1, c1), (r2, c2) = action.cells
    out[r1, c1], out[r2, c2] = board[r2, c2], board[r1, c1]
    return out


def _apply_swap_specials(specials: np.ndarray, action: Action) -> np.ndarray:
    out = specials.copy()
    (r1, c1), (r2, c2) = action.cells
    out[r1, c1], out[r2, c2] = specials[r2, c2], specials[r1, c1]
    return out


def legal_moves(board: np.ndarray) -> list[Action]:
    """Every swap that creates a match. Swaps that do not are simply illegal."""
    height, width = board.shape
    moves: list[Action] = []
    for row in range(height):
        for col in range(width):
            for drow, dcol in ((0, 1), (1, 0)):
                r2, c2 = row + drow, col + dcol
                if r2 >= height or c2 >= width:
                    continue
                if board[row, col] == board[r2, c2]:
                    continue
                action = Action(row, col, drow, dcol)
                if creates_match(board, action):
                    moves.append(action)
    return moves


def has_legal_move(board: np.ndarray) -> bool:
    """Cheap deadlock test: stop at the first legal swap."""
    height, width = board.shape
    for row in range(height):
        for col in range(width):
            for drow, dcol in ((0, 1), (1, 0)):
                r2, c2 = row + drow, col + dcol
                if r2 >= height or c2 >= width:
                    continue
                if board[row, col] == board[r2, c2]:
                    continue
                if creates_match(board, Action(row, col, drow, dcol)):
                    return True
    return False


def _run_cells(
    board: np.ndarray, row: int, col: int, drow: int, dcol: int
) -> list[tuple[int, int]]:
    height, width = board.shape
    value = board[row, col]
    cells = [(row, col)]
    for sign in (1, -1):
        r, c = row + sign * drow, col + sign * dcol
        while 0 <= r < height and 0 <= c < width and board[r, c] == value:
            cells.append((r, c))
            r += sign * drow
            c += sign * dcol
    return cells if len(cells) >= 3 else []


def local_match_cells(
    swapped: np.ndarray, cells: tuple[tuple[int, int], tuple[int, int]]
) -> set[tuple[int, int]]:
    """Cells cleared by the first round after a swap.

    Valid because the board is always settled before a move, so the only new
    runs are the ones passing through the two cells that moved.
    """
    matched: set[tuple[int, int]] = set()
    for row, col in cells:
        if swapped[row, col] == EMPTY:
            continue
        matched.update(_run_cells(swapped, row, col, 0, 1))
        matched.update(_run_cells(swapped, row, col, 1, 0))
    return matched


def _created_stripe(
    swapped: np.ndarray,
    action: Action,
    special_grid: np.ndarray | None = None,
) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for row, col in action.cells:
        if special_grid is not None and special_grid[row, col] != NO_SPECIAL:
            continue
        horizontal = _run_cells(swapped, row, col, 0, 1)
        vertical = _run_cells(swapped, row, col, 1, 0)
        if len(horizontal) == 4:
            candidates.append((len(horizontal), row, col, HORIZONTAL_STRIPE))
        if len(vertical) == 4:
            candidates.append((len(vertical), row, col, VERTICAL_STRIPE))
    if not candidates:
        return None
    _, row, col, kind = candidates[0]
    return row, col, kind


def _expand_striped_clear(
    mask: np.ndarray,
    specials: np.ndarray,
    *,
    preserve: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    expanded = mask.copy()
    if preserve is not None:
        expanded[preserve] = False
    activated: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while True:
        rows, cols = np.nonzero(expanded)
        pending = [
            (int(row), int(col))
            for row, col in zip(rows, cols)
            if (int(row), int(col)) not in seen
            and specials[row, col] != NO_SPECIAL
        ]
        if not pending:
            break
        for row, col in pending:
            seen.add((row, col))
            activated.append((row, col))
            if specials[row, col] == HORIZONTAL_STRIPE:
                expanded[row, :] = True
            elif specials[row, col] == VERTICAL_STRIPE:
                expanded[:, col] = True
        if preserve is not None:
            expanded[preserve] = False
    return expanded, activated


def immediate_effect(
    board: np.ndarray,
    action: Action,
    goal_colour: int,
    specials: np.ndarray | None = None,
) -> tuple[int, int]:
    """Tiles cleared and goal tiles cleared by the first round only.

    The policy scores moves with this rather than a full cascade rollout: a
    human sees the immediate match, not the knock-on effects.
    """
    swapped = apply_swap(board, action)
    matched = local_match_cells(swapped, action.cells)
    mask = np.zeros(swapped.shape, dtype=bool)
    if matched:
        rows, cols = zip(*matched)
        mask[rows, cols] = True
    special_grid = (
        np.full(board.shape, NO_SPECIAL, dtype=np.int8)
        if specials is None
        else _apply_swap_specials(specials, action)
    )
    creation = _created_stripe(swapped, action, special_grid)
    preserve = None if creation is None else creation[:2]
    cleared_mask, _ = _expand_striped_clear(
        mask, special_grid, preserve=preserve
    )
    cleared = int(cleared_mask.sum())
    goal = int((swapped[cleared_mask] == goal_colour).sum())
    return cleared, goal


# ------------------------------------------------------------ gravity/fill ---


def collapse(
    board: np.ndarray,
    draw: Draw,
    tag: str,
    specials: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    """Drop survivors to the bottom of each column and refill from the top.

    Mutates ``board``. Returns the fall mapping and the spawned tiles, both of
    which the renderer needs in order to animate rather than cut.
    """
    height, width = board.shape
    fall: list[tuple[int, int, int, int]] = []
    spawned: list[tuple[int, int, int]] = []

    for col in range(width):
        column = board[:, col]
        special_column = None if specials is None else specials[:, col]
        survivors = [
            (
                row,
                column[row],
                (
                    NO_SPECIAL
                    if special_column is None
                    else special_column[row]
                ),
            )
            for row in range(height)
            if column[row] != EMPTY
        ]
        rebuilt = np.full(height, EMPTY, dtype=board.dtype)
        rebuilt_specials = np.full(height, NO_SPECIAL, dtype=np.int8)

        offset = height - len(survivors)
        for index, (row, value, special) in enumerate(survivors):
            landing = offset + index
            rebuilt[landing] = value
            rebuilt_specials[landing] = special
            if landing != row:
                fall.append((row, col, landing, col))

        if offset:
            values = draw(offset, f"{tag}_c{col}")
            for row in range(offset):
                rebuilt[row] = values[row]
                spawned.append((row, col, int(values[row])))

        board[:, col] = rebuilt
        if specials is not None:
            specials[:, col] = rebuilt_specials

    return fall, spawned


def settle(
    board: np.ndarray,
    draw: Draw,
    goal_colour: int,
    tag: str,
    *,
    specials: np.ndarray | None = None,
    created_special: tuple[int, int, int] | None = None,
) -> list[CascadeStep]:
    """Run clear-fall-refill to a fixpoint. Mutates ``board``."""
    steps: list[CascadeStep] = []
    while True:
        mask = match_mask(board)
        if not mask.any():
            return steps

        special_grid = (
            np.full(board.shape, NO_SPECIAL, dtype=np.int8)
            if specials is None
            else specials
        )
        creation = created_special if not steps else None
        preserve = None if creation is None else creation[:2]
        if creation is not None:
            row, col, kind = creation
            special_grid[row, col] = kind
        specials_before = special_grid.copy()
        clear_mask, activated = _expand_striped_clear(
            mask, special_grid, preserve=preserve
        )
        rows, cols = np.nonzero(clear_mask)
        matched = [(int(r), int(c)) for r, c in zip(rows, cols)]
        board_before = board.copy()
        goal_cleared = int((board[clear_mask] == goal_colour).sum())

        board[clear_mask] = EMPTY
        special_grid[clear_mask] = NO_SPECIAL
        fall, spawned = collapse(
            board,
            draw,
            f"{tag}_s{len(steps)}",
            specials=special_grid,
        )

        steps.append(
            CascadeStep(
                matched=matched,
                board_before=board_before,
                board_after=board.copy(),
                fall=fall,
                spawned=spawned,
                goal_cleared=goal_cleared,
                specials_before=specials_before,
                specials_after=special_grid.copy(),
                created_specials=[] if creation is None else [creation],
                activated_specials=activated,
            )
        )


# ------------------------------------------------------------------ dealing --


def deal(level: LevelContext, draw: Draw, tag: str) -> np.ndarray:
    """Deal an opening board with no free matches and at least one legal move.

    Cells sitting in a match are redrawn rather than the whole board, so the
    rejection loop terminates quickly.
    """
    flat = draw(level.height * level.width, f"{tag}_init")
    board = np.asarray(flat, dtype=np.int8).reshape(level.height, level.width)

    for attempt in range(MAX_DEAL_ATTEMPTS):
        mask = match_mask(board)
        if mask.any():
            count = int(mask.sum())
            board[mask] = np.asarray(
                draw(count, f"{tag}_fix{attempt}"), dtype=np.int8
            )
            continue
        if legal_moves(board):
            return board
        board = np.asarray(
            draw(level.height * level.width, f"{tag}_re{attempt}"), dtype=np.int8
        ).reshape(level.height, level.width)

    return board


def reshuffle(
    board: np.ndarray,
    level: LevelContext,
    draw: Draw,
    tag: str,
    specials: np.ndarray | None = None,
) -> bool:
    """Redraw a deadlocked board in place. Returns whether a reshuffle happened.

    Reshuffles are unpopular with players and the industry treats two or more on
    a level as a design defect, so they are recorded on the transition.
    """
    if has_legal_move(board):
        return False
    board[:, :] = deal(level, draw, tag)
    if specials is not None:
        specials.fill(NO_SPECIAL)
    return True


# --------------------------------------------------------------- transition --


def resolve_move(
    state: State,
    action: Action,
    level: LevelContext,
    draw: Draw,
) -> tuple[State, Transition]:
    """Apply one action and settle the board: the ``S_t, A_t -> S_{t+1}`` map."""
    board_before = state.board.copy()
    board = apply_swap(state.board, action)
    board_swapped = board.copy()
    specials = _apply_swap_specials(state.specials, action)
    specials_swapped = specials.copy()
    created_special = _created_stripe(board, action, specials)

    steps = settle(
        board,
        draw,
        state.goal_colour,
        tag=f"t{state.t}",
        specials=specials,
        created_special=created_special,
    )
    transition = Transition(
        action=action,
        board_before=board_before,
        board_swapped=board_swapped,
        specials_swapped=specials_swapped,
        steps=steps,
    )
    transition.reshuffled = reshuffle(
        board,
        level,
        draw,
        tag=f"t{state.t}_shuffle",
        specials=specials,
    )

    nxt = State(
        board=board,
        moves_left=state.moves_left - 1,
        goals_left=max(0, state.goals_left - transition.goal_cleared),
        goal_colour=state.goal_colour,
        t=state.t + 1,
        specials=specials,
    )
    return nxt, transition


__all__ = [
    "apply_swap",
    "collapse",
    "creates_match",
    "deal",
    "has_legal_move",
    "has_match",
    "immediate_effect",
    "legal_moves",
    "match_mask",
    "reshuffle",
    "resolve_move",
    "settle",
]
