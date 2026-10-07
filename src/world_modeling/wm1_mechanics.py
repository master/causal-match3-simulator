from __future__ import annotations

import torch

from match3_simulator.learned_model.tokens import BOARD_HEIGHT, BOARD_WIDTH, CELL1_TOKEN, CELL2_TOKEN, N_CELLS
from match3_simulator.spec import HORIZONTAL_STRIPE, NO_SPECIAL, VERTICAL_STRIPE
from match3_simulator.world_modeling.decoder import _run_mask

N_SPECIAL_KINDS = 3


def _cells(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs: actions (N,). Outputs: the two flat cell indices (N,), (N,) of each swap."""
    device = actions.device
    cell1 = torch.as_tensor(CELL1_TOKEN, device=device)[actions.long()]
    cell2 = torch.as_tensor(CELL2_TOKEN, device=device)[actions.long()]
    return cell1, cell2


def swap_grid(values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Exchange the two cells of each action in a flat (N, 64) grid (colours or specials).

    Inputs: values (N, 64); actions (N,). Outputs: swapped copy (N, 64).
    """
    cell1, cell2 = _cells(actions)
    out = values.clone()
    index = torch.arange(values.shape[0], device=values.device)
    out[index, cell1] = values[index, cell2]
    out[index, cell2] = values[index, cell1]
    return out


def run_lengths(grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Length of the maximal horizontal and vertical run of equal colours through every cell.

    Inputs: grid (N, 8, 8). Outputs: horizontal lengths (N, 8, 8), vertical lengths (N, 8, 8) (each >= 1).
    """
    def along(g: torch.Tensor) -> torch.Tensor:  # runs along the last axis
        n, rows, cols = g.shape
        left = torch.zeros_like(g, dtype=torch.long)
        right = torch.zeros_like(g, dtype=torch.long)
        same_left = torch.ones_like(g, dtype=torch.bool)
        same_right = torch.ones_like(g, dtype=torch.bool)
        for d in range(1, cols):
            step_left = torch.zeros_like(g, dtype=torch.bool)
            step_left[..., d:] = same_left[..., d:] & (g[..., d:] == g[..., :-d])
            same_left = step_left
            left += step_left.long()
            step_right = torch.zeros_like(g, dtype=torch.bool)
            step_right[..., :-d] = same_right[..., :-d] & (g[..., :-d] == g[..., d:])
            same_right = step_right
            right += step_right.long()
        return left + right + 1

    horizontal = along(grid)
    vertical = along(grid.transpose(1, 2)).transpose(1, 2)
    return horizontal, vertical


def created_stripes(swapped: torch.Tensor, specials_swapped: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The engine's created-stripe rule (``board._created_stripe``): the first of (cell1 horizontal, cell1 vertical, cell2 horizontal,
    cell2 vertical) whose run through the moved cell has length exactly 4 creates a stripe of that orientation at the moved cell,
    unless the moved cell already carries a special.

    Inputs: swapped boards (N, 64); post-swap specials (N, 64); actions (N,).
    Outputs: cell index (N,) long (-1 when nothing is created) and kind (N,) long (0 when nothing is created).
    """
    n = swapped.shape[0]
    grid = swapped.view(n, BOARD_HEIGHT, BOARD_WIDTH)
    horizontal, vertical = run_lengths(grid)
    horizontal = horizontal.reshape(n, N_CELLS)
    vertical = vertical.reshape(n, N_CELLS)
    cell1, cell2 = _cells(actions)
    index = torch.arange(n, device=swapped.device)
    out_cell = torch.full((n,), -1, dtype=torch.long, device=swapped.device)
    out_kind = torch.zeros(n, dtype=torch.long, device=swapped.device)
    for cell in (cell1, cell2):
        free = specials_swapped[index, cell] == NO_SPECIAL
        for lengths, kind in ((horizontal, HORIZONTAL_STRIPE), (vertical, VERTICAL_STRIPE)):
            hit = free & (lengths[index, cell] == 4) & (out_cell < 0)
            out_cell = torch.where(hit, cell, out_cell)
            out_kind = torch.where(hit, torch.full_like(out_kind, kind), out_kind)
    return out_cell, out_kind


def expand_striped_clear(cleared: torch.Tensor, specials: torch.Tensor, preserve: torch.Tensor | None = None, max_rounds: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain stripe activations inside one cascade round (``board._expand_striped_clear``): every stripe inside the clear set clears its
    full row (horizontal) or column (vertical), repeatedly, never clearing the preserved (created) cell.

    Inputs: cleared (N, 64) bool; specials (N, 64); optional preserve cell index (N,) long (-1 = none).
    Outputs: expanded clear set (N, 64) bool and activated stripes (N, 64) bool.
    """
    n = cleared.shape[0]
    index = torch.arange(n, device=cleared.device)
    keep = torch.ones_like(cleared)
    if preserve is not None:
        has = preserve >= 0
        keep[index[has], preserve[has]] = False
    expanded = cleared & keep
    activated = torch.zeros_like(cleared)
    for _ in range(max_rounds):
        pending = expanded & (specials != NO_SPECIAL) & ~activated
        if not bool(pending.any()):
            break
        activated |= pending
        grid = pending.view(n, BOARD_HEIGHT, BOARD_WIDTH)
        kinds = specials.view(n, BOARD_HEIGHT, BOARD_WIDTH)
        rows_hit = (grid & (kinds == HORIZONTAL_STRIPE)).any(dim=2, keepdim=True).expand(-1, -1, BOARD_WIDTH)
        cols_hit = (grid & (kinds == VERTICAL_STRIPE)).any(dim=1, keepdim=True).expand(-1, BOARD_HEIGHT, -1)
        expanded = (expanded | rows_hit.reshape(n, N_CELLS) | cols_hit.reshape(n, N_CELLS)) & keep
    return expanded, activated


def round0(boards: torch.Tensor, specials: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """The known part of a move: swap, created stripe, round-0 clear with chained activations, and the stripe sources of the next board.

    Inputs: pre-move boards (N, 64) long; pre-move specials (N, 64) long; actions (N,) long.
    Outputs: dict with
      swapped (N, 64) colours after the swap; specials_swapped (N, 64) specials after the swap **including the created stripe**;
      created_cell (N,) long (-1 none), created_kind (N,) long;
      cleared_swapped (N, 64) bool round-0 clear set in post-swap positions; cleared_pre (N, 64) bool the same set in **pre-move** cell indexing;
      activated (N, 64) bool stripes activated in round 0 (post-swap positions);
      sources (N, 64) long specials that may still be on the board after the move (post-swap grid with created stripe, round-0 clears removed);
      n_cleared (N,) long.
    """
    if boards.shape != specials.shape or boards.ndim != 2 or boards.shape[1] != N_CELLS:
        raise ValueError("boards and specials must have shape (N, 64)")
    n = boards.shape[0]
    index = torch.arange(n, device=boards.device)
    swapped = swap_grid(boards.long(), actions)
    specials_swapped = swap_grid(specials.long(), actions)
    created_cell, created_kind = created_stripes(swapped, specials_swapped, actions)
    has = created_cell >= 0
    specials_swapped[index[has], created_cell[has]] = created_kind[has]
    runs = _run_mask(swapped.view(n, BOARD_HEIGHT, BOARD_WIDTH)).reshape(n, N_CELLS)
    cleared, activated = expand_striped_clear(runs, specials_swapped, preserve=created_cell)
    sources = specials_swapped.masked_fill(cleared, NO_SPECIAL)
    return {
        "swapped": swapped, "specials_swapped": specials_swapped, "created_cell": created_cell, "created_kind": created_kind,
        "cleared_swapped": cleared, "cleared_pre": swap_grid(cleared, actions), "activated": activated, "sources": sources, "n_cleared": cleared.long().sum(dim=1),
    }


def admissible_special_kinds(sources: torch.Tensor) -> torch.Tensor:
    """Per-cell admissible special kinds of the next board: kind k is possible at (r, c) only if a source stripe of kind k sits in column c
    at a row <= r (gravity only moves tiles down their own column); 'none' is always admissible.

    Inputs: sources (N, 64) long. Outputs: bool (N, 64, 3).
    """
    n = sources.shape[0]
    grid = sources.view(n, BOARD_HEIGHT, BOARD_WIDTH)
    allowed = torch.zeros((n, BOARD_HEIGHT, BOARD_WIDTH, N_SPECIAL_KINDS), dtype=torch.bool, device=sources.device)
    allowed[..., NO_SPECIAL] = True
    for kind in (HORIZONTAL_STRIPE, VERTICAL_STRIPE):
        allowed[..., kind] = torch.cummax((grid == kind).long(), dim=1).values.bool()
    return allowed.view(n, N_CELLS, N_SPECIAL_KINDS)


def sample_specials(logits: torch.Tensor, sources: torch.Tensor, *, stochastic: bool, generator: torch.Generator | None) -> torch.Tensor:
    """Decode next-board special kinds column by column, bottom-up, under the order-preserving source constraint: a decoded stripe at
    row r consumes the lowest unconsumed source of its kind at a row <= r and above the last consumed source, so stripes never change
    kind or column, never multiply and never overtake each other.

    Inputs: logits (N, 64, 3); sources (N, 64) long; stochastic flag; generator.
    Outputs: specials (N, 64) long.
    """
    n = logits.shape[0]
    device = logits.device
    grid = sources.view(n, BOARD_HEIGHT, BOARD_WIDTH)
    out = torch.zeros((n, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.long, device=device)
    ceiling = torch.full((n, BOARD_WIDTH), BOARD_HEIGHT, dtype=torch.long, device=device)  # rows < ceiling are still available
    rows = torch.arange(BOARD_HEIGHT, device=device).view(1, -1, 1)
    logits = logits.view(n, BOARD_HEIGHT, BOARD_WIDTH, N_SPECIAL_KINDS)
    for r in range(BOARD_HEIGHT - 1, -1, -1):
        available = (rows <= r) & (rows < ceiling.unsqueeze(1))  # (N, 8 rows, 8 cols)
        allowed = torch.zeros((n, BOARD_WIDTH, N_SPECIAL_KINDS), dtype=torch.bool, device=device)
        allowed[..., NO_SPECIAL] = True
        for kind in (HORIZONTAL_STRIPE, VERTICAL_STRIPE):
            allowed[..., kind] = ((grid == kind) & available).any(dim=1)
        cell_logits = logits[:, r].masked_fill(~allowed, float("-inf"))
        if stochastic:
            kinds = torch.multinomial(cell_logits.reshape(-1, N_SPECIAL_KINDS).softmax(dim=-1), 1, generator=generator).view(n, BOARD_WIDTH)
        else:
            kinds = cell_logits.argmax(dim=-1)
        out[:, r] = kinds
        picked = kinds > 0
        if bool(picked.any()):
            match = (grid == kinds.unsqueeze(1)) & available & picked.unsqueeze(1)  # (N, 8 rows, 8 cols)
            candidate_rows = torch.where(match, rows.expand(n, -1, BOARD_WIDTH), torch.full_like(match, -1, dtype=torch.long))
            consumed = candidate_rows.max(dim=1).values  # lowest (largest row) admissible source
            ceiling = torch.where(picked, consumed, ceiling)
    return out.view(n, N_CELLS)


__all__ = ["N_SPECIAL_KINDS", "admissible_special_kinds", "created_stripes", "expand_striped_clear", "round0", "run_lengths", "sample_specials", "swap_grid"]
