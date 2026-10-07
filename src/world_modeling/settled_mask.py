"""Canonical settled-pair cleared mask (WM-1 §4 default target).

The release ships settled boards only — ``board_before`` / ``board_after`` with their special grids — and no cascade steps, so the
64-cell mask of pre-move tiles cleared by a move is *reconstructed* from the pair with a fixed, documented convention:

1. Swap (``board.apply_swap``), swap the specials, apply the created-stripe rule (``board._created_stripe``) — all deterministic.
2. Round 0 exactly as the engine: ``board.match_mask`` on the swapped board, then ``board._expand_striped_clear`` with the created
   stripe preserved. Stripes in that clear set are round-0 activations.
3. Deterministic cascade closure over *known* tiles: clear, gravity with refills marked unknown, repeat match -> stripe expansion ->
   clear -> gravity while a run of known tiles exists. (Vertical known runs are exact; horizontal known runs assume the columns
   fell as simulated — the source of the rare over-count.)
4. Per-column embedding of the remainder into the after board: the m refilled cells of a column are its top m; the bottom 8-m are
   survivors matched bottom-up, order-preserving, to the remaining known (colour, special) sequence, always matching the **lowest
   possible** old cell, so an ambiguous pair of equal tiles resolves to "topmost cleared". Unmatched old cells are cleared.
5. Assumption **A1** (activation inference): a stripe that neither cleared in steps 2-3 nor survived left the board by activation.
   Vertical -> its whole remaining column is cleared. Horizontal -> the row it cleared is unknown after gravity, so only the bound
   "every column lost >= 1 tile" is enforced (m >= 1) and step 4 is re-run. The stripe-count identity
   ``count(post-swap stripes) + created - count(after stripes) == activations`` is checked on every transition.
6. Reshuffled attempts (``episodes.csv.reshuffles > 0``) are flagged ``mask_valid = False`` by the data pipeline (board losses only).
7. ``residual_goal := logged goal delta - derived count`` (signed): goal tiles that arrive by refill and clear inside the same move
   have no pre-move cell and are invisible in a settled pair; coincidental colour matches under-count, the horizontal-alignment
   assumption of step 3 can over-count.

Mask indexing: ``mask[c]`` refers to the tile that sat at **pre-move** cell ``c``, so ``sum_c mask[c] * 1{board_before[c] == goal}`` counts
the goal tiles among cleared pre-move tiles. Exactness against full cascades is measured on a simulated cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from match3_simulator.board import _apply_swap_specials, _created_stripe, _expand_striped_clear, apply_swap, match_mask
from match3_simulator.learned_model.tokens import BOARD_HEIGHT, BOARD_WIDTH, N_CELLS, index_to_action
from match3_simulator.spec import EMPTY, HORIZONTAL_STRIPE, NO_SPECIAL, VERTICAL_STRIPE

UNKNOWN = -1  # identity of a refilled (unknown) tile


@dataclass
class SettledMask:
    """Result of the reconstruction for one transition."""

    mask: np.ndarray  # (64,) int8 over pre-move cells
    m_per_column: np.ndarray  # (8,) int refilled cells per column
    created: tuple[int, int, int] | None  # (row, col, kind) in post-swap coordinates
    activations_round0: list[int]  # pre-move tile ids
    activations_closure: list[int]
    activations_inferred: list[int]
    ambiguous_columns: list[int] = field(default_factory=list)
    consistent: bool = True  # stripe-count identity and survivor-stripe matching hold
    notes: list[str] = field(default_factory=list)

    @property
    def n_activations(self) -> int:
        return len(self.activations_round0) + len(self.activations_closure) + len(self.activations_inferred)

    @property
    def stripe_present(self) -> bool:
        return self.created is not None or self.n_activations > 0 or bool(self.notes and self.notes[0] == "stripes")

    def derived_goal(self, board_before: np.ndarray, goal_colour: int) -> int:
        """Goal tiles among cleared pre-move tiles."""
        return int(((self.mask == 1) & (np.asarray(board_before).reshape(-1) == int(goal_colour))).sum())


def _gravity(colours: np.ndarray, specials: np.ndarray, ids: np.ndarray) -> None:
    """Drop known tiles to the bottom of every column and mark the refilled cells unknown (in place; mirrors ``board.collapse``)."""
    for col in range(BOARD_WIDTH):
        keep = colours[:, col] != EMPTY
        survivors = [(colours[r, col], specials[r, col], ids[r, col]) for r in range(BOARD_HEIGHT) if keep[r]]
        offset = BOARD_HEIGHT - len(survivors)
        colours[:offset, col] = EMPTY
        specials[:offset, col] = NO_SPECIAL
        ids[:offset, col] = UNKNOWN
        for index, (colour, special, tile) in enumerate(survivors):
            colours[offset + index, col] = colour
            specials[offset + index, col] = special
            ids[offset + index, col] = tile


def _match_column(old: list[tuple[int, int, int]], new_colours: np.ndarray, new_specials: np.ndarray, forced_refills: int) -> tuple[list[int], list[int], int, bool, bool]:
    """Greedy bottom-up, order-preserving matching of the after column onto the remaining known tiles of a column.

    Inputs: old known tiles bottom-up as (colour, special, id); after column colours / specials (8,) top-down; forced minimum refills.
    Outputs: matched ids, cleared ids, m (refills), ambiguous flag, consistent flag (every stripe in the after column was matched).
    """
    matched: list[int] = []
    cleared: list[int] = []
    i = 0
    for r in range(BOARD_HEIGHT - 1, forced_refills - 1, -1):  # bottom-up survivors candidates
        colour, special = int(new_colours[r]), int(new_specials[r])
        while i < len(old) and (old[i][0], old[i][1]) != (colour, special):
            cleared.append(old[i][2])
            i += 1
        if i < len(old):
            matched.append(old[i][2])
            i += 1
        else:
            break
    cleared.extend(tile for _, _, tile in old[i:])
    m = BOARD_HEIGHT - len(matched)
    consistent = all(int(new_specials[r]) == NO_SPECIAL for r in range(m))  # refills carry no special
    matched_keys = {(c, s) for c, s, t in old if t in set(matched)}
    ambiguous = any((c, s) in matched_keys for c, s, t in old if t in set(cleared))
    return matched, cleared, m, ambiguous, consistent


def settled_mask(board_before: np.ndarray, specials_before: np.ndarray, action_index: int, board_after: np.ndarray, specials_after: np.ndarray) -> SettledMask:
    """Reconstruct the cleared mask of one logged move from its settled before / after pair (see module docstring).

    Inputs: board_before, specials_before, board_after, specials_after as (8, 8) or (64,) arrays; action index in [0, 128).
    Outputs: SettledMask.
    """
    before = np.asarray(board_before, dtype=np.int64).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    after = np.asarray(board_after, dtype=np.int64).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    sp_before = np.asarray(specials_before, dtype=np.int64).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    sp_after = np.asarray(specials_after, dtype=np.int64).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    action = index_to_action(int(action_index))
    (r1, c1), (r2, c2) = action.cells
    # 1. swap + created stripe
    colours = apply_swap(before, action).astype(np.int64)
    specials = _apply_swap_specials(sp_before, action).astype(np.int64)
    ids = np.arange(N_CELLS).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    ids[r1, c1], ids[r2, c2] = ids[r2, c2], ids[r1, c1]
    created = _created_stripe(colours, action, specials)
    stripes_total = int((specials != NO_SPECIAL).sum())
    if created is not None:
        specials[created[0], created[1]] = created[2]
    # every stripe that could be on the board during the move: pre-move tile id -> (kind, column)
    sources = {int(ids[r, c]): (int(specials[r, c]), int(c)) for r, c in zip(*np.nonzero(specials != NO_SPECIAL))}
    mask = np.zeros(N_CELLS, dtype=np.int8)
    notes: list[str] = ["stripes"] if sources else []
    # 2. round 0 exactly as the engine
    runs = match_mask(colours)
    cleared, activated0 = _expand_striped_clear(runs, specials, preserve=None if created is None else created[:2])
    activated_round0 = [int(ids[r, c]) for r, c in activated0]
    for tile in ids[cleared]:
        mask[int(tile)] = 1
    colours[cleared] = EMPTY
    specials[cleared] = NO_SPECIAL
    ids[cleared] = UNKNOWN
    _gravity(colours, specials, ids)
    # 3. deterministic closure over known tiles
    activated_closure: list[int] = []
    for _ in range(N_CELLS):
        runs = match_mask(colours)
        if not runs.any():
            break
        cleared, activated = _expand_striped_clear(runs, specials)
        activated_closure.extend(int(ids[r, c]) for r, c in activated)
        for tile in ids[cleared]:
            if tile != UNKNOWN:
                mask[int(tile)] = 1
        colours[cleared] = EMPTY
        specials[cleared] = NO_SPECIAL
        ids[cleared] = UNKNOWN
        _gravity(colours, specials, ids)
    # 4. per-column embedding of the remainder
    old_columns = [[(int(colours[r, col]), int(specials[r, col]), int(ids[r, col])) for r in range(BOARD_HEIGHT - 1, -1, -1) if ids[r, col] != UNKNOWN]
                   for col in range(BOARD_WIDTH)]
    forced = np.zeros(BOARD_WIDTH, dtype=np.int64)
    full_clear = np.zeros(BOARD_WIDTH, dtype=bool)
    m_per_column = np.zeros(BOARD_WIDTH, dtype=np.int64)

    def embed() -> tuple[set[int], list[int], bool]:
        matched_all: set[int] = set()
        ambiguous_cols: list[int] = []
        ok = True
        for col in range(BOARD_WIDTH):
            if full_clear[col]:
                m_per_column[col] = BOARD_HEIGHT
                for _, _, tile in old_columns[col]:
                    mask[tile] = 1
                ok &= not bool((sp_after[:, col] != NO_SPECIAL).any())
                continue
            matched, cleared_ids, m, ambiguous, column_ok = _match_column(old_columns[col], after[:, col], sp_after[:, col], int(forced[col]))
            m_per_column[col] = m
            matched_all.update(matched)
            for tile in cleared_ids:
                mask[tile] = 1
            if ambiguous:
                ambiguous_cols.append(col)
            ok &= column_ok
        return matched_all, ambiguous_cols, ok

    consistent = True
    matched_all, ambiguous_cols, ok = embed()
    consistent &= ok
    # 5. A1: stripes that neither cleared (steps 2-3) nor survived were activated
    known_activated = set(activated_round0) | set(activated_closure)
    inferred: list[int] = []
    for _ in range(8):
        changed = False
        for tile, (kind, col) in sources.items():
            if tile in known_activated or tile in matched_all or tile in inferred:
                continue
            inferred.append(tile)
            mask[tile] = 1
            if kind == VERTICAL_STRIPE and not full_clear[col]:
                full_clear[col] = True
                changed = True
            elif kind == HORIZONTAL_STRIPE:
                for cc in range(BOARD_WIDTH):
                    if not full_clear[cc] and forced[cc] < 1:
                        forced[cc] = 1
                        changed = True
        if not changed:
            break
        matched_all, ambiguous_cols, ok = embed()
        consistent &= ok
    created_count = 0 if created is None else 1
    after_count = int((sp_after != NO_SPECIAL).sum())
    n_activations = len(activated_round0) + len(activated_closure) + len(inferred)
    if stripes_total + created_count - after_count != n_activations:
        consistent = False
        notes.append(f"stripe identity violated: {stripes_total}+{created_count}-{after_count} != {n_activations}")
    return SettledMask(mask=mask, m_per_column=m_per_column, created=created, activations_round0=activated_round0, activations_closure=activated_closure,
                       activations_inferred=inferred, ambiguous_columns=sorted(set(ambiguous_cols)), consistent=consistent, notes=notes)


__all__ = ["SettledMask", "UNKNOWN", "settled_mask"]
