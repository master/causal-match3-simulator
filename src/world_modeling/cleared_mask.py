"""Symmetry augmentations for kernel training and the Poisson-binomial goal count.

The cleared-mask reconstruction itself lives in ``learned_model.cleared`` (it is needed when the transition dataset is built); this module
re-exports it and adds the horizontal-mirror and colour-permutation augmentations of padded transition batches plus the exact distribution of
the derived goal count.
"""

from __future__ import annotations

import numpy as np
import torch

from match3_simulator.world_modeling.cleared import check_transition, cleared_mask_from_transition  # noqa: F401
from match3_simulator.learned_model.tokens import ACTION_SLOTS, BOARD_HEIGHT, BOARD_WIDTH, N_CELLS, action_to_index, index_to_action
from match3_simulator.scm import LEVELS
from match3_simulator.spec import Action


# ------------------------------------------------------------- augmentation ----


def _mirror_action_table() -> np.ndarray:
    table = np.zeros(ACTION_SLOTS, dtype=np.int64)
    for index in range(ACTION_SLOTS):
        action = index_to_action(index)
        if action.drow == 0:  # right swap (r, c)-(r, c+1) -> (r, W-2-c)-(r, W-1-c)
            col = BOARD_WIDTH - 2 - action.col
            if col < 0:
                table[index] = index  # out-of-bounds slot; never legal
                continue
            table[index] = action_to_index(Action(action.row, col, 0, 1))
        else:  # down swap (r, c)-(r+1, c) -> (r, W-1-c)
            table[index] = action_to_index(Action(action.row, BOARD_WIDTH - 1 - action.col, 1, 0))
    return table


MIRROR_ACTION = torch.as_tensor(_mirror_action_table())
COLOUR_PERMUTATION_LEVELS: tuple[str, ...] = ("orchard", "foundry")  # uniform spawn weights; harbour's palette is not exchangeable


def mirror_episodes(batch: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Horizontally mirror the selected episodes of a padded transition batch (boards, next boards, cleared masks, specials and actions).

    Inputs: padded batch (B, T, ...) as built by GameplayTransitionDataset.batch; boolean row selector (B,).
    Outputs: a new dict; unselected episodes are untouched.
    """
    out = dict(batch)
    if not bool(rows.any()):
        return out
    for key in ("boards", "next_boards", "cleared_masks", "specials", "next_specials"):
        if key in out:
            grid = out[key].clone()
            grid[rows] = grid[rows].reshape(int(rows.sum()), grid.shape[1], BOARD_HEIGHT, BOARD_WIDTH).flip(-1).reshape(int(rows.sum()), grid.shape[1], N_CELLS)
            out[key] = grid
    actions = out["actions"].clone()
    actions[rows] = MIRROR_ACTION.to(actions.device)[actions[rows]]
    out["actions"] = actions
    return out


def permute_colours(batch: dict[str, torch.Tensor], rows: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Apply an independent random permutation of the level's palette to each selected episode (boards, next boards and the goal colour together).

    Inputs: padded batch; boolean row selector (B,) — only episodes whose level allows exchangeable colours should be selected; optional generator.
    Outputs: a new dict.
    """
    out = dict(batch)
    boards = out["boards"].clone()
    next_boards = out["next_boards"].clone()
    goal_colours = out["goal_colours"].clone()
    levels = out["levels"]
    for b in torch.nonzero(rows).flatten().tolist():
        n_colours = LEVELS[int(levels[b])].n_colours
        perm = torch.randperm(n_colours, generator=generator, device="cpu").to(boards.device)
        full = torch.arange(max(int(boards.max().item()) + 1, n_colours), device=boards.device)
        full[:n_colours] = perm
        boards[b] = full[boards[b].long()]
        next_boards[b] = full[next_boards[b].long()]
        goal_colours[b] = full[goal_colours[b].long()]
    out["boards"], out["next_boards"], out["goal_colours"] = boards, next_boards, goal_colours
    return out


def augment_batch(batch: dict[str, torch.Tensor], *, mirror_probability: float, colour_probability: float, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Kernel-dataloader augmentation: horizontal mirror (any level) and colour permutation (exchangeable-palette levels only), each per episode with the given probability."""
    if mirror_probability <= 0 and colour_probability <= 0:
        return batch
    n = batch["boards"].shape[0]
    draws = torch.rand((2, n), generator=generator)
    out = batch
    if mirror_probability > 0:
        out = mirror_episodes(out, (draws[0] < mirror_probability).to(batch["boards"].device))
    if colour_probability > 0:
        allowed = torch.as_tensor([LEVELS[int(l)].name in COLOUR_PERMUTATION_LEVELS for l in batch["levels"].tolist()], device=batch["boards"].device)
        out = permute_colours(out, (draws[1] < colour_probability).to(batch["boards"].device) & allowed, generator=generator)
    return out


def poisson_binomial_pmf(probabilities: torch.Tensor, keep: torch.Tensor, max_count: int | None = None) -> torch.Tensor:
    """Exact distribution of the number of successes among independent Bernoulli cells (probabilities zeroed where keep is False).

    Inputs: probabilities (rows, 64) in [0, 1]; keep (rows, 64) bool; optional truncation of the count axis.
    Outputs: pmf (rows, K + 1) with K = 64 or max_count (tail mass folded into the last bin).
    """
    q = probabilities * keep.to(probabilities.dtype)
    rows, cells = q.shape
    size = cells + 1
    pmf = q.new_zeros(rows, size)
    pmf[:, 0] = 1.0
    for c in range(cells):
        qc = q[:, c : c + 1]
        shifted = torch.cat((pmf.new_zeros(rows, 1), pmf[:, :-1]), dim=1)
        pmf = pmf * (1.0 - qc) + shifted * qc
    if max_count is not None and max_count + 1 < size:
        head = pmf[:, :max_count]
        tail = pmf[:, max_count:].sum(dim=1, keepdim=True)
        pmf = torch.cat((head, tail), dim=1)
    return pmf


__all__ = ["COLOUR_PERMUTATION_LEVELS", "MIRROR_ACTION", "augment_batch", "check_transition", "cleared_mask_from_transition", "mirror_episodes", "permute_colours", "poisson_binomial_pmf"]
