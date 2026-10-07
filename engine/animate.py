"""Turn an episode into a frame sequence.

A move is not a cut from one board to the next. It has phases, and the phases
are what make the dynamics legible: the swap, the matched run lighting up, the
clear, the survivors falling, new tiles entering from above, and then the whole
cycle again for each cascade round.

All geometry is computed from the :class:`~match3.spec.Transition` record that
the model already produces, so the animation cannot drift from the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pygame

from .render import Hud, Theme, Tile, board_tiles, render_tiles
from .scm import Episode
from .spec import CascadeStep, State, Transition


@dataclass
class Timing:
    """Phase durations in frames. Tuned for 30 fps."""

    fps: int = 30
    hold_start: int = 12
    swap: int = 8
    flash: int = 7
    clear: int = 6
    fall: int = 9
    settle_pause: int = 2
    hold_end: int = 30


def ease_out(u: float) -> float:
    return 1.0 - (1.0 - u) ** 3


def ease_in_out(u: float) -> float:
    return 3 * u**2 - 2 * u**3


def _hud(state: State, episode: Episode, note: str = "") -> Hud:
    return Hud(
        moves_left=state.moves_left,
        goals_left=state.goals_left,
        goal_colour=episode.difficulty.goal_colour,
        caption=f"{episode.level.name} · {episode.tier or 'given'}",
        subcaption=note
        or f"K={episode.player.label}  E={episode.E:+.2f}  "
        f"collect {episode.served_goal_count}",
    )


def _static_tiles(
    board: np.ndarray,
    theme: Theme,
    skip: set[tuple[int, int]],
    specials: np.ndarray | None = None,
) -> list[Tile]:
    tiles: list[Tile] = []
    height, width = board.shape
    for row in range(height):
        for col in range(width):
            if (row, col) in skip:
                continue
            value = int(board[row, col])
            if value < 0:
                continue
            x, y = theme.centre(row, col)
            special = 0 if specials is None else int(specials[row, col])
            tiles.append(Tile(x, y, value, special=special))
    return tiles


def _swap_frames(
    transition: Transition, theme: Theme, timing: Timing
) -> Iterator[list[Tile]]:
    action = transition.action
    assert action is not None
    (r1, c1), (r2, c2) = action.cells
    board = transition.board_before
    moving = {(r1, c1), (r2, c2)}
    before_specials = (
        np.zeros(board.shape, dtype=np.int8)
        if transition.specials_swapped is None
        else _apply_reverse_swap(transition.specials_swapped, action)
    )
    static = _static_tiles(board, theme, moving, before_specials)

    x1, y1 = theme.centre(r1, c1)
    x2, y2 = theme.centre(r2, c2)
    v1, v2 = int(board[r1, c1]), int(board[r2, c2])

    for frame in range(timing.swap):
        u = ease_in_out((frame + 1) / timing.swap)
        yield static + [
            Tile(x1 + (x2 - x1) * u, y1 + (y2 - y1) * u, v1, scale=1.0 + 0.10 * np.sin(np.pi * u), special=int(before_specials[r1, c1])),
            Tile(x2 + (x1 - x2) * u, y2 + (y1 - y2) * u, v2, scale=1.0 + 0.10 * np.sin(np.pi * u), special=int(before_specials[r2, c2])),
        ]


def _apply_reverse_swap(specials: np.ndarray, action) -> np.ndarray:
    before = specials.copy()
    (r1, c1), (r2, c2) = action.cells
    before[r1, c1], before[r2, c2] = specials[r2, c2], specials[r1, c1]
    return before


def _flash_frames(
    step: CascadeStep, theme: Theme, timing: Timing
) -> Iterator[list[Tile]]:
    matched = set(step.matched)
    static = _static_tiles(
        step.board_before, theme, matched, step.specials_before
    )
    for frame in range(timing.flash):
        u = (frame + 1) / timing.flash
        pulse = np.sin(np.pi * u)
        hot = []
        for row, col in step.matched:
            x, y = theme.centre(row, col)
            hot.append(
                Tile(
                    x,
                    y,
                    int(step.board_before[row, col]),
                    scale=1.0 + 0.14 * pulse,
                    glow=pulse,
                    special=(
                        0
                        if step.specials_before is None
                        else int(step.specials_before[row, col])
                    ),
                )
            )
        yield static + hot


def _clear_frames(
    step: CascadeStep, theme: Theme, timing: Timing
) -> Iterator[list[Tile]]:
    matched = set(step.matched)
    static = _static_tiles(
        step.board_before, theme, matched, step.specials_before
    )
    for frame in range(timing.clear):
        u = (frame + 1) / timing.clear
        gone = []
        for row, col in step.matched:
            x, y = theme.centre(row, col)
            gone.append(
                Tile(
                    x,
                    y,
                    int(step.board_before[row, col]),
                    scale=max(0.05, 1.0 - 0.85 * u),
                    alpha=max(0.0, 1.0 - u),
                    glow=1.0,
                    special=(
                        0
                        if step.specials_before is None
                        else int(step.specials_before[row, col])
                    ),
                )
            )
        yield static + gone


def _fall_frames(
    step: CascadeStep, theme: Theme, timing: Timing
) -> Iterator[list[Tile]]:
    """Interpolate survivors to their landing rows and drop new tiles in."""
    after = step.board_after
    height = after.shape[0]

    landing = {(tr, tc) for _, _, tr, tc in step.fall}
    spawn_cells = {(r, c) for r, c, _ in step.spawned}
    static = _static_tiles(
        after, theme, landing | spawn_cells, step.specials_after
    )

    # Spawned tiles in a column enter as a stack from above the top edge.
    per_column: dict[int, list[tuple[int, int]]] = {}
    for row, col, colour in step.spawned:
        per_column.setdefault(col, []).append((row, colour))

    for frame in range(timing.fall):
        u = ease_out((frame + 1) / timing.fall)
        tiles = list(static)

        for from_row, from_col, to_row, to_col in step.fall:
            x, _ = theme.centre(to_row, to_col)
            _, y0 = theme.centre(from_row, from_col)
            _, y1 = theme.centre(to_row, to_col)
            colour = int(step.board_before[from_row, from_col])
            special = (
                0
                if step.specials_before is None
                else int(step.specials_before[from_row, from_col])
            )
            tiles.append(Tile(x, y0 + (y1 - y0) * u, colour, special=special))

        for col, entries in per_column.items():
            count = len(entries)
            for row, colour in entries:
                x, y1 = theme.centre(row, col)
                _, y0 = theme.centre(row - count, col)
                tiles.append(Tile(x, y0 + (y1 - y0) * u, colour))

        yield tiles


def _still(
    board: np.ndarray,
    theme: Theme,
    count: int,
    specials: np.ndarray | None = None,
) -> Iterator[list[Tile]]:
    tiles = board_tiles(board, theme, specials)
    for _ in range(count):
        yield list(tiles)


def episode_frames(
    episode: Episode,
    theme: Theme | None = None,
    timing: Timing | None = None,
) -> Iterator[pygame.Surface]:
    """Yield every frame of the episode, in order."""
    theme = theme or Theme()
    timing = timing or Timing()
    height, width = episode.states[0].board.shape

    def emit(tiles: list[Tile], hud: Hud) -> pygame.Surface:
        return render_tiles(tiles, hud, height, width, theme)

    opening = episode.states[0]
    for tiles in _still(
        opening.board, theme, timing.hold_start, opening.specials
    ):
        yield emit(tiles, _hud(opening, episode))

    for index, transition in enumerate(episode.transitions):
        before = episode.states[index]
        after = episode.states[index + 1]
        hud_mid = _hud(before, episode)

        for tiles in _swap_frames(transition, theme, timing):
            yield emit(tiles, hud_mid)

        for step in transition.steps:
            for tiles in _flash_frames(step, theme, timing):
                yield emit(tiles, hud_mid)
            for tiles in _clear_frames(step, theme, timing):
                yield emit(tiles, hud_mid)
            for tiles in _fall_frames(step, theme, timing):
                yield emit(tiles, hud_mid)
            for tiles in _still(
                step.board_after,
                theme,
                timing.settle_pause,
                step.specials_after,
            ):
                yield emit(tiles, hud_mid)

        if transition.reshuffled:
            for tiles in _still(
                after.board, theme, timing.flash, after.specials
            ):
                yield emit(tiles, _hud(after, episode, note="board reshuffled"))

    final = episode.states[-1]
    verdict = "cleared" if episode.R else "out of moves"
    note = f"{verdict} · {episode.goals_cleared}/{episode.served_goal_count} collected"
    for tiles in _still(final.board, theme, timing.hold_end, final.specials):
        yield emit(tiles, _hud(final, episode, note=note))


def count_frames(episode: Episode, timing: Timing | None = None) -> int:
    timing = timing or Timing()
    total = timing.hold_start + timing.hold_end
    for transition in episode.transitions:
        total += timing.swap
        for _ in transition.steps:
            total += timing.flash + timing.clear + timing.fall + timing.settle_pause
        if transition.reshuffled:
            total += timing.flash
    return total


__all__ = ["Timing", "count_frames", "ease_in_out", "ease_out", "episode_frames"]
