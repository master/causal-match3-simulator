"""pygame drawing: turn a board, or a bag of tile sprites, into a surface.

Rendering is deliberately a pure consumer of state produced by
:mod:`match3.scm`. Nothing here re-implements a game rule, so a frame cannot
disagree with the model that produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402  (import after the driver hints)

from .spec import HORIZONTAL_STRIPE, NO_SPECIAL, VERTICAL_STRIPE

#: Tile colours. Each also gets a distinct inner glyph so the board stays
#: readable in greyscale and for colour-vision deficiency.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (226, 86, 74),
    (61, 127, 209),
    (63, 170, 106),
    (232, 179, 60),
    (142, 99, 201),
    (47, 182, 182),
)

GLYPHS = ("circle", "diamond", "square", "triangle", "hex", "cross")

BACKDROP = (26, 24, 32)
BOARD_BG = (244, 241, 236)
BOARD_LINE = (222, 217, 208)
TEXT = (238, 236, 244)
TEXT_DIM = (150, 146, 162)


@dataclass
class Theme:
    cell: int = 56
    gap: int = 4
    margin: int = 18
    hud_height: int = 74
    radius: int = 12
    font_name: str | None = None

    def board_px(self, height: int, width: int) -> tuple[int, int]:
        w = width * self.cell + (width - 1) * self.gap
        h = height * self.cell + (height - 1) * self.gap
        return w, h

    def surface_size(self, height: int, width: int) -> tuple[int, int]:
        w, h = self.board_px(height, width)
        return w + 2 * self.margin, h + 2 * self.margin + self.hud_height

    def origin(self) -> tuple[int, int]:
        return self.margin, self.margin + self.hud_height

    def centre(self, row: float, col: float) -> tuple[float, float]:
        ox, oy = self.origin()
        x = ox + col * (self.cell + self.gap) + self.cell / 2
        y = oy + row * (self.cell + self.gap) + self.cell / 2
        return x, y


@dataclass
class Tile:
    """A drawable tile in pixel space, so animations can place it freely."""

    x: float
    y: float
    colour: int
    scale: float = 1.0
    alpha: float = 1.0
    glow: float = 0.0
    special: int = NO_SPECIAL


@dataclass
class Hud:
    moves_left: int
    goals_left: int
    goal_colour: int
    caption: str = ""
    subcaption: str = ""


_FONT_CACHE: dict[tuple[str | None, int, bool], pygame.font.Font] = {}


def _font(theme: Theme, size: int, bold: bool = False) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    key = (theme.font_name, size, bold)
    if key not in _FONT_CACHE:
        font = pygame.font.SysFont(theme.font_name or "helvetica,arial", size, bold=bold)
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


def _mix(colour: tuple[int, int, int], target: tuple[int, int, int], f: float):
    return tuple(int(round(c + (t - c) * f)) for c, t in zip(colour, target))


def _glyph_points(kind: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    if kind == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    if kind == "triangle":
        return [(cx, cy - r), (cx + r * 0.92, cy + r * 0.72), (cx - r * 0.92, cy + r * 0.72)]
    if kind == "hex":
        return [
            (cx + r * np.cos(a), cy + r * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
        ]
    return []


def draw_tile(surface: pygame.Surface, tile: Tile, theme: Theme) -> None:
    if tile.alpha <= 0.01 or tile.scale <= 0.01:
        return

    size = max(2, int(round(theme.cell * tile.scale)))
    layer = pygame.Surface((size, size), pygame.SRCALPHA)

    base = PALETTE[tile.colour % len(PALETTE)]
    if tile.glow > 0:
        base = _mix(base, (255, 255, 255), min(1.0, tile.glow) * 0.55)

    radius = max(2, int(theme.radius * tile.scale))
    rect = pygame.Rect(0, 0, size, size)
    pygame.draw.rect(layer, (*base, 255), rect, border_radius=radius)

    # A narrow band along the top edge reads as a light source without
    # competing with the glyph for the middle of the tile.
    band = pygame.Rect(size * 0.16, size * 0.09, size * 0.68, size * 0.16)
    pygame.draw.ellipse(layer, (*_mix(base, (255, 255, 255), 0.45), 60), band)

    glyph = GLYPHS[tile.colour % len(GLYPHS)]
    ink = (*_mix(base, (0, 0, 0), 0.30), 235)
    cx = cy = size / 2
    cy = size * 0.56
    r = size * 0.23
    if glyph == "circle":
        pygame.draw.circle(layer, ink, (cx, cy), r)
    elif glyph == "square":
        pygame.draw.rect(layer, ink, pygame.Rect(cx - r, cy - r, 2 * r, 2 * r),
                         border_radius=int(r * 0.35))
    elif glyph == "cross":
        pygame.draw.rect(layer, ink, pygame.Rect(cx - r, cy - r * 0.33, 2 * r, r * 0.66))
        pygame.draw.rect(layer, ink, pygame.Rect(cx - r * 0.33, cy - r, r * 0.66, 2 * r))
    else:
        pygame.draw.polygon(layer, ink, _glyph_points(glyph, cx, cy, r))

    stripe_ink = (*_mix(base, (255, 255, 255), 0.82), 245)
    thickness = max(3, int(size * 0.10))
    if tile.special == HORIZONTAL_STRIPE:
        pygame.draw.line(
            layer,
            stripe_ink,
            (size * 0.14, size * 0.50),
            (size * 0.86, size * 0.50),
            thickness,
        )
    elif tile.special == VERTICAL_STRIPE:
        pygame.draw.line(
            layer,
            stripe_ink,
            (size * 0.50, size * 0.14),
            (size * 0.50, size * 0.86),
            thickness,
        )

    if tile.alpha < 1.0:
        layer.set_alpha(int(round(255 * tile.alpha)))

    surface.blit(layer, (tile.x - size / 2, tile.y - size / 2))


def board_tiles(
    board: np.ndarray,
    theme: Theme,
    specials: np.ndarray | None = None,
) -> list[Tile]:
    tiles: list[Tile] = []
    height, width = board.shape
    for row in range(height):
        for col in range(width):
            value = int(board[row, col])
            if value < 0:
                continue
            x, y = theme.centre(row, col)
            special = NO_SPECIAL if specials is None else int(specials[row, col])
            tiles.append(Tile(x, y, value, special=special))
    return tiles


def draw_hud(surface: pygame.Surface, hud: Hud, theme: Theme, width_px: int) -> None:
    title = _font(theme, 21, bold=True)
    small = _font(theme, 14)

    if hud.caption:
        surface.blit(title.render(hud.caption, True, TEXT), (theme.margin, 14))
    if hud.subcaption:
        surface.blit(
            small.render(hud.subcaption, True, TEXT_DIM), (theme.margin, 42)
        )

    chip = 18
    right = width_px - theme.margin
    moves = title.render(str(hud.moves_left), True, TEXT)
    goals = title.render(str(hud.goals_left), True, TEXT)

    surface.blit(moves, (right - moves.get_width(), 14))
    surface.blit(
        small.render("moves", True, TEXT_DIM),
        (right - moves.get_width() - 4 - small.size("moves")[0] - 6, 20),
    )

    gx = right - goals.get_width()
    surface.blit(goals, (gx, 42))
    swatch = pygame.Rect(gx - chip - 8, 44, chip, chip)
    pygame.draw.rect(
        surface, PALETTE[hud.goal_colour % len(PALETTE)], swatch, border_radius=5
    )


def new_surface(height: int, width: int, theme: Theme) -> pygame.Surface:
    w, h = theme.surface_size(height, width)
    surface = pygame.Surface((w, h))
    surface.fill(BACKDROP)

    ox, oy = theme.origin()
    bw, bh = theme.board_px(height, width)
    pad = theme.gap
    pygame.draw.rect(
        surface,
        BOARD_BG,
        pygame.Rect(ox - pad, oy - pad, bw + 2 * pad, bh + 2 * pad),
        border_radius=theme.radius + 4,
    )
    return surface


def render_tiles(
    tiles: list[Tile],
    hud: Hud,
    height: int,
    width: int,
    theme: Theme | None = None,
) -> pygame.Surface:
    theme = theme or Theme()
    surface = new_surface(height, width, theme)
    for tile in tiles:
        draw_tile(surface, tile, theme)
    draw_hud(surface, hud, theme, surface.get_width())
    return surface


def render_board(
    board: np.ndarray,
    hud: Hud,
    theme: Theme | None = None,
    specials: np.ndarray | None = None,
) -> pygame.Surface:
    theme = theme or Theme()
    height, width = board.shape
    return render_tiles(
        board_tiles(board, theme, specials), hud, height, width, theme
    )


def save_png(surface: pygame.Surface, path) -> None:
    pygame.image.save(surface, str(path))


__all__ = [
    "BACKDROP",
    "Hud",
    "PALETTE",
    "Theme",
    "Tile",
    "board_tiles",
    "draw_tile",
    "new_surface",
    "render_board",
    "render_tiles",
    "save_png",
]
