"""Latent world models: a per-cell encoder, a transformer predictor of the next latent and the shared structured read-out as a detached probe.
The objectives differ only in the target latent: "jepa" encodes it without gradient (by the online encoder, target="stopgrad", or by an EMA copy, target="ema") and "lewm" encodes it with gradient (target="shared") and applies SIGReg to both states.

WM-1 additions (off by default): encoders take the special-kind grid (``use_specials``); a ``BoardEmbedding`` of the *current* board feeds the
cleared-cell head (on ``cat(predicted_cells, board_embed)``) and the residual / terminal heads (on ``cat(predicted_global, board_embed.mean, counters)``)
— these outcome heads are **never detached** so the predictor keeps goal-relevant information; ``readout_detach`` governs only the colour / special
read-out. ``goal_head="mask"`` replaces the categorical goal head by the Poisson-binomial derived count (see ``outcome.py``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from match3_simulator.learned_model.tokens import ACTION_SLOTS, BOARD_HEIGHT, BOARD_WIDTH, CELL1_TOKEN, CELL2_TOKEN, N_CELLS
from match3_simulator.scm import TIER_MOVE_BUDGETS
from match3_simulator.world_modeling.decoder import SpecialKindHead, StructuredDecoderConfig, StructuredStateDecoder
from match3_simulator.world_modeling.outcome import ResidualSpec, TerminalHead, cleared_probabilities, derived_goal_nll, mask_bce, sample_goal_delta, terminal_target
from match3_simulator.world_modeling.sigreg import sigreg
from match3_simulator.world_modeling.wm1_mechanics import N_SPECIAL_KINDS, round0, sample_specials

OBJECTIVES: tuple[str, ...] = ("jepa", "lewm")
TARGETS: tuple[str, ...] = ("stopgrad", "ema", "shared")
ENCODERS: tuple[str, ...] = ("conv", "transformer")
SIGREG_ROWS: tuple[str, ...] = ("cells", "states")
ROLLOUTS: tuple[str, ...] = ("latent", "reencode")
GOAL_HEADS: tuple[str, ...] = ("scalar", "mask")


@dataclass(frozen=True)
class LatentWorldModelConfig:
    """Sizes and objective switches of the latent world model."""

    kind: str = "latent"
    objective: str = "jepa"
    target: str = "stopgrad"
    ema_decay: float = 0.99
    n_colours: int = 6
    n_levels: int = 3
    n_tiers: int = 3
    embedding_size: int = 32
    task_context_size: int = 32
    max_moves_left: int = max(TIER_MOVE_BUDGETS)
    goals_scale: float = 32.0
    encoder: str = "conv"
    encoder_layers: int = 2
    spatial_colour_embedding_size: int = 16
    latent_channels: int = 32
    predictor_width: int = 128
    predictor_layers: int = 4
    predictor_heads: int = 4
    dropout: float = 0.0
    predictor_residual: bool = True
    latent_loss_weight: float = 1.0
    changed_cell_weight: float = 1.0
    sigreg_weight: float = 1.0
    sigreg_rows: str = "cells"
    sigreg_projections: int = 256
    colour_head_weight: float = 1.0
    readout_detach: bool = True
    rollout: str = "latent"
    decoder_hidden_size: int = 128
    max_goal_delta: int = 20
    # WM-1 heads and supports
    use_specials: bool = False
    special_head: bool = False
    terminal_head: bool = False
    cleared_head: bool = False
    goal_head: str = "scalar"
    round0_clamp: bool = False
    residual_goal_min: int = 0
    residual_goal_classes: int = 8
    board_embedding_size: int = 32
    outcome_detach: bool = False  # ablation only: read the cleared / residual / terminal heads from detached latents (the WM-1 default is non-detached, as the task requires)
    # post-hoc variant: carried-latent (open-loop) training — roll the predictor open_loop_steps further steps on its own predicted latents with the logged
    # actions and apply the latent, cleared-mask, derived-goal and terminal losses at every carried step (the heads then train on the latents they read in imagination)
    open_loop_steps: int = 0
    open_loop_weight: float = 1.0
    mask_loss_weight: float = 1.0
    special_loss_weight: float = 1.0
    terminal_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("n_colours", "n_levels", "n_tiers", "embedding_size", "task_context_size", "max_moves_left", "encoder_layers",
                     "spatial_colour_embedding_size", "latent_channels", "predictor_width", "predictor_layers", "predictor_heads",
                     "sigreg_projections", "decoder_hidden_size", "max_goal_delta", "residual_goal_classes", "board_embedding_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.kind != "latent":
            raise ValueError("kind must be 'latent'")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        if (self.objective == "lewm") != (self.target == "shared"):
            raise ValueError("objective='lewm' requires target='shared' (end-to-end target); objective='jepa' takes target='stopgrad' or 'ema'")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}")
        if self.sigreg_rows not in SIGREG_ROWS:
            raise ValueError(f"sigreg_rows must be one of {SIGREG_ROWS}")
        if self.objective == "lewm" and self.sigreg_rows != "states":
            raise ValueError("objective='lewm' requires sigreg_rows='states': cell rows are satisfied by a board-independent positional code")
        if self.rollout not in ROLLOUTS:
            raise ValueError(f"rollout must be one of {ROLLOUTS}")
        if self.goal_head not in GOAL_HEADS:
            raise ValueError(f"goal_head must be one of {GOAL_HEADS}")
        if self.goal_head == "mask" and not self.cleared_head:
            raise ValueError("goal_head='mask' requires cleared_head=True")
        if self.round0_clamp and not self.cleared_head:
            raise ValueError("round0_clamp requires cleared_head=True")
        if self.goals_scale <= 0 or self.changed_cell_weight <= 0:
            raise ValueError("goals_scale and changed_cell_weight must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.predictor_width % self.predictor_heads:
            raise ValueError("predictor_width must be divisible by predictor_heads")
        if min(self.latent_loss_weight, self.sigreg_weight, self.colour_head_weight, self.mask_loss_weight, self.special_loss_weight, self.terminal_loss_weight, self.open_loop_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.open_loop_steps < 0 or (self.open_loop_steps > 0 and self.goal_head != "mask"):
            raise ValueError("open_loop_steps must be non-negative, and open_loop_steps > 0 requires goal_head='mask'")

    @property
    def target_has_gradient(self) -> bool:
        return self.target == "shared"

    @property
    def uses_ema(self) -> bool:
        return self.target == "ema"

    @property
    def needs_mechanics(self) -> bool:
        return self.use_specials or self.special_head or self.round0_clamp


class ConvEncoder(nn.Module):
    """Colour, goal, special and coordinate planes through two 3x3 convolutions and a 1x1 map to per-cell latents."""

    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()
        self.config = config
        channels = config.latent_channels
        self.colour = nn.Embedding(config.n_colours, config.spatial_colour_embedding_size)
        extra = 3 + (N_SPECIAL_KINDS - 1 if config.use_specials else 0)
        self.convolutions = nn.Sequential(
            nn.Conv2d(config.spatial_colour_embedding_size + extra, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.counters = nn.Sequential(nn.Linear(2, channels), nn.GELU(), nn.Linear(channels, channels))
        rows = torch.arange(BOARD_HEIGHT, dtype=torch.float32) / max(BOARD_HEIGHT - 1, 1)
        cols = torch.arange(BOARD_WIDTH, dtype=torch.float32) / max(BOARD_WIDTH - 1, 1)
        coordinates = torch.stack((rows.view(BOARD_HEIGHT, 1).expand(BOARD_HEIGHT, BOARD_WIDTH), cols.view(1, BOARD_WIDTH).expand(BOARD_HEIGHT, BOARD_WIDTH)), dim=0)
        self.register_buffer("coordinates", coordinates, persistent=False)

    def forward(self, boards: torch.Tensor, goal_colours: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor, specials: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode boards and counters into per-cell and global latents.

        Inputs: boards (N, 64); goal_colours, moves_left, goals_left (N,); specials (N, 64) or None.
        Outputs: cell latents (N, 64, C) and global latents (N, C).
        """
        rows = _check_state(boards, goal_colours, moves_left, goals_left)
        boards = boards.long()
        colours = self.colour(boards).view(rows, BOARD_HEIGHT, BOARD_WIDTH, -1).permute(0, 3, 1, 2)
        goal = (boards == goal_colours.long().unsqueeze(1)).to(colours.dtype).view(rows, 1, BOARD_HEIGHT, BOARD_WIDTH)
        coordinates = self.coordinates.unsqueeze(0).expand(rows, -1, -1, -1).to(colours.dtype)
        planes = [colours, goal, coordinates]
        if self.config.use_specials:
            kinds = torch.zeros_like(boards) if specials is None else specials.long()
            planes.append(F.one_hot(kinds, N_SPECIAL_KINDS)[..., 1:].to(colours.dtype).view(rows, BOARD_HEIGHT, BOARD_WIDTH, -1).permute(0, 3, 1, 2))
        cells = self.convolutions(torch.cat(planes, dim=1)).flatten(2).transpose(1, 2)
        counters = torch.stack((moves_left.to(cells.dtype) / self.config.max_moves_left, goals_left.to(cells.dtype) / self.config.goals_scale), dim=-1)
        return cells, cells.mean(dim=1) + self.counters(counters)


class TokenEncoder(nn.Module):
    """Tile tokens (colour, row, column, is-goal, special kind) and a counter token through a transformer encoder to per-cell latents."""

    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()
        self.config = config
        width = config.predictor_width
        self.colour = nn.Embedding(config.n_colours, width)
        self.row = nn.Embedding(BOARD_HEIGHT, width)
        self.col = nn.Embedding(BOARD_WIDTH, width)
        self.is_goal = nn.Embedding(2, width)
        self.special: nn.Embedding | None = nn.Embedding(N_SPECIAL_KINDS, width) if config.use_specials else None
        self.moves = nn.Embedding(config.max_moves_left + 1, width)
        self.goals = nn.Linear(1, width)
        self.token_type = nn.Embedding(2, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=config.predictor_heads, dim_feedforward=4 * width, dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=config.encoder_layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.cell_out = nn.Linear(width, config.latent_channels)
        self.global_out = nn.Linear(width, config.latent_channels)
        positions = torch.arange(N_CELLS)
        self.register_buffer("cell_rows", positions // BOARD_WIDTH, persistent=False)
        self.register_buffer("cell_cols", positions % BOARD_WIDTH, persistent=False)

    def forward(self, boards: torch.Tensor, goal_colours: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor, specials: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode boards and counters into per-cell and global latents.

        Inputs: boards (N, 64); goal_colours, moves_left, goals_left (N,); specials (N, 64) or None.
        Outputs: cell latents (N, 64, C) and global latents (N, C).
        """
        _check_state(boards, goal_colours, moves_left, goals_left)
        boards = boards.long()
        tiles = (self.colour(boards) + self.row(self.cell_rows).unsqueeze(0) + self.col(self.cell_cols).unsqueeze(0)
                 + self.is_goal((boards == goal_colours.long().unsqueeze(1)).long()) + self.token_type.weight[0])
        if self.special is not None:
            tiles = tiles + self.special(torch.zeros_like(boards) if specials is None else specials.long())
        counter = (self.moves(moves_left.long().clamp(0, self.config.max_moves_left))
                   + self.goals((goals_left.to(tiles.dtype) / self.config.goals_scale).unsqueeze(1)) + self.token_type.weight[1]).unsqueeze(1)
        hidden = self.trunk(torch.cat((tiles, counter), dim=1))
        return self.cell_out(hidden[:, :N_CELLS]), self.global_out(hidden[:, N_CELLS])


class BoardEmbedding(nn.Module):
    """Per-cell embedding of the *current* board (colour, is-goal, special kind) that feeds the outcome heads."""

    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()
        size = config.board_embedding_size
        self.colour = nn.Embedding(config.n_colours, size)
        self.is_goal = nn.Embedding(2, size)
        self.special = nn.Embedding(N_SPECIAL_KINDS, size)

    def forward(self, boards: torch.Tensor, goal_colours: torch.Tensor, specials: torch.Tensor | None) -> torch.Tensor:
        """Inputs: boards (N, 64); goal_colours (N,); specials (N, 64) or None. Outputs: (N, 64, size)."""
        boards = boards.long()
        kinds = torch.zeros_like(boards) if specials is None else specials.long()
        return self.colour(boards) + self.is_goal((boards == goal_colours.long().unsqueeze(1)).long()) + self.special(kinds)


def _check_state(boards: torch.Tensor, goal_colours: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor) -> int:
    """Validate a flat batch of states and return its size.

    Inputs: boards (N, 64); goal_colours, moves_left, goals_left (N,).
    Outputs: N.
    """
    if boards.ndim != 2 or boards.shape[1] != N_CELLS:
        raise ValueError("boards must have shape (rows, 64)")
    rows = boards.shape[0]
    for name, values in {"goal_colours": goal_colours, "moves_left": moves_left, "goals_left": goals_left}.items():
        if values.shape != (rows,):
            raise ValueError(f"{name} must have shape ({rows},)")
    return rows


class LatentWorldModel(nn.Module):
    """Encode the board, predict the next latent from the action and task, and read boards out through a detached structured decoder."""

    kind = "latent"

    def __init__(self, config: LatentWorldModelConfig = LatentWorldModelConfig()):
        super().__init__()
        self.config = config
        channels = config.latent_channels
        width = config.predictor_width
        embedding = config.embedding_size
        self.encoder: nn.Module = TokenEncoder(config) if config.encoder == "transformer" else ConvEncoder(config)
        # target="ema": a frozen copy of the encoder that follows it by exponential moving average (post_optimizer_step).
        self.target_encoder: nn.Module | None = None
        if config.uses_ema:
            self.target_encoder = copy.deepcopy(self.encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        self.cell_in = nn.Linear(channels, width)
        self.row = nn.Embedding(BOARD_HEIGHT, width)
        self.col = nn.Embedding(BOARD_WIDTH, width)
        self.action_role = nn.Embedding(3, width)
        self.action = nn.Embedding(ACTION_SLOTS, width)
        self.level = nn.Embedding(config.n_levels, embedding)
        self.tier = nn.Embedding(config.n_tiers, embedding)
        self.task_projection = nn.Sequential(nn.Linear(2 * embedding + 1, config.task_context_size), nn.GELU(), nn.Linear(config.task_context_size, config.task_context_size))
        self.task_token = nn.Linear(config.task_context_size, width)
        self.global_in = nn.Linear(channels, width)
        self.moves = nn.Embedding(config.max_moves_left + 1, width)
        self.goals = nn.Linear(1, width)
        self.token_type = nn.Embedding(4, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=config.predictor_heads, dim_feedforward=4 * width, dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=config.predictor_layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.cell_out = nn.Linear(width, channels)
        self.global_out = nn.Linear(width, channels)
        self.colour_head: nn.Module | None = nn.Linear(channels, config.n_colours) if config.colour_head_weight > 0 else None
        self.structured = StructuredStateDecoder(StructuredDecoderConfig(
            n_colours=config.n_colours, cell_size=channels, global_size=channels, hidden_size=config.decoder_hidden_size, max_goal_delta=config.max_goal_delta,
            goal_head=config.goal_head == "scalar",
        ))
        self.initial_board_decoder = nn.Sequential(nn.Linear(config.task_context_size, width), nn.GELU(), nn.Linear(width, N_CELLS * config.n_colours))
        self.initial_counter_decoder = nn.Sequential(nn.Linear(config.task_context_size, width), nn.GELU(), nn.Linear(width, 2))
        # WM-1 outcome heads (never detached) and read-out heads
        hidden = config.decoder_hidden_size
        size = config.board_embedding_size
        self.board_embedding: BoardEmbedding | None = BoardEmbedding(config) if (config.cleared_head or config.terminal_head) else None
        self.cleared_head_net: nn.Module | None = nn.Sequential(nn.Linear(channels + size, hidden), nn.GELU(), nn.Linear(hidden, 1)) if config.cleared_head else None
        self.residual_goal: nn.Module | None = (nn.Sequential(nn.Linear(channels + size + 2, hidden), nn.GELU(), nn.Linear(hidden, config.residual_goal_classes))
                                                if config.goal_head == "mask" else None)
        self.residual = ResidualSpec(config.residual_goal_min, config.residual_goal_classes)
        self.terminal_head_net: TerminalHead | None = TerminalHead(channels + size, hidden) if config.terminal_head else None
        self.special_head_net: SpecialKindHead | None = SpecialKindHead(channels, config.n_colours, hidden) if config.special_head else None
        positions = torch.arange(N_CELLS)
        self.register_buffer("cell_rows", positions // BOARD_WIDTH, persistent=False)
        self.register_buffer("cell_cols", positions % BOARD_WIDTH, persistent=False)
        self.register_buffer("cell1", torch.as_tensor(CELL1_TOKEN), persistent=False)
        self.register_buffer("cell2", torch.as_tensor(CELL2_TOKEN), persistent=False)

    def encode_task(self, levels: torch.Tensor, tiers: torch.Tensor, served_difficulty: torch.Tensor) -> torch.Tensor:
        """Embed the observed level, baseline tier and served difficulty.

        Inputs: levels (B,) long; tiers (B,) long; served_difficulty (B,) float.
        Outputs: task context (B, task_context_size).
        """
        if levels.ndim != 1 or tiers.shape != levels.shape or served_difficulty.shape != levels.shape:
            raise ValueError("levels, tiers and served_difficulty must be aligned vectors")
        return self.task_projection(torch.cat((self.level(levels.long()), self.tier(tiers.long()), served_difficulty.to(self.level.weight.dtype).unsqueeze(-1)), dim=-1))

    def initial_state_predictions(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameterise the opening board and counters from the task context.

        Inputs: context (B, task_context_size).
        Outputs: board logits (B, 64, colours) and counter means (B, 2).
        """
        if context.ndim != 2 or context.shape[1] != self.config.task_context_size:
            raise ValueError("context has the wrong shape")
        return self.initial_board_decoder(context).view(context.shape[0], N_CELLS, self.config.n_colours), self.initial_counter_decoder(context)

    def encode(self, boards: torch.Tensor, goal_colours: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor, specials: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the current state with the online encoder.

        Inputs: boards (N, 64); goal_colours, moves_left, goals_left (N,); specials (N, 64) or None.
        Outputs: cell latents (N, 64, C) and global latents (N, C).
        """
        return self.encoder(boards, goal_colours, moves_left, goals_left, specials)

    def encode_target(self, boards: torch.Tensor, goal_colours: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor, specials: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the next state as the prediction target: with gradient under "shared", detached from the online encoder under "stopgrad", detached from the EMA copy under "ema".

        Inputs: boards (N, 64); goal_colours, moves_left, goals_left (N,); specials (N, 64) or None.
        Outputs: target cell latents (N, 64, C) and global latents (N, C).
        """
        if self.config.target_has_gradient:
            return self.encoder(boards, goal_colours, moves_left, goals_left, specials)
        encoder = self.target_encoder if self.target_encoder is not None else self.encoder
        with torch.no_grad():
            cells, global_latent = encoder(boards, goal_colours, moves_left, goals_left, specials)
        return cells.detach(), global_latent.detach()

    @torch.no_grad()
    def post_optimizer_step(self) -> None:
        """Move the EMA target encoder toward the online encoder; a no-op without a target network.

        Inputs: none (call once after every optimizer.step()).
        Outputs: none.
        """
        if self.target_encoder is None:
            return
        decay = self.config.ema_decay
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(decay).add_(online.detach(), alpha=1.0 - decay)

    def action_roles(self, actions: torch.Tensor) -> torch.Tensor:
        """Mark the two swapped cells of each action on the 64-cell grid.

        Inputs: actions (N,) long in [0, 128).
        Outputs: roles (N, 64) long with 0 untouched, 1 first cell, 2 second cell.
        """
        roles = torch.zeros((actions.shape[0], N_CELLS), dtype=torch.long, device=actions.device)
        index = torch.arange(actions.shape[0], device=actions.device)
        roles[index, self.cell1[actions.long()]] = 1
        roles[index, self.cell2[actions.long()]] = 2
        return roles

    def predict(
        self,
        cells: torch.Tensor,
        global_latent: torch.Tensor,
        *,
        actions: torch.Tensor,
        task_context: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next latents from the current latents, the action, the task context and the counters.

        Inputs: cells (N, 64, C); global_latent (N, C); actions, moves_left, goals_left (N,); task_context (N, task_context_size).
        Outputs: predicted cell latents (N, 64, C) and global latent (N, C), added to the current latents when predictor_residual is set.
        """
        rows = global_latent.shape[0]
        if cells.shape != (rows, N_CELLS, self.config.latent_channels) or global_latent.shape != (rows, self.config.latent_channels):
            raise ValueError("cells must have shape (rows, 64, C) and global_latent (rows, C)")
        for name, values in {"actions": actions, "moves_left": moves_left, "goals_left": goals_left}.items():
            if values.shape != (rows,):
                raise ValueError(f"{name} must have shape ({rows},)")
        if task_context.shape != (rows, self.config.task_context_size):
            raise ValueError("task_context has the wrong shape")
        tiles = (self.cell_in(cells) + self.row(self.cell_rows).unsqueeze(0) + self.col(self.cell_cols).unsqueeze(0)
                 + self.action_role(self.action_roles(actions)) + self.token_type.weight[0])
        action_token = (self.action(actions.long()) + self.token_type.weight[1]).unsqueeze(1)
        task_token = (self.task_token(task_context) + self.token_type.weight[2]).unsqueeze(1)
        global_token = (self.global_in(global_latent) + self.moves(moves_left.long().clamp(0, self.config.max_moves_left))
                        + self.goals((goals_left.to(global_latent.dtype) / self.config.goals_scale).unsqueeze(1)) + self.token_type.weight[3]).unsqueeze(1)
        hidden = self.trunk(torch.cat((tiles, action_token, task_token, global_token), dim=1))
        predicted_cells = self.cell_out(hidden[:, :N_CELLS])
        predicted_global = self.global_out(hidden[:, N_CELLS + 2])
        if self.config.predictor_residual:
            return cells + predicted_cells, global_latent + predicted_global
        return predicted_cells, predicted_global

    def forward(self, *, boards, goal_colours, moves_left, goals_left, actions, task_context, specials=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the current state and predict the next latents.

        Inputs: boards (N, 64); goal_colours, moves_left, goals_left, actions (N,); task_context (N, task_context_size); specials (N, 64) or None.
        Outputs: predicted cell latents (N, 64, C) and global latent (N, C).
        """
        cells, global_latent = self.encode(boards, goal_colours, moves_left, goals_left, specials)
        return self.predict(cells, global_latent, actions=actions, task_context=task_context, moves_left=moves_left, goals_left=goals_left)

    def _sigreg(self, branches: list[torch.Tensor], generator: torch.Generator | None) -> torch.Tensor:
        """Apply SIGReg to the configured row unit of the encoded cell latents.

        Inputs: list of cell-latent tensors (N_i, 64, C), one per branch; optional generator.
        Outputs: scalar statistic (one row per cell for "cells", one flattened row per state for "states").
        """
        channels = self.config.latent_channels
        if self.config.sigreg_rows == "cells":
            rows = torch.cat([c.reshape(-1, channels) for c in branches], dim=0)
        else:
            rows = torch.cat([c.reshape(c.shape[0], N_CELLS * channels) for c in branches], dim=0)
        return sigreg(rows, projections=self.config.sigreg_projections, generator=generator)

    def _known(self, boards: torch.Tensor, specials: torch.Tensor | None, actions: torch.Tensor) -> dict[str, torch.Tensor] | None:
        if not self.config.needs_mechanics:
            return None
        with torch.no_grad():
            return round0(boards.long(), torch.zeros_like(boards, dtype=torch.long) if specials is None else specials.long(), actions.long())

    def _outcome(self, predicted_cells: torch.Tensor, predicted_global: torch.Tensor, boards: torch.Tensor, goal_colours: torch.Tensor, specials: torch.Tensor | None,
                 moves_left: torch.Tensor, goals_left: torch.Tensor, known: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
        """Mask probabilities, residual logits, expected delta and terminal logit from the (non-detached) predicted latents and the current board embedding."""
        assert self.board_embedding is not None
        if self.config.outcome_detach:
            predicted_cells, predicted_global = predicted_cells.detach(), predicted_global.detach()
        embed = self.board_embedding(boards, goal_colours, specials)
        pooled = embed.mean(dim=1)
        counters = torch.stack((moves_left.to(pooled.dtype) / self.config.max_moves_left, goals_left.to(pooled.dtype) / self.config.goals_scale), dim=-1)
        global_features = torch.cat((predicted_global, pooled), dim=-1)
        out: dict[str, torch.Tensor] = {}
        goal_cells = boards.long() == goal_colours.long().unsqueeze(1)
        out["goal_cells"] = goal_cells
        expected_delta = None
        if self.cleared_head_net is not None:
            mask_logits = self.cleared_head_net(torch.cat((predicted_cells, embed), dim=-1)).squeeze(-1)
            round0_pre = known["cleared_pre"] if (known is not None and self.config.round0_clamp) else None
            probabilities = cleared_probabilities(mask_logits, round0_pre)
            out.update({"mask_logits": mask_logits, "round0_pre": round0_pre, "probabilities": probabilities})
            if self.residual_goal is not None:
                residual_logits = self.residual_goal(torch.cat((global_features, counters), dim=-1))
                expected_delta = (probabilities * goal_cells.to(probabilities.dtype)).sum(dim=-1) + self.residual.expected(residual_logits)
                out.update({"residual_logits": residual_logits, "expected_delta": expected_delta})
        if self.terminal_head_net is not None:
            delta = expected_delta if expected_delta is not None else torch.zeros(boards.shape[0], device=boards.device, dtype=pooled.dtype)
            out["terminal_logit"] = self.terminal_head_net(global_features, moves_left, goals_left, delta, max_moves=self.config.max_moves_left, goals_scale=self.config.goals_scale)
        return out

    def objective(
        self,
        *,
        boards: torch.Tensor,
        next_boards: torch.Tensor,
        actions: torch.Tensor,
        goal_colours: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        next_moves_left: torch.Tensor,
        next_goals_left: torch.Tensor,
        levels: torch.Tensor,
        tiers: torch.Tensor,
        served_difficulty: torch.Tensor,
        step_mask: torch.Tensor,
        cleared_masks: torch.Tensor | None = None,
        specials: torch.Tensor | None = None,
        next_specials: torch.Tensor | None = None,
        residual_goal: torch.Tensor | None = None,
        refill_goal_cleared: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
        mask_valid: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Compute the latent prediction loss, SIGReg, the colour head loss, the read-out losses and the WM-1 outcome losses on padded episodes.

        Inputs: boards and next_boards (B, T, 64); actions, goal_colours, counters and step_mask (B, T); levels, tiers, served_difficulty (B,);
        WM-1 batches add specials / next_specials (B, T, 64), cleared_masks (B, T, 64), residual_goal, terminal, mask_valid (B, T).
        Outputs: dict with loss, board_nll, counter_mse, goal_nll, mask_nll, special_nll, terminal_nll, goal_clamped, latent_mse, sigreg, colour_nll, board_logits,
        counter_mean, goal_logits, mask_logits, special_logits, terminal_logits, the opening predictions and the predicted / target latents.
        """
        if next_boards.shape != boards.shape or boards.ndim != 3 or boards.shape[-1] != N_CELLS:
            raise ValueError("boards and next_boards must have shape (batch, steps, 64)")
        batch_size, steps, _ = boards.shape
        for name, values in {"actions": actions, "goal_colours": goal_colours, "moves_left": moves_left, "goals_left": goals_left,
                             "next_moves_left": next_moves_left, "next_goals_left": next_goals_left, "step_mask": step_mask}.items():
            if values.shape != (batch_size, steps):
                raise ValueError(f"{name} must have shape {(batch_size, steps)}")
        rows = batch_size * steps
        flat = lambda values: values.reshape(rows)
        flat_boards = boards.reshape(rows, N_CELLS)
        flat_next = next_boards.reshape(rows, N_CELLS)
        flat_specials = None if specials is None else specials.reshape(rows, N_CELLS)
        flat_next_specials = None if next_specials is None else next_specials.reshape(rows, N_CELLS)
        context = self.encode_task(levels, tiers, served_difficulty)
        initial_board_logits, initial_counter_mean = self.initial_state_predictions(context)
        flat_context = context.unsqueeze(1).expand(-1, steps, -1).reshape(rows, -1)
        cells, global_latent = self.encode(flat_boards, flat(goal_colours), flat(moves_left), flat(goals_left), flat_specials)
        predicted_cells, predicted_global = self.predict(cells, global_latent, actions=flat(actions), task_context=flat_context, moves_left=flat(moves_left), goals_left=flat(goals_left))
        target_cells, target_global = self.encode_target(flat_next, flat(goal_colours), flat(next_moves_left), flat(next_goals_left), flat_next_specials)
        mask = step_mask.to(cells.dtype)
        valid = (step_mask if mask_valid is None else (step_mask & mask_valid.bool())).to(cells.dtype)
        flat_mask = step_mask.reshape(rows)
        transition_count = mask.sum().clamp_min(1.0)
        valid_count = valid.sum().clamp_min(1.0)
        state_count = transition_count + batch_size
        cell_errors = (predicted_cells - target_cells).square().mean(dim=2)
        if self.config.changed_cell_weight != 1.0:
            changed = (flat_boards != flat_next).to(cell_errors.dtype)
            weights = 1.0 + (self.config.changed_cell_weight - 1.0) * changed
            cell_term = (cell_errors * weights).sum(dim=1) / weights.sum(dim=1)
        else:
            cell_term = cell_errors.mean(dim=1)
        latent_rows = ((predicted_global - target_global).square().mean(dim=1) + cell_term).view(batch_size, steps)
        latent_mse = (latent_rows * mask).sum() / transition_count
        branches = [cells[flat_mask]] + ([target_cells[flat_mask]] if self.config.objective == "lewm" else [])
        generator = None if self.training else torch.Generator(device=cells.device).manual_seed(0)
        regulariser = self._sigreg(branches, generator)
        colour_nll = latent_mse.new_zeros(())
        if self.colour_head is not None:
            encoded = [(cells, flat_boards)] + ([(target_cells, flat_next)] if self.config.objective == "lewm" else [])
            terms = []
            for latents, targets in encoded:
                nll = F.cross_entropy(self.colour_head(latents).reshape(-1, self.config.n_colours), targets.long().reshape(-1), reduction="none").view(rows, N_CELLS).mean(dim=1)
                terms.append((nll.view(batch_size, steps) * mask).sum() / transition_count)
            colour_nll = torch.stack(terms).mean()
        readout_cells, readout_global = (predicted_cells.detach(), predicted_global.detach()) if self.config.readout_detach else (predicted_cells, predicted_global)
        known = self._known(flat_boards, flat_specials, flat(actions))
        zero = latent_mse.new_zeros(())
        mask_nll = special_nll = terminal_nll = zero
        mask_logits = special_logits = terminal_logits = None
        if self.config.goal_head == "mask":
            if cleared_masks is None:
                raise RuntimeError("goal_head='mask' needs cleared_masks in the batch")
            board_logits, _ = self.structured.board.teacher_forced(readout_cells, readout_global, flat_next)
            board_logits = board_logits.view(batch_size, steps, N_CELLS, -1)
            outcome = self._outcome(predicted_cells, predicted_global, flat_boards, flat(goal_colours), flat_specials, flat(moves_left), flat(goals_left), known)
            mask_logits = outcome["mask_logits"]
            target_mask = cleared_masks.reshape(rows, N_CELLS)
            mask_nll = (mask_bce(mask_logits, target_mask, outcome["round0_pre"]).view(batch_size, steps) * valid).sum() / valid_count
            s_goal_target = (target_mask.long() * outcome["goal_cells"].long()).sum(dim=-1)
            pb_nll, goal_logits, pb_clamped = derived_goal_nll(outcome["probabilities"], outcome["goal_cells"], s_goal_target, self.config.max_goal_delta)
            if self.config.residual_goal_min < 0:
                if residual_goal is None:
                    raise RuntimeError("a signed residual vocabulary needs residual_goal in the batch")
                residual_target = flat(residual_goal).long()
                censored = flat(next_goals_left).long() <= 0
            else:
                residual_target = flat(refill_goal_cleared).long() if refill_goal_cleared is not None else torch.zeros_like(s_goal_target)
                censored = None
            residual_nll = self.residual.nll(outcome["residual_logits"], residual_target, censored)
            goal_nll = (((pb_nll + residual_nll).view(batch_size, steps)) * valid).sum() / valid_count
            goal_clamped = ((pb_clamped | self.residual.clamped(residual_target)).view(batch_size, steps) & valid.bool()).sum()
            goal_logits = goal_logits.view(batch_size, steps, -1)
            expected_delta = outcome["expected_delta"].view(batch_size, steps)
        else:
            decoded = self.structured.teacher_forced(cell_features=readout_cells, global_features=readout_global, next_boards=flat_next, goals_left=flat(goals_left), next_goals_left=flat(next_goals_left))
            board_logits = decoded["board_logits"].view(batch_size, steps, N_CELLS, -1)
            goal_logits = decoded["goal_logits"].view(batch_size, steps, -1)
            goal_nll = (decoded["goal_nll"].view(batch_size, steps) * mask).sum() / transition_count
            goal_clamped = (decoded["goal_clamped"].view(batch_size, steps) & step_mask).sum()
            expected_delta = decoded["expected_delta"].view(batch_size, steps)
            outcome = self._outcome(predicted_cells, predicted_global, flat_boards, flat(goal_colours), flat_specials, flat(moves_left), flat(goals_left), known) if self.board_embedding is not None else {}
            if self.cleared_head_net is not None:
                if cleared_masks is None:
                    raise RuntimeError("the cleared-cell head needs cleared_masks in the batch")
                mask_logits = outcome["mask_logits"]
                mask_nll = (mask_bce(mask_logits, cleared_masks.reshape(rows, N_CELLS), outcome["round0_pre"]).view(batch_size, steps) * valid).sum() / valid_count
        if "terminal_logit" in outcome:
            target = terminal_target(flat(next_moves_left), flat(next_goals_left)) if terminal is None else flat(terminal).bool()
            terminal_logits = outcome["terminal_logit"].view(batch_size, steps)
            terminal_nll = (F.binary_cross_entropy_with_logits(terminal_logits, target.view(batch_size, steps).to(cells.dtype), reduction="none") * mask).sum() / transition_count
        if self.special_head_net is not None:
            if flat_next_specials is None or known is None:
                raise RuntimeError("the special-kind head needs specials and next_specials in the batch")
            special_logits = self.special_head_net.logits(readout_cells, flat_next, known["sources"])
            nll_rows, _ = self.special_head_net.nll(special_logits, flat_next_specials)
            special_nll = (nll_rows.view(batch_size, steps) * valid).sum() / valid_count
            special_logits = special_logits.view(batch_size, steps, N_CELLS, N_SPECIAL_KINDS)
        counter_mean = torch.stack(((moves_left.to(cells.dtype) - 1.0).clamp_min(0.0) / self.config.max_moves_left,
                                    (goals_left.to(cells.dtype) - expected_delta).clamp_min(0.0) / self.config.goals_scale), dim=-1)
        board_losses = F.cross_entropy(board_logits.reshape(-1, self.config.n_colours), next_boards.long().reshape(-1), reduction="none").view(batch_size, steps, N_CELLS).mean(dim=-1)
        initial_board_losses = F.cross_entropy(initial_board_logits.reshape(-1, self.config.n_colours), boards[:, 0].long().reshape(-1), reduction="none").view(batch_size, N_CELLS).mean(dim=-1)
        board_nll = ((board_losses * mask).sum() + initial_board_losses.sum()) / state_count
        counter_targets = torch.stack((next_moves_left.to(counter_mean.dtype) / self.config.max_moves_left, next_goals_left.to(counter_mean.dtype) / self.config.goals_scale), dim=-1)
        initial_counter_targets = torch.stack((moves_left[:, 0].to(initial_counter_mean.dtype) / self.config.max_moves_left, goals_left[:, 0].to(initial_counter_mean.dtype) / self.config.goals_scale), dim=-1)
        initial_counter_losses = (initial_counter_mean - initial_counter_targets).square().mean(dim=-1)
        counter_mse = (((counter_mean - counter_targets).square().mean(dim=-1) * mask).sum() + initial_counter_losses.sum()) / state_count
        loss = (self.config.latent_loss_weight * latent_mse + self.config.sigreg_weight * regulariser + self.config.colour_head_weight * colour_nll
                + board_nll + goal_nll + self.config.mask_loss_weight * mask_nll + self.config.special_loss_weight * special_nll + self.config.terminal_loss_weight * terminal_nll
                + initial_counter_losses.sum() / state_count)
        open_loop = latent_mse.new_zeros(())
        if self.config.open_loop_steps > 0 and steps > 1:
            open_loop = self._open_loop_terms(predicted_cells.view(batch_size, steps, N_CELLS, -1), predicted_global.view(batch_size, steps, -1), target_cells.view(batch_size, steps, N_CELLS, -1),
                                              target_global.view(batch_size, steps, -1), boards=boards, specials=specials, actions=actions, goal_colours=goal_colours, moves_left=moves_left, goals_left=goals_left,
                                              next_moves_left=next_moves_left, next_goals_left=next_goals_left, cleared_masks=cleared_masks, residual_goal=residual_goal, terminal=terminal,
                                              valid=valid.bool(), context=context)
            loss = loss + self.config.open_loop_weight * open_loop
        return {
            "loss": loss, "board_nll": board_nll, "counter_mse": counter_mse, "goal_nll": goal_nll, "mask_nll": mask_nll, "special_nll": special_nll, "terminal_nll": terminal_nll, "open_loop_loss": open_loop,
            "goal_clamped": goal_clamped, "latent_mse": latent_mse, "sigreg": regulariser, "colour_nll": colour_nll, "board_logits": board_logits, "counter_mean": counter_mean,
            "goal_logits": goal_logits, "mask_logits": mask_logits, "special_logits": special_logits, "terminal_logits": terminal_logits,
            "initial_board_logits": initial_board_logits, "initial_counter_mean": initial_counter_mean,
            "predicted_cells": predicted_cells, "target_cells": target_cells, "predicted_global": predicted_global, "target_global": target_global,
        }

    def _open_loop_terms(self, predicted_cells, predicted_global, target_cells, target_global, *, boards, specials, actions, goal_colours, moves_left, goals_left, next_moves_left, next_goals_left,
                         cleared_masks, residual_goal, terminal, valid, context) -> torch.Tensor:
        """Carried-latent losses: from the one-step prediction of every logged state, keep predicting with the logged actions for open_loop_steps further steps
        and score each carried latent (latent MSE against the encoded true state, cleared-mask BCE, derived-goal NLL, terminal BCE at that step).

        Inputs: predicted / target latents as (B, T, 64, C) and (B, T, C); the padded batch tensors; valid (B, T) bool; task context (B, task_context_size).
        Outputs: scalar (mean over carried steps of the summed per-step losses, each averaged over the valid rows).
        """
        batch_size, steps = valid.shape
        carried_cells, carried_global = predicted_cells, predicted_global
        total = predicted_cells.new_zeros(())
        used = 0
        for j in range(1, self.config.open_loop_steps + 1):
            if steps - j < 1:
                break
            n = batch_size * (steps - j)
            flat = lambda values: values[:, j:].reshape(n, *values.shape[2:])
            ok = valid[:, j:].reshape(n)  # step t+j valid implies steps t..t+j-1 valid (contiguous episodes)
            if not bool(ok.any()):
                break
            ctx = context.unsqueeze(1).expand(-1, steps - j, -1).reshape(n, -1)
            cells_in, global_in = carried_cells[:, : steps - j].reshape(n, N_CELLS, -1), carried_global[:, : steps - j].reshape(n, -1)
            carried_c, carried_g = self.predict(cells_in, global_in, actions=flat(actions), task_context=ctx, moves_left=flat(moves_left), goals_left=flat(goals_left))
            weight = ok.to(carried_c.dtype)
            count = weight.sum().clamp_min(1.0)
            latent_rows = (carried_c - flat(target_cells)).square().mean(dim=(1, 2)) + (carried_g - flat(target_global)).square().mean(dim=1)
            step_loss = (latent_rows * weight).sum() / count
            flat_boards = flat(boards)
            flat_specials = None if specials is None else flat(specials)
            known = self._known(flat_boards, flat_specials, flat(actions))
            outcome = self._outcome(carried_c, carried_g, flat_boards, flat(goal_colours), flat_specials, flat(moves_left), flat(goals_left), known)
            target_mask = flat(cleared_masks)
            step_loss = step_loss + self.config.mask_loss_weight * (mask_bce(outcome["mask_logits"], target_mask, outcome["round0_pre"]) * weight).sum() / count
            s_goal_target = (target_mask.long() * outcome["goal_cells"].long()).sum(dim=-1)
            pb_nll, _, _ = derived_goal_nll(outcome["probabilities"], outcome["goal_cells"], s_goal_target, self.config.max_goal_delta)
            residual_target = flat(residual_goal).long() if (self.config.residual_goal_min < 0 and residual_goal is not None) else torch.zeros_like(s_goal_target)
            residual_nll = self.residual.nll(outcome["residual_logits"], residual_target, flat(next_goals_left).long() <= 0)
            step_loss = step_loss + ((pb_nll + residual_nll) * weight).sum() / count
            if "terminal_logit" in outcome:
                target = terminal_target(flat(next_moves_left), flat(next_goals_left)) if terminal is None else flat(terminal).bool()
                step_loss = step_loss + self.config.terminal_loss_weight * (F.binary_cross_entropy_with_logits(outcome["terminal_logit"], target.to(carried_c.dtype), reduction="none") * weight).sum() / count
            total = total + step_loss
            used += 1
            carried_cells = carried_c.view(batch_size, steps - j, N_CELLS, -1)
            carried_global = carried_g.view(batch_size, steps - j, -1)
            # the carried tensors now start at origin step 0 and represent states t + j + 1; align them with the next iteration's [:, : steps - (j + 1)] slice
            carried_cells = torch.cat((carried_cells, carried_cells.new_zeros(batch_size, j, N_CELLS, carried_cells.shape[-1])), dim=1)
            carried_global = torch.cat((carried_global, carried_global.new_zeros(batch_size, j, carried_global.shape[-1])), dim=1)
        return total / max(used, 1)

    @torch.no_grad()
    def latent_diagnostics(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Measure collapse indicators of the latent on one padded batch.

        Inputs: a batch with the keys of objective().
        Outputs: dict with per_dim_std, across_board_variance, within_board_variance, change_salience, copy_ratio, cosine_cells, latent_mse and sigreg.
        """
        was_training = self.training
        self.eval()
        try:
            out = self.objective(**batch)
        finally:
            self.train(was_training)
        rows = batch["boards"].shape[0] * batch["boards"].shape[1]
        mask = batch["step_mask"].reshape(rows)
        channels = self.config.latent_channels
        specials = batch.get("specials")
        cells, _ = self.encode(batch["boards"].reshape(rows, N_CELLS), batch["goal_colours"].reshape(rows), batch["moves_left"].reshape(rows), batch["goals_left"].reshape(rows),
                               None if specials is None else specials.reshape(rows, N_CELLS))
        cells = cells[mask]
        predicted, target = out["predicted_cells"][mask], out["target_cells"][mask]
        changed = (batch["boards"].reshape(rows, N_CELLS) != batch["next_boards"].reshape(rows, N_CELLS))[mask]
        move = (cells - target).square().mean(dim=-1)
        permutation = torch.randperm(cells.shape[0], generator=torch.Generator().manual_seed(0)).to(cells.device)
        other = (cells - cells[permutation]).square().mean(dim=-1).mean() if cells.shape[0] > 1 else move.new_zeros(())
        to_current = (predicted - cells).square().mean(dim=-1)
        to_next = (predicted - target).square().mean(dim=-1)
        any_changed = bool(changed.any())
        flat = cells.reshape(-1, channels)
        return {
            "n_rows": int(mask.sum()),
            "per_dim_std": float(flat.std(dim=0, unbiased=False).mean()) if flat.shape[0] > 1 else 0.0,
            "across_board_variance": float(cells.var(dim=0, unbiased=False).mean()) if cells.shape[0] > 1 else 0.0,
            "within_board_variance": float(cells.var(dim=1, unbiased=False).mean()) if cells.numel() else 0.0,
            "change_salience": float(move[changed].mean() / other.clamp_min(1e-12)) if any_changed and float(other) > 0 else 0.0,
            "copy_ratio": float(to_current[changed].mean() / to_next[changed].mean().clamp_min(1e-12)) if any_changed else 0.0,
            "cosine_cells": float(F.cosine_similarity(predicted.reshape(-1, channels), target.reshape(-1, channels), dim=-1, eps=1e-8).mean()) if predicted.numel() else 0.0,
            "latent_mse": float(out["latent_mse"]),
            "sigreg": float(out["sigreg"]),
        }

    def rollout_begin(self, batch_size: int, task_context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return zero carried latents, replaced by the encoding of the opening board at the first observed step.

        Inputs: batch size; task_context (B, task_context_size).
        Outputs: dict with cells (B, 64, C) and global (B, C).
        """
        if batch_size < 1 or task_context.shape[0] != batch_size:
            raise ValueError("task_context must align with a positive batch size")
        channels = self.config.latent_channels
        return {"cells": task_context.new_zeros(batch_size, N_CELLS, channels), "global": task_context.new_zeros(batch_size, channels)}

    @torch.no_grad()
    def rollout_sample(
        self,
        state: dict[str, torch.Tensor],
        *,
        boards: torch.Tensor,
        goal_colours: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        actions: torch.Tensor,
        task_context: torch.Tensor,
        colour_support: torch.Tensor,
        specials: torch.Tensor | None = None,
        observe: bool = False,
        stochastic: bool = False,
        generator: torch.Generator | None = None,
        targets: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        """Advance one step in latent space and decode a match-free board, its specials and the goal decrement.

        Inputs: carried latents; boards / specials (N, 64) — the *decoded model board* in autonomous rollouts — used to (re-)encode when observe is set or
        rollout is "reencode", and always by the outcome heads (board embedding, goal cells, round-0 supports); goal_colours, moves_left, goals_left, actions,
        colour_support (N,); task_context (N, task_context_size); observe, stochastic, generator; optional targets as in TransitionTransformer.rollout_sample.
        Outputs: the next carried latents and a dict with boards, specials, goals_left, goal_delta, cleared_mask, terminal_logit, board_logits, goal_logits, retries,
        fallback, deadlocked (+ board_nll / special_nll / terminal_nll with targets).
        """
        if observe or self.config.rollout == "reencode" or not state:
            cells, global_latent = self.encode(boards, goal_colours, moves_left, goals_left, specials)
        else:
            cells, global_latent = state["cells"], state["global"]
        predicted_cells, predicted_global = self.predict(cells, global_latent, actions=actions, task_context=task_context, moves_left=moves_left, goals_left=goals_left)
        known = self._known(boards, specials, actions)
        out: dict[str, torch.Tensor | None] = {}
        if self.config.goal_head == "mask":
            generated = self.structured.board.generate(predicted_cells, predicted_global, colour_support=colour_support, stochastic=stochastic, generator=generator)
            outcome = self._outcome(predicted_cells, predicted_global, boards, goal_colours, specials, moves_left, goals_left, known)
            cleared, delta = sample_goal_delta(outcome["probabilities"], outcome["goal_cells"], outcome["residual_logits"], self.residual, stochastic=stochastic, generator=generator)
            out.update({"boards": generated["boards"], "goals_left": (goals_left.long() - delta).clamp_min(0), "goal_delta": delta, "cleared_mask": cleared,
                        "cleared_probabilities": outcome["probabilities"], "board_logits": None, "goal_logits": None, "retries": generated["retries"], "fallback": generated["fallback"],
                        "deadlocked": generated["deadlocked"], "terminal_logit": outcome.get("terminal_logit")})
        else:
            sampled = self.structured.sample(cell_features=predicted_cells, global_features=predicted_global, goals_left=goals_left, colour_support=colour_support, stochastic=stochastic, generator=generator)
            outcome = self._outcome(predicted_cells, predicted_global, boards, goal_colours, specials, moves_left, goals_left, known) if self.board_embedding is not None else {}
            out.update({"boards": sampled["boards"], "goals_left": sampled["goals_left"], "goal_delta": sampled["goal_delta"], "board_logits": None, "goal_logits": sampled["goal_logits"],
                        "retries": sampled["retries"], "fallback": sampled["fallback"], "deadlocked": sampled["deadlocked"], "terminal_logit": outcome.get("terminal_logit")})
        if self.special_head_net is not None and known is not None:
            logits = self.special_head_net.logits(predicted_cells, out["boards"], known["sources"])
            out["specials"] = sample_specials(logits, known["sources"], stochastic=stochastic, generator=generator)
        else:
            out["specials"] = torch.zeros_like(out["boards"])
        if targets is not None:
            rows = boards.shape[0]
            board_logits, _ = self.structured.board.teacher_forced(predicted_cells, predicted_global, targets["next_boards"].long())
            out["board_nll"] = F.cross_entropy(board_logits.reshape(-1, self.config.n_colours), targets["next_boards"].long().reshape(-1), reduction="none").view(rows, N_CELLS).mean(dim=-1)
            if self.special_head_net is not None and known is not None and "next_specials" in targets:
                logits = self.special_head_net.logits(predicted_cells, targets["next_boards"].long(), known["sources"])
                out["special_nll"], out["special_ambiguous_cells"] = self.special_head_net.nll(logits, targets["next_specials"].long())
            if out.get("terminal_logit") is not None and "next_moves_left" in targets:
                target = terminal_target(targets["next_moves_left"], targets["next_goals_left"]).to(predicted_cells.dtype)
                out["terminal_nll"] = F.binary_cross_entropy_with_logits(out["terminal_logit"], target, reduction="none")
        return {"cells": predicted_cells, "global": predicted_global}, out

    def parameter_counts(self) -> dict[str, int]:
        """Count trainable parameters by component; the frozen EMA target encoder is excluded (reported separately as frozen_target_encoder).

        Inputs: none.
        Outputs: dict of component counts plus total (asserted against the module's trainable parameters).
        """
        components: dict[str, nn.Module] = {
            "encoder": self.encoder,
            "predictor": nn.ModuleList([self.cell_in, self.row, self.col, self.action_role, self.action, self.task_token, self.global_in, self.moves, self.goals, self.token_type, self.trunk, self.cell_out, self.global_out]),
            "task_encoder": nn.ModuleList([self.level, self.tier, self.task_projection]),
            "opening_decoder": nn.ModuleList([self.initial_board_decoder, self.initial_counter_decoder]),
            "structured_decoder": self.structured,
        }
        for name, module in (("colour_head", self.colour_head), ("board_embedding", self.board_embedding), ("cleared_head", self.cleared_head_net), ("residual_head", self.residual_goal),
                             ("special_head", self.special_head_net), ("terminal_head", self.terminal_head_net)):
            if module is not None:
                components[name] = module
        counts = {name: sum(p.numel() for p in module.parameters() if p.requires_grad) for name, module in components.items()}
        counts["total"] = sum(counts.values())
        expected = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if counts["total"] != expected:
            raise RuntimeError(f"parameter_counts covers {counts['total']} of {expected} trainable parameters")
        return counts

    def frozen_parameter_count(self) -> int:
        """Parameters of the frozen EMA target encoder (0 without one)."""
        return 0 if self.target_encoder is None else sum(p.numel() for p in self.target_encoder.parameters())


__all__ = ["ENCODERS", "GOAL_HEADS", "OBJECTIVES", "ROLLOUTS", "SIGREG_ROWS", "TARGETS", "BoardEmbedding", "LatentWorldModel", "LatentWorldModelConfig"]
