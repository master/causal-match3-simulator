"""This is the implementation of the next-board (like next-token) transformer.
The main approach consists of having a Markov world model that attends over the 64 tiles of the current board plus action,
task and counter tokens. It has no latent state and no temporal attention, and player skill has no input path.

WM-1 additions (all off by default so earlier presets / checkpoints keep their meaning): special-kind tile embedding (``use_specials``),
special-kind head with the gravity admissibility mask (``special_head``), terminal head (``terminal_head``), round-0 support clamp of the
cleared-cell head (``round0_clamp``) and a signed residual vocabulary (``residual_goal_min`` < 0). See ``outcome.py`` for the shared
definitions of the outcome channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from match3_simulator.learned_model.tokens import ACTION_SLOTS, BOARD_HEIGHT, BOARD_WIDTH, CELL1_TOKEN, CELL2_TOKEN, N_CELLS
from match3_simulator.scm import LEVELS, TIER_MOVE_BUDGETS
from match3_simulator.world_modeling.decoder import DECODER_KINDS, SpecialKindHead, StructuredDecoderConfig, StructuredStateDecoder, decode_boards
from match3_simulator.world_modeling.outcome import ResidualSpec, TerminalHead, cleared_probabilities, derived_goal_nll, mask_bce, sample_goal_delta, terminal_target
from match3_simulator.world_modeling.wm1_mechanics import N_SPECIAL_KINDS, round0, sample_specials


@dataclass(frozen=True)
class TransitionTransformerConfig:
    """Sizes of the next-board transformer."""

    kind: str = "transformer"
    n_colours: int = 6
    n_levels: int = 3
    n_tiers: int = 3
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.0
    embedding_size: int = 32
    task_context_size: int = 32
    max_moves_left: int = max(TIER_MOVE_BUDGETS)
    goals_scale: float = 32.0
    decoder: str = "structured"
    decoder_hidden_size: int = 128
    max_goal_delta: int = 20
    cleared_head: bool = False
    goal_from_mask: bool = False
    mask_loss_weight: float = 1.0
    residual_goal_classes: int = 8
    # WM-1 heads and supports
    use_specials: bool = False
    special_head: bool = False
    terminal_head: bool = False
    round0_clamp: bool = False
    residual_goal_min: int = 0
    special_loss_weight: float = 1.0
    terminal_loss_weight: float = 1.0
    # Tier 2: k-step open-loop outcome consistency. For each horizon k the kernel is rolled k steps from a logged state with the logged
    # actions, feeding its own sampled boards back, and the accumulated expected goal decrement is regressed on the logged decrement.
    kstep_horizons: tuple[int, ...] = ()
    kstep_weight: float = 0.0
    kstep_episodes: int = 8

    def __post_init__(self) -> None:
        for name in ("n_colours", "n_levels", "n_tiers", "d_model", "n_layers", "n_heads", "embedding_size",
                     "task_context_size", "max_moves_left", "decoder_hidden_size", "max_goal_delta", "residual_goal_classes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.goal_from_mask and not (self.cleared_head and self.decoder == "structured"):
            raise ValueError("goal_from_mask requires cleared_head=True and the structured decoder")
        if self.round0_clamp and not self.cleared_head:
            raise ValueError("round0_clamp requires cleared_head=True")
        if self.special_head and self.decoder != "structured":
            raise ValueError("special_head requires the structured decoder")
        if min(self.mask_loss_weight, self.special_loss_weight, self.terminal_loss_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.kstep_weight < 0 or self.kstep_episodes < 1 or any(int(k) < 1 for k in self.kstep_horizons):
            raise ValueError("kstep_weight must be non-negative, kstep_episodes and every kstep horizon positive")
        if self.kstep_horizons and not self.goal_from_mask:
            raise ValueError("k-step outcome consistency requires goal_from_mask=True")
        object.__setattr__(self, "kstep_horizons", tuple(int(k) for k in self.kstep_horizons))
        if self.goals_scale <= 0:
            raise ValueError("goals_scale must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.decoder not in DECODER_KINDS:
            raise ValueError(f"decoder must be one of {DECODER_KINDS}")
        if self.kind != "transformer":
            raise ValueError("kind must be 'transformer'")

    @property
    def structured(self) -> bool:
        return self.decoder == "structured"

    @property
    def needs_mechanics(self) -> bool:
        return self.use_specials or self.special_head or self.round0_clamp


class TransitionTransformer(nn.Module):
    """Predict the next board and counters from tile tokens of the current board, the action and the task context."""

    kind = "transformer"

    def __init__(self, config: TransitionTransformerConfig = TransitionTransformerConfig()):
        super().__init__()
        self.config = config
        width = config.d_model
        embedding = config.embedding_size
        self.colour = nn.Embedding(config.n_colours, width)
        self.row = nn.Embedding(BOARD_HEIGHT, width)
        self.col = nn.Embedding(BOARD_WIDTH, width)
        self.is_goal = nn.Embedding(2, width)
        self.action_role = nn.Embedding(3, width)
        self.special: nn.Embedding | None = nn.Embedding(N_SPECIAL_KINDS, width) if config.use_specials else None
        self.action = nn.Embedding(ACTION_SLOTS, width)
        self.level = nn.Embedding(config.n_levels, embedding)
        self.tier = nn.Embedding(config.n_tiers, embedding)
        self.task_projection = nn.Sequential(
            nn.Linear(2 * embedding + 1, config.task_context_size), nn.GELU(), nn.Linear(config.task_context_size, config.task_context_size),
        )
        self.task_token = nn.Linear(config.task_context_size, width)
        self.moves = nn.Embedding(config.max_moves_left + 1, width)
        self.goals = nn.Linear(1, width)
        self.token_type = nn.Embedding(4, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=config.n_heads, dim_feedforward=4 * width, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=config.n_layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.board_head: nn.Module | None = None
        self.counter_head: nn.Module | None = None
        self.structured: StructuredStateDecoder | None = None
        if config.structured:
            self.structured = StructuredStateDecoder(StructuredDecoderConfig(
                n_colours=config.n_colours, cell_size=width, global_size=width,
                hidden_size=config.decoder_hidden_size, max_goal_delta=config.max_goal_delta, goal_head=not config.goal_from_mask,
            ))
        else:
            self.board_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, config.n_colours))
            self.counter_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        self.initial_board_decoder = nn.Sequential(nn.Linear(config.task_context_size, width), nn.GELU(), nn.Linear(width, N_CELLS * config.n_colours))
        self.initial_counter_decoder = nn.Sequential(nn.Linear(config.task_context_size, width), nn.GELU(), nn.Linear(width, 2))
        # cleared-cell head: per-cell Bernoulli "this pre-move tile is cleared during the cascade"; with goal_from_mask the goal decrement is the
        # Poisson-binomial count over goal-coloured cells plus a residual head for goal tiles cleared among refilled tiles (no pre-move cell)
        self.cleared_head_net: nn.Module | None = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)) if config.cleared_head else None
        self.residual_goal: nn.Module | None = (nn.Sequential(nn.Linear(width, config.decoder_hidden_size), nn.GELU(), nn.Linear(config.decoder_hidden_size, config.residual_goal_classes))
                                                if config.goal_from_mask else None)
        self.residual = ResidualSpec(config.residual_goal_min, config.residual_goal_classes)
        self.special_head_net: SpecialKindHead | None = SpecialKindHead(width, config.n_colours, config.decoder_hidden_size) if config.special_head else None
        self.terminal_head_net: TerminalHead | None = TerminalHead(width, config.decoder_hidden_size) if config.terminal_head else None
        positions = torch.arange(N_CELLS)
        self.register_buffer("cell_rows", positions // BOARD_WIDTH, persistent=False)
        self.register_buffer("cell_cols", positions % BOARD_WIDTH, persistent=False)
        self.register_buffer("cell1", torch.as_tensor(CELL1_TOKEN), persistent=False)
        self.register_buffer("cell2", torch.as_tensor(CELL2_TOKEN), persistent=False)
        self.register_buffer("level_colours", torch.tensor([LEVELS[i].n_colours if i < len(LEVELS) else config.n_colours for i in range(config.n_levels)]), persistent=False)

    def encode_task(self, levels: torch.Tensor, tiers: torch.Tensor, served_difficulty: torch.Tensor) -> torch.Tensor:
        """Embed the observed level, baseline tier and served difficulty.

        Inputs: levels (B,) long; tiers (B,) long; served_difficulty (B,) float.
        Outputs: task context (B, task_context_size).
        """
        if levels.ndim != 1 or tiers.shape != levels.shape or served_difficulty.shape != levels.shape:
            raise ValueError("levels, tiers and served_difficulty must be aligned vectors")
        return self.task_projection(torch.cat(
            (self.level(levels.long()), self.tier(tiers.long()), served_difficulty.to(self.level.weight.dtype).unsqueeze(-1)), dim=-1,
        ))

    def initial_state_predictions(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameterise the opening board and counters from the task context.

        Inputs: context (B, task_context_size).
        Outputs: board logits (B, 64, colours) and counter means (B, 2).
        """
        if context.ndim != 2 or context.shape[1] != self.config.task_context_size:
            raise ValueError("context has the wrong shape")
        return self.initial_board_decoder(context).view(context.shape[0], N_CELLS, self.config.n_colours), self.initial_counter_decoder(context)

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

    def features(
        self,
        *,
        boards: torch.Tensor,
        goal_colours: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        actions: torch.Tensor,
        task_context: torch.Tensor,
        specials: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk over tile, action, task and counter tokens.

        Inputs: boards (N, 64); goal_colours, moves_left, goals_left, actions (N,); task_context (N, task_context_size); specials (N, 64) or None (no stripes).
        Outputs: per-cell states (N, 64, d_model) and the counter-token state (N, d_model).
        """
        if boards.ndim != 2 or boards.shape[1] != N_CELLS:
            raise ValueError("boards must have shape (rows, 64)")
        rows = boards.shape[0]
        for name, values in {"goal_colours": goal_colours, "moves_left": moves_left, "goals_left": goals_left, "actions": actions}.items():
            if values.shape != (rows,):
                raise ValueError(f"{name} must have shape ({rows},)")
        if task_context.shape != (rows, self.config.task_context_size):
            raise ValueError("task_context has the wrong shape")
        boards = boards.long()
        tiles = (
            self.colour(boards) + self.row(self.cell_rows).unsqueeze(0) + self.col(self.cell_cols).unsqueeze(0)
            + self.is_goal((boards == goal_colours.long().unsqueeze(1)).long()) + self.action_role(self.action_roles(actions))
            + self.token_type.weight[0]
        )
        if self.special is not None:
            tiles = tiles + self.special(torch.zeros_like(boards) if specials is None else specials.long())
        action_token = (self.action(actions.long()) + self.token_type.weight[1]).unsqueeze(1)
        task_token = (self.task_token(task_context) + self.token_type.weight[2]).unsqueeze(1)
        counter_token = (
            self.moves(moves_left.long().clamp(0, self.config.max_moves_left))
            + self.goals((goals_left.to(tiles.dtype) / self.config.goals_scale).unsqueeze(1)) + self.token_type.weight[3]
        ).unsqueeze(1)
        hidden = self.trunk(torch.cat((tiles, action_token, task_token, counter_token), dim=1))
        return hidden[:, :N_CELLS], hidden[:, N_CELLS + 2]

    def forward(
        self,
        *,
        boards: torch.Tensor,
        goal_colours: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        actions: torch.Tensor,
        task_context: torch.Tensor,
        specials: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict independent next-board logits and counter means (independent decoder only).

        Inputs: boards (N, 64); goal_colours, moves_left, goals_left, actions (N,); task_context (N, task_context_size).
        Outputs: board logits (N, 64, colours) and counter means (N, 2).
        """
        if self.board_head is None or self.counter_head is None:
            raise RuntimeError("forward requires the independent decoder")
        cells, counter = self.features(boards=boards, goal_colours=goal_colours, moves_left=moves_left, goals_left=goals_left, actions=actions, task_context=task_context, specials=specials)
        return self.board_head(cells), self.counter_head(counter)

    def _known(self, boards: torch.Tensor, specials: torch.Tensor | None, actions: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Deterministic mechanics of the move (round-0 clear, stripe sources) when any WM-1 support needs them."""
        if not self.config.needs_mechanics:
            return None
        with torch.no_grad():
            return round0(boards.long(), torch.zeros_like(boards, dtype=torch.long) if specials is None else specials.long(), actions.long())

    def _outcome(self, cells: torch.Tensor, counter: torch.Tensor, boards: torch.Tensor, goal_colours: torch.Tensor, moves_left: torch.Tensor, goals_left: torch.Tensor,
                 known: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
        """Mask probabilities, residual logits, expected delta and terminal logit of flat rows (goal_from_mask path)."""
        assert self.cleared_head_net is not None and self.residual_goal is not None
        mask_logits = self.cleared_head_net(cells).squeeze(-1)
        round0_pre = known["cleared_pre"] if (known is not None and self.config.round0_clamp) else None
        probabilities = cleared_probabilities(mask_logits, round0_pre)
        goal_cells = boards.long() == goal_colours.long().unsqueeze(1)
        residual_logits = self.residual_goal(counter)
        expected_delta = (probabilities * goal_cells.to(probabilities.dtype)).sum(dim=-1) + self.residual.expected(residual_logits)
        out = {"mask_logits": mask_logits, "round0_pre": round0_pre, "probabilities": probabilities, "goal_cells": goal_cells, "residual_logits": residual_logits, "expected_delta": expected_delta}
        if self.terminal_head_net is not None:
            out["terminal_logit"] = self.terminal_head_net(counter, moves_left, goals_left, expected_delta, max_moves=self.config.max_moves_left, goals_scale=self.config.goals_scale)
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
        refill_goal_cleared: torch.Tensor | None = None,
        specials: torch.Tensor | None = None,
        next_specials: torch.Tensor | None = None,
        residual_goal: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
        mask_valid: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Score the logged next states of padded episodes with teacher forcing.

        Inputs: boards and next_boards (B, T, 64); actions, goal_colours, counters and step_mask (B, T); levels, tiers, served_difficulty (B,);
        with the cleared-cell head also cleared_masks (B, T, 64) and refill_goal_cleared / residual_goal (B, T); WM-1 batches add specials and
        next_specials (B, T, 64), terminal and mask_valid (B, T).
        Outputs: dict with loss, board_nll, counter_mse, goal_nll, mask_nll, special_nll, terminal_nll, goal_clamped, board_logits, counter_mean,
        goal_logits, mask_logits, special_logits, terminal_logits and the opening predictions.
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
        flat_specials = None if specials is None else specials.reshape(rows, N_CELLS)
        context = self.encode_task(levels, tiers, served_difficulty)
        initial_board_logits, initial_counter_mean = self.initial_state_predictions(context)
        cells, counter = self.features(
            boards=flat_boards, goal_colours=flat(goal_colours), moves_left=flat(moves_left), goals_left=flat(goals_left), actions=flat(actions),
            task_context=context.unsqueeze(1).expand(-1, steps, -1).reshape(rows, -1), specials=flat_specials,
        )
        known = self._known(flat_boards, flat_specials, flat(actions))
        mask = step_mask.to(cells.dtype)
        valid = (step_mask if mask_valid is None else (step_mask & mask_valid.bool())).to(cells.dtype)
        transition_count = mask.sum().clamp_min(1.0)
        valid_count = valid.sum().clamp_min(1.0)
        state_count = transition_count + batch_size
        zero = cells.new_zeros(())
        goal_nll = mask_nll = special_nll = terminal_nll = zero
        mask_logits = goal_logits = special_logits = terminal_logits = None
        goal_clamped = cells.new_zeros((), dtype=torch.long)
        if self.cleared_head_net is not None and not self.config.goal_from_mask:
            if cleared_masks is None:
                raise RuntimeError("the cleared-cell head needs cleared_masks in the batch")
            mask_logits = self.cleared_head_net(cells).squeeze(-1)
            round0_pre = known["cleared_pre"] if (known is not None and self.config.round0_clamp) else None
            mask_nll = (mask_bce(mask_logits, cleared_masks.reshape(rows, N_CELLS), round0_pre).view(batch_size, steps) * valid).sum() / valid_count
        if self.structured is not None and self.config.goal_from_mask:
            if cleared_masks is None:
                raise RuntimeError("the cleared-cell head needs cleared_masks in the batch (dataset built from episodes with transitions)")
            board_logits, _ = self.structured.board.teacher_forced(cells, counter, next_boards.reshape(rows, N_CELLS))
            board_logits = board_logits.view(batch_size, steps, N_CELLS, -1)
            outcome = self._outcome(cells, counter, flat_boards, flat(goal_colours), flat(moves_left), flat(goals_left), known)
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
                residual_target = (flat(refill_goal_cleared).long() if refill_goal_cleared is not None else torch.zeros_like(s_goal_target))
                censored = None
            residual_nll = self.residual.nll(outcome["residual_logits"], residual_target, censored)
            goal_nll = (((pb_nll + residual_nll).view(batch_size, steps)) * valid).sum() / valid_count
            goal_clamped = ((pb_clamped | self.residual.clamped(residual_target)).view(batch_size, steps) & valid.bool()).sum()
            goal_logits = goal_logits.view(batch_size, steps, -1)
            expected_delta = outcome["expected_delta"].view(batch_size, steps)
            counter_mean = torch.stack((
                (moves_left.to(cells.dtype) - 1.0).clamp_min(0.0) / self.config.max_moves_left,
                (goals_left.to(cells.dtype) - expected_delta).clamp_min(0.0) / self.config.goals_scale,
            ), dim=-1)
            if "terminal_logit" in outcome:
                target = terminal_target(flat(next_moves_left), flat(next_goals_left)) if terminal is None else flat(terminal).bool()
                terminal_logits = outcome["terminal_logit"].view(batch_size, steps)
                terminal_nll = (F.binary_cross_entropy_with_logits(terminal_logits, target.view(batch_size, steps).to(cells.dtype), reduction="none") * mask).sum() / transition_count
        elif self.structured is not None:
            decoded = self.structured.teacher_forced(
                cell_features=cells, global_features=counter, next_boards=next_boards.reshape(rows, N_CELLS),
                goals_left=goals_left.reshape(-1), next_goals_left=next_goals_left.reshape(-1),
            )
            board_logits = decoded["board_logits"].view(batch_size, steps, N_CELLS, -1)
            goal_logits = decoded["goal_logits"].view(batch_size, steps, -1)
            goal_nll = (decoded["goal_nll"].view(batch_size, steps) * mask).sum() / transition_count
            goal_clamped = (decoded["goal_clamped"].view(batch_size, steps) & step_mask).sum()
            expected_delta = decoded["expected_delta"].view(batch_size, steps)
            counter_mean = torch.stack((
                (moves_left.to(cells.dtype) - 1.0).clamp_min(0.0) / self.config.max_moves_left,
                (goals_left.to(cells.dtype) - expected_delta).clamp_min(0.0) / self.config.goals_scale,
            ), dim=-1)
        else:
            assert self.board_head is not None and self.counter_head is not None
            board_logits = self.board_head(cells).view(batch_size, steps, N_CELLS, -1)
            counter_mean = self.counter_head(counter).view(batch_size, steps, 2)
        if self.special_head_net is not None:
            if next_specials is None or known is None:
                raise RuntimeError("the special-kind head needs specials and next_specials in the batch")
            special_logits = self.special_head_net.logits(cells, next_boards.reshape(rows, N_CELLS), known["sources"])
            nll_rows, _ = self.special_head_net.nll(special_logits, next_specials.reshape(rows, N_CELLS))
            special_nll = (nll_rows.view(batch_size, steps) * valid).sum() / valid_count
            special_logits = special_logits.view(batch_size, steps, N_CELLS, N_SPECIAL_KINDS)
        board_losses = F.cross_entropy(board_logits.reshape(-1, self.config.n_colours), next_boards.long().reshape(-1), reduction="none").view(batch_size, steps, N_CELLS).mean(dim=-1)
        initial_board_losses = F.cross_entropy(initial_board_logits.reshape(-1, self.config.n_colours), boards[:, 0].long().reshape(-1), reduction="none").view(batch_size, N_CELLS).mean(dim=-1)
        board_nll = ((board_losses * mask).sum() + initial_board_losses.sum()) / state_count
        counter_targets = torch.stack((next_moves_left.to(counter_mean.dtype) / self.config.max_moves_left, next_goals_left.to(counter_mean.dtype) / self.config.goals_scale), dim=-1)
        counter_losses = (counter_mean - counter_targets).square().mean(dim=-1)
        initial_counter_targets = torch.stack((moves_left[:, 0].to(initial_counter_mean.dtype) / self.config.max_moves_left, goals_left[:, 0].to(initial_counter_mean.dtype) / self.config.goals_scale), dim=-1)
        initial_counter_losses = (initial_counter_mean - initial_counter_targets).square().mean(dim=-1)
        counter_mse = ((counter_losses * mask).sum() + initial_counter_losses.sum()) / state_count
        if self.structured is not None:
            loss = (board_nll + goal_nll + self.config.mask_loss_weight * mask_nll + self.config.special_loss_weight * special_nll
                    + self.config.terminal_loss_weight * terminal_nll + initial_counter_losses.sum() / state_count)
        else:
            loss = board_nll + counter_mse + self.config.mask_loss_weight * mask_nll
        kstep_mse = cells.new_zeros(())
        if self.config.kstep_horizons:
            kstep = self.kstep_consistency(
                boards=boards, actions=actions, goal_colours=goal_colours, moves_left=moves_left, goals_left=goals_left, next_goals_left=next_goals_left,
                step_mask=step_mask, levels=levels, task_context=context,
            )
            kstep_mse = kstep["kstep_mse"]
            loss = loss + self.config.kstep_weight * kstep_mse
        return {
            "loss": loss, "board_nll": board_nll, "counter_mse": counter_mse, "goal_nll": goal_nll, "mask_nll": mask_nll, "special_nll": special_nll, "terminal_nll": terminal_nll,
            "kstep_mse": kstep_mse, "goal_clamped": goal_clamped, "board_logits": board_logits, "counter_mean": counter_mean, "goal_logits": goal_logits, "mask_logits": mask_logits,
            "special_logits": special_logits, "terminal_logits": terminal_logits, "initial_board_logits": initial_board_logits, "initial_counter_mean": initial_counter_mean,
        }

    def kstep_consistency(
        self,
        *,
        boards: torch.Tensor,
        actions: torch.Tensor,
        goal_colours: torch.Tensor,
        moves_left: torch.Tensor,
        goals_left: torch.Tensor,
        next_goals_left: torch.Tensor,
        step_mask: torch.Tensor,
        levels: torch.Tensor,
        task_context: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Open-loop k-step outcome consistency on a sub-batch of logged episodes (Tier 2; specials are treated as absent).

        Inputs: boards (B, T, 64); actions, goal_colours, moves_left, goals_left, next_goals_left, step_mask (B, T); levels (B,); task_context (B, task_context_size).
        Outputs: dict with kstep_mse (scalar, mean over horizons; zero when no episode is long enough) and kstep_rows (long, rows used).
        """
        assert self.structured is not None and self.cleared_head_net is not None and self.residual_goal is not None
        device = boards.device
        lengths = step_mask.long().sum(dim=1)
        losses = []
        rows_used = 0
        for k in self.config.kstep_horizons:
            eligible = torch.nonzero(lengths >= k).squeeze(1)
            if eligible.numel() == 0:
                continue
            if eligible.numel() > self.config.kstep_episodes:
                pick = torch.randperm(eligible.numel(), generator=generator, device="cpu")[: self.config.kstep_episodes].to(device)
                eligible = eligible[pick]
            n = eligible.numel()
            span = (lengths[eligible] - k + 1).to(torch.float32)
            start = (torch.rand(n, generator=generator, device="cpu").to(device) * span).long().clamp_max((lengths[eligible] - k).clamp_min(0))
            end = start + k - 1
            target = (goals_left[eligible, start].float() - next_goals_left[eligible, end].float()).clamp_min(0.0) / k
            board = boards[eligible, start].long()
            goals = goals_left[eligible, start].long()
            moves = moves_left[eligible, start].long()
            colours = goal_colours[eligible, start].long()
            context = task_context[eligible]
            support = self.level_colours[levels[eligible].long()]
            cumulative = boards.new_zeros(n, dtype=torch.float32)
            for j in range(k):
                action = actions[eligible, start + j].long()
                cells, counter = self.features(boards=board, goal_colours=colours, moves_left=moves, goals_left=goals, actions=action, task_context=context)
                outcome = self._outcome(cells, counter, board, colours, moves, goals, self._known(board, None, action))
                cumulative = cumulative + outcome["expected_delta"]
                if j + 1 < k:
                    with torch.no_grad():
                        generated = self.structured.board.generate(cells.detach(), counter.detach(), colour_support=support, stochastic=True, generator=generator)
                        _, delta = sample_goal_delta(outcome["probabilities"].detach(), outcome["goal_cells"], outcome["residual_logits"].detach(), self.residual, stochastic=True, generator=generator)
                        board = generated["boards"].long()
                        goals = (goals - delta).clamp_min(0)
                        moves = (moves - 1).clamp_min(0)
            losses.append(((cumulative / k - target) ** 2).mean())
            rows_used += n
        if not losses:
            return {"kstep_mse": boards.new_zeros((), dtype=torch.float32), "kstep_rows": torch.zeros((), dtype=torch.long, device=device)}
        return {"kstep_mse": torch.stack(losses).mean(), "kstep_rows": torch.tensor(rows_used, device=device)}

    def rollout_begin(self, batch_size: int, task_context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the empty carried state of a Markov model.

        Inputs: batch size; task_context (B, task_context_size).
        Outputs: an empty dict.
        """
        if batch_size < 1 or task_context.shape[0] != batch_size:
            raise ValueError("task_context must align with a positive batch size")
        return {}

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
        """Generate the next board, specials and goals-left from the current generated board.

        Inputs: carried state (ignored); boards (N, 64); goal_colours, moves_left, goals_left, actions, colour_support (N,); task_context (N, task_context_size);
        specials (N, 64) or None; observe, stochastic, generator; optional targets {next_boards, next_specials, next_moves_left, next_goals_left} to score
        (teacher-forced NLL of the logged next state from this model state).
        Outputs: the unchanged state and a dict with boards, specials, goals_left, goal_delta, cleared_mask, terminal_logit, board_logits, goal_logits, retries, fallback,
        deadlocked and, with targets, board_nll (N,) per-cell mean, special_nll (N,), terminal_nll (N,).
        """
        rows = boards.shape[0]
        cells, counter = self.features(boards=boards, goal_colours=goal_colours, moves_left=moves_left, goals_left=goals_left, actions=actions, task_context=task_context, specials=specials)
        if self.structured is None:
            assert self.board_head is not None and self.counter_head is not None
            board_logits = self.board_head(cells)
            counter_mean = self.counter_head(counter)
            next_boards = decode_boards(board_logits, stochastic=stochastic, generator=generator, colour_support=colour_support)
            next_goals = torch.minimum(goals_left.long(), torch.round(counter_mean[:, 1] * self.config.goals_scale).long().clamp_min(0))
            return state, {
                "boards": next_boards, "specials": torch.zeros_like(next_boards), "goals_left": next_goals, "board_logits": board_logits, "goal_logits": None,
                "retries": torch.zeros(rows, dtype=torch.long, device=boards.device),
                "fallback": torch.zeros(rows, dtype=torch.bool, device=boards.device), "deadlocked": None, "terminal_logit": None,
            }
        known = self._known(boards, specials, actions)
        out: dict[str, torch.Tensor | None] = {}
        if self.config.goal_from_mask:
            generated = self.structured.board.generate(cells, counter, colour_support=colour_support, stochastic=stochastic, generator=generator)
            outcome = self._outcome(cells, counter, boards, goal_colours, moves_left, goals_left, known)
            cleared, delta = sample_goal_delta(outcome["probabilities"], outcome["goal_cells"], outcome["residual_logits"], self.residual, stochastic=stochastic, generator=generator)
            out.update({"boards": generated["boards"], "goals_left": (goals_left.long() - delta).clamp_min(0), "goal_delta": delta, "cleared_mask": cleared,
                        "cleared_probabilities": outcome["probabilities"], "board_logits": None, "goal_logits": None, "retries": generated["retries"], "fallback": generated["fallback"],
                        "deadlocked": generated["deadlocked"], "terminal_logit": outcome.get("terminal_logit")})
        else:
            sampled = self.structured.sample(cell_features=cells, global_features=counter, goals_left=goals_left, colour_support=colour_support, stochastic=stochastic, generator=generator)
            out.update({"boards": sampled["boards"], "goals_left": sampled["goals_left"], "goal_delta": sampled["goal_delta"], "board_logits": None, "goal_logits": sampled["goal_logits"],
                        "retries": sampled["retries"], "fallback": sampled["fallback"], "deadlocked": sampled["deadlocked"], "terminal_logit": None})
        if self.special_head_net is not None and known is not None:
            logits = self.special_head_net.logits(cells, out["boards"], known["sources"])
            out["specials"] = sample_specials(logits, known["sources"], stochastic=stochastic, generator=generator)
        else:
            out["specials"] = torch.zeros_like(out["boards"])
        if targets is not None:
            board_logits, _ = self.structured.board.teacher_forced(cells, counter, targets["next_boards"].long())
            out["board_nll"] = F.cross_entropy(board_logits.reshape(-1, self.config.n_colours), targets["next_boards"].long().reshape(-1), reduction="none").view(rows, N_CELLS).mean(dim=-1)
            if self.special_head_net is not None and known is not None and "next_specials" in targets:
                logits = self.special_head_net.logits(cells, targets["next_boards"].long(), known["sources"])
                out["special_nll"], out["special_ambiguous_cells"] = self.special_head_net.nll(logits, targets["next_specials"].long())
            if out.get("terminal_logit") is not None and "next_moves_left" in targets:
                target = terminal_target(targets["next_moves_left"], targets["next_goals_left"]).to(cells.dtype)
                out["terminal_nll"] = F.binary_cross_entropy_with_logits(out["terminal_logit"], target, reduction="none")
        return state, out

    def parameter_counts(self) -> dict[str, int]:
        """Count trainable parameters by component (every parameter belongs to exactly one component; total is asserted against the module).

        Inputs: none.
        Outputs: dict of component counts plus total.
        """
        components: dict[str, nn.Module] = {
            "tile_embeddings": nn.ModuleList([m for m in (self.colour, self.row, self.col, self.is_goal, self.action_role, self.token_type, self.special) if m is not None]),
            "action_embedding": self.action,
            "task_encoder": nn.ModuleList([self.level, self.tier, self.task_projection, self.task_token]),
            "counter_tokens": nn.ModuleList([self.moves, self.goals]),
            "trunk": self.trunk,
            "opening_decoder": nn.ModuleList([self.initial_board_decoder, self.initial_counter_decoder]),
        }
        if self.structured is not None:
            components["structured_decoder"] = self.structured
        else:
            components["board_head"] = self.board_head
            components["counter_head"] = self.counter_head
        for name, module in (("cleared_head", self.cleared_head_net), ("residual_head", self.residual_goal), ("special_head", self.special_head_net), ("terminal_head", self.terminal_head_net)):
            if module is not None:
                components[name] = module
        counts = {name: sum(p.numel() for p in module.parameters() if p.requires_grad) for name, module in components.items()}
        counts["total"] = sum(counts.values())
        expected = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if counts["total"] != expected:
            raise RuntimeError(f"parameter_counts covers {counts['total']} of {expected} trainable parameters")
        return counts


__all__ = ["TransitionTransformer", "TransitionTransformerConfig"]
