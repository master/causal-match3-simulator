"""Structured next-state read-out shared by every world model: an autoregressive match-free board decoder and a categorical goal-decrement head.
Generated boards never contain an unresolved run of three and the goal counter never goes below zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from match3_simulator.learned_model.tokens import (
    ACTION_SLOTS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CELL1_TOKEN,
    CELL2_TOKEN,
    IN_BOUNDS,
    N_CELLS,
)

DECODER_KINDS: tuple[str, ...] = ("independent", "structured")


def _run_mask(grid: torch.Tensor) -> torch.Tensor:
    """Mark every cell that lies in a horizontal or vertical run of at least three equal colours.

    Inputs: grid (..., 8, 8) of colour indices.
    Outputs: boolean tensor of the same shape.
    """
    matched = torch.zeros_like(grid, dtype=torch.bool)
    horizontal = (grid[..., :, :-2] == grid[..., :, 1:-1]) & (grid[..., :, 1:-1] == grid[..., :, 2:])
    matched[..., :, :-2] |= horizontal
    matched[..., :, 1:-1] |= horizontal
    matched[..., :, 2:] |= horizontal
    vertical = (grid[..., :-2, :] == grid[..., 1:-1, :]) & (grid[..., 1:-1, :] == grid[..., 2:, :])
    matched[..., :-2, :] |= vertical
    matched[..., 1:-1, :] |= vertical
    matched[..., 2:, :] |= vertical
    return matched


def torch_legal_mask(boards: torch.Tensor) -> torch.Tensor:
    """Compute the exact 128-slot legal-swap mask of a batch of flattened boards.

    Inputs: boards (batch, 64) of colour indices.
    Outputs: boolean tensor (batch, 128), true where the swap is in bounds, changes colours and creates a run.
    """
    if boards.ndim != 2 or boards.shape[1] != N_CELLS:
        raise ValueError("boards must have shape (batch, 64)")
    device = boards.device
    batch_size = boards.shape[0]
    cell1 = torch.as_tensor(CELL1_TOKEN, device=device)
    cell2 = torch.as_tensor(CELL2_TOKEN, device=device)
    in_bounds = torch.as_tensor(IN_BOUNDS, device=device)
    swapped = boards.unsqueeze(1).expand(-1, ACTION_SLOTS, -1).clone()
    first_index = cell1.view(1, -1, 1).expand(batch_size, -1, -1)
    second_index = cell2.view(1, -1, 1).expand(batch_size, -1, -1)
    first_value = swapped.gather(2, first_index)
    second_value = swapped.gather(2, second_index)
    swapped.scatter_(2, first_index, second_value)
    swapped.scatter_(2, second_index, first_value)
    matched = _run_mask(swapped.view(batch_size, ACTION_SLOTS, BOARD_HEIGHT, BOARD_WIDTH)).view(batch_size, ACTION_SLOTS, N_CELLS)
    moved = matched.gather(2, first_index).squeeze(2) | matched.gather(2, second_index).squeeze(2)
    different = first_value.squeeze(2) != second_value.squeeze(2)
    return moved & different & in_bounds.unsqueeze(0)


def support_mask(colour_support: torch.Tensor, n_colours: int) -> torch.Tensor:
    """Mark the colours admissible for each row given its level's colour count.

    Inputs: colour_support (batch,) long; total number of colour classes.
    Outputs: boolean tensor (batch, n_colours).
    """
    colours = torch.arange(n_colours, device=colour_support.device)
    return colours.unsqueeze(0) < colour_support.long().unsqueeze(1)


def decode_boards(
    logits: torch.Tensor,
    *,
    stochastic: bool,
    generator: torch.Generator | None,
    colour_support: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode independent per-cell colour logits by sampling or argmax within the colour support.

    Inputs: logits (batch, 64, colours); stochastic flag; optional generator; optional colour_support (batch,).
    Outputs: boards (batch, 64) long.
    """
    if colour_support is not None:
        mask = support_mask(colour_support, logits.shape[-1]).unsqueeze(1)
        logits = logits.masked_fill(~mask, float("-inf"))
    if stochastic:
        flat = logits.reshape(-1, logits.shape[-1]).softmax(dim=-1)
        colours = torch.multinomial(flat, 1, generator=generator).view(logits.shape[:-1])
    else:
        colours = logits.argmax(dim=-1)
    return colours.long()


def run_completion_mask(partial: torch.Tensor, cell: int, n_colours: int) -> torch.Tensor:
    """Mark the colours that would complete a run of three at one cell given the cells generated before it.

    Inputs: partial boards (batch, 64) with earlier cells filled; row-major cell index; number of colours.
    Outputs: boolean tensor (batch, n_colours), true for forbidden colours.
    """
    row, col = divmod(cell, BOARD_WIDTH)
    batch_size = partial.shape[0]
    forbidden = torch.zeros((batch_size, n_colours), dtype=torch.bool, device=partial.device)
    index = torch.arange(batch_size, device=partial.device)
    if col >= 2:
        left1, left2 = partial[:, cell - 1], partial[:, cell - 2]
        forbidden[index, left1.clamp_min(0)] |= left1 == left2
    if row >= 2:
        up1, up2 = partial[:, cell - BOARD_WIDTH], partial[:, cell - 2 * BOARD_WIDTH]
        forbidden[index, up1.clamp_min(0)] |= up1 == up2
    return forbidden


def constrained_sequential_decode(
    logits: torch.Tensor,
    *,
    colour_support: torch.Tensor,
    stochastic: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Decode independent per-cell logits cell by cell under the no-run and colour-support masks.

    Inputs: logits (batch, 64, colours); colour_support (batch,); stochastic flag; optional generator.
    Outputs: match-free boards (batch, 64) long.
    """
    if logits.ndim != 3 or logits.shape[1] != N_CELLS:
        raise ValueError("logits must have shape (batch, 64, colours)")
    batch_size, _, n_colours = logits.shape
    support = support_mask(colour_support, n_colours)
    boards = torch.full((batch_size, N_CELLS), -1, dtype=torch.long, device=logits.device)
    for cell in range(N_CELLS):
        allowed = support & ~run_completion_mask(boards, cell, n_colours)
        cell_logits = logits[:, cell].masked_fill(~allowed, float("-inf"))
        if stochastic:
            colours = torch.multinomial(cell_logits.softmax(dim=-1), 1, generator=generator).squeeze(1)
        else:
            colours = cell_logits.argmax(dim=-1)
        boards[:, cell] = colours
    return boards


def masked_uniform_boards(colour_support: torch.Tensor, n_colours: int, *, generator: torch.Generator | None) -> torch.Tensor:
    """Draw uniform match-free boards on the colour support, the documented fallback when generation deadlocks.

    Inputs: colour_support (batch,); number of colours; optional generator.
    Outputs: boards (batch, 64) long.
    """
    logits = torch.zeros((colour_support.shape[0], N_CELLS, n_colours), device=colour_support.device)
    return constrained_sequential_decode(logits, colour_support=colour_support, stochastic=True, generator=generator)


@dataclass(frozen=True)
class StructuredDecoderConfig:
    """Sizes shared by the board decoder and the goal head. ``goal_head=False`` omits the categorical goal head (WM-1 mask-derived goal)."""

    n_colours: int = 6
    cell_size: int = 32
    global_size: int = 192
    hidden_size: int = 128
    colour_embedding_size: int = 16
    max_goal_delta: int = 20
    max_retries: int = 4
    goal_head: bool = True

    def __post_init__(self) -> None:
        for name in ("n_colours", "cell_size", "global_size", "hidden_size", "colour_embedding_size", "max_goal_delta"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


class SpecialKindHead(nn.Module):
    """Next-board special kind per cell from the cell feature and the embedding of that cell's decoded next colour, under the admissibility
    mask of ``wm1_mechanics.admissible_special_kinds`` (a kind is possible only where a source stripe of that kind sits in the same column at or
    above the cell). Teacher forcing uses the logged next colour; rollouts use the generated colour and ``wm1_mechanics.sample_specials`` so no
    stripe is invented, changes kind or column, or multiplies."""

    def __init__(self, cell_size: int, n_colours: int, hidden_size: int, colour_embedding_size: int = 16):
        super().__init__()
        self.next_colour = nn.Embedding(n_colours, colour_embedding_size)
        self.network = nn.Sequential(nn.Linear(cell_size + colour_embedding_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 3))

    def logits(self, cell_features: torch.Tensor, next_boards: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        """Inputs: cell_features (rows, 64, cell_size); next colours (rows, 64); sources (rows, 64) special grid that may survive.
        Outputs: admissibility-masked logits (rows, 64, 3)."""
        from match3_simulator.world_modeling.wm1_mechanics import admissible_special_kinds

        raw = self.network(torch.cat((cell_features, self.next_colour(next_boards.long())), dim=-1))
        return raw.masked_fill(~admissible_special_kinds(sources), float("-inf"))

    @staticmethod
    def nll(logits: torch.Tensor, next_specials: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy averaged over the *ambiguous* cells of each row (cells with more than one admissible kind; the others are exact).

        Inputs: logits (rows, 64, 3); next_specials (rows, 64). Outputs: per-row NLL (rows,) and the ambiguous-cell count (rows,)."""
        ambiguous = torch.isfinite(logits).sum(dim=-1) > 1
        safe = logits.masked_fill(~torch.isfinite(logits), -1e4)
        losses = F.cross_entropy(safe.reshape(-1, 3), next_specials.long().reshape(-1), reduction="none").view(next_specials.shape)
        count = ambiguous.sum(dim=-1)
        return (losses * ambiguous.to(losses.dtype)).sum(dim=-1) / count.clamp_min(1).to(losses.dtype), count


class AutoregressiveBoardDecoder(nn.Module):
    """Row-major GRU over the 64 cells, conditioned on per-cell features and the previous colour."""

    def __init__(self, config: StructuredDecoderConfig):
        super().__init__()
        self.config = config
        self.previous_colour = nn.Embedding(config.n_colours + 1, config.colour_embedding_size)
        self.initial_hidden = nn.Linear(config.global_size, config.hidden_size)
        self.gru = nn.GRU(config.cell_size + config.colour_embedding_size, config.hidden_size, batch_first=True)
        self.output = nn.Linear(config.hidden_size, config.n_colours)

    @property
    def bos(self) -> int:
        return self.config.n_colours

    def _check(self, cell_features: torch.Tensor, global_features: torch.Tensor) -> int:
        """Validate feature shapes and return the row count.

        Inputs: cell_features (rows, 64, cell_size); global_features (rows, global_size).
        Outputs: number of rows.
        """
        if cell_features.ndim != 3 or cell_features.shape[1:] != (N_CELLS, self.config.cell_size):
            raise ValueError("cell_features must have shape (rows, 64, cell_size)")
        rows = cell_features.shape[0]
        if global_features.shape != (rows, self.config.global_size):
            raise ValueError("global_features must have shape (rows, global_size)")
        return rows

    def teacher_forced(self, cell_features: torch.Tensor, global_features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the true next board with each cell conditioned on the true colour of the previous cell.

        Inputs: cell_features (rows, 64, cell_size); global_features (rows, global_size); targets (rows, 64).
        Outputs: logits (rows, 64, colours) and the final GRU state (rows, hidden).
        """
        rows = self._check(cell_features, global_features)
        if targets.shape != (rows, N_CELLS):
            raise ValueError("targets must have shape (rows, 64)")
        shifted = torch.cat((torch.full((rows, 1), self.bos, dtype=torch.long, device=targets.device), targets.long()[:, :-1]), dim=1)
        inputs = torch.cat((cell_features, self.previous_colour(shifted)), dim=-1)
        h0 = torch.tanh(self.initial_hidden(global_features)).unsqueeze(0)
        outputs, final = self.gru(inputs, h0)
        return self.output(outputs), final.squeeze(0)

    def _generate_once(
        self,
        cell_features: torch.Tensor,
        global_features: torch.Tensor,
        colour_support: torch.Tensor,
        *,
        stochastic: bool,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one match-free board per row cell by cell.

        Inputs: cell_features (rows, 64, cell_size); global_features (rows, global_size); colour_support (rows,); stochastic flag; generator.
        Outputs: boards (rows, 64) and the final GRU state (rows, hidden).
        """
        rows = cell_features.shape[0]
        n_colours = self.config.n_colours
        device = cell_features.device
        support = support_mask(colour_support, n_colours)
        boards = torch.full((rows, N_CELLS), -1, dtype=torch.long, device=device)
        previous = torch.full((rows,), self.bos, dtype=torch.long, device=device)
        hidden = torch.tanh(self.initial_hidden(global_features)).unsqueeze(0)
        for cell in range(N_CELLS):
            step_input = torch.cat((cell_features[:, cell], self.previous_colour(previous)), dim=-1).unsqueeze(1)
            output, hidden = self.gru(step_input, hidden)
            logits = self.output(output.squeeze(1)).masked_fill(~(support & ~run_completion_mask(boards, cell, n_colours)), float("-inf"))
            colours = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(1) if stochastic else logits.argmax(dim=-1)
            boards[:, cell] = colours
            previous = colours
        return boards, hidden.squeeze(0)

    @torch.no_grad()
    def generate(
        self,
        cell_features: torch.Tensor,
        global_features: torch.Tensor,
        *,
        colour_support: torch.Tensor,
        stochastic: bool,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        """Generate match-free boards with bounded resampling of deadlocked rows and a uniform fallback.

        Inputs: cell_features (rows, 64, cell_size); global_features (rows, global_size); colour_support (rows,); stochastic flag; generator.
        Outputs: dict with boards (rows, 64), final_hidden (rows, hidden), retries (rows,), fallback (rows,) and deadlocked (rows,).
        """
        rows = self._check(cell_features, global_features)
        if colour_support.shape != (rows,):
            raise ValueError("colour_support must have shape (rows,)")
        boards, final_hidden = self._generate_once(cell_features, global_features, colour_support, stochastic=stochastic, generator=generator)
        retries = torch.zeros(rows, dtype=torch.long, device=boards.device)
        fallback = torch.zeros(rows, dtype=torch.bool, device=boards.device)
        deadlocked = ~torch_legal_mask(boards).any(dim=1)
        for _ in range(self.config.max_retries):
            if not bool(deadlocked.any()):
                break
            index = torch.nonzero(deadlocked).squeeze(1)
            regenerated, regenerated_hidden = self._generate_once(
                cell_features[index], global_features[index], colour_support[index], stochastic=True, generator=generator,
            )
            boards[index] = regenerated
            final_hidden[index] = regenerated_hidden
            retries[index] += 1
            deadlocked = ~torch_legal_mask(boards).any(dim=1)
        if bool(deadlocked.any()):
            index = torch.nonzero(deadlocked).squeeze(1)
            for _ in range(max(1, self.config.max_retries)):
                shuffled = masked_uniform_boards(colour_support[index], self.config.n_colours, generator=generator)
                boards[index] = shuffled
                fallback[index] = True
                still = ~torch_legal_mask(shuffled).any(dim=1)
                if not bool(still.any()):
                    break
                index = index[still]
            deadlocked = ~torch_legal_mask(boards).any(dim=1)
        return {"boards": boards, "final_hidden": final_hidden, "retries": retries, "fallback": fallback, "deadlocked": deadlocked}


class CategoricalGoalHead(nn.Module):
    """Categorical distribution over the number of goal tiles cleared this move, masked to at most goals_left."""

    def __init__(self, config: StructuredDecoderConfig):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.global_size + config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.max_goal_delta + 1),
        )

    @property
    def n_classes(self) -> int:
        return self.config.max_goal_delta + 1

    def logits(self, global_features: torch.Tensor, board_state: torch.Tensor, goals_left: torch.Tensor) -> torch.Tensor:
        """Compute masked decrement logits from the global feature and the board decoder's final state.

        Inputs: global_features (rows, global_size); board_state (rows, hidden); goals_left (rows,).
        Outputs: logits (rows, max_goal_delta + 1) with classes above goals_left set to -inf.
        """
        rows = global_features.shape[0]
        if board_state.shape != (rows, self.config.hidden_size):
            raise ValueError("board_state must have shape (rows, hidden_size)")
        if goals_left.shape != (rows,):
            raise ValueError("goals_left must have shape (rows,)")
        raw = self.network(torch.cat((global_features, board_state), dim=-1))
        classes = torch.arange(self.n_classes, device=raw.device)
        return raw.masked_fill(~(classes.unsqueeze(0) <= goals_left.long().clamp_min(0).unsqueeze(1)), float("-inf"))

    def nll(self, logits: torch.Tensor, goals_left: torch.Tensor, next_goals_left: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the observed decrement, clamped to the vocabulary.

        Inputs: logits (rows, classes); goals_left (rows,); next_goals_left (rows,).
        Outputs: per-row NLL (rows,) and a flag (rows,) for decrements above the vocabulary.
        """
        delta = goals_left.long() - next_goals_left.long()
        above = delta > self.config.max_goal_delta
        return F.cross_entropy(logits, delta.clamp(0, self.config.max_goal_delta), reduction="none"), above

    def expected_delta(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute the expected decrement under the categorical.

        Inputs: logits (rows, classes).
        Outputs: expectation (rows,).
        """
        classes = torch.arange(self.n_classes, device=logits.device, dtype=logits.dtype)
        return (logits.softmax(dim=-1) * classes).sum(dim=-1)

    def sample(self, logits: torch.Tensor, *, stochastic: bool, generator: torch.Generator | None) -> torch.Tensor:
        """Draw or argmax a decrement.

        Inputs: logits (rows, classes); stochastic flag; generator.
        Outputs: decrement (rows,) long.
        """
        if stochastic:
            return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(1)
        return logits.argmax(dim=-1)


class StructuredStateDecoder(nn.Module):
    """Board decoder and goal head behind one interface used by every world model."""

    def __init__(self, config: StructuredDecoderConfig):
        super().__init__()
        self.config = config
        self.board = AutoregressiveBoardDecoder(config)
        self.goals: CategoricalGoalHead | None = CategoricalGoalHead(config) if config.goal_head else None

    def teacher_forced(
        self,
        *,
        cell_features: torch.Tensor,
        global_features: torch.Tensor,
        next_boards: torch.Tensor,
        goals_left: torch.Tensor,
        next_goals_left: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score the logged next board and goal decrement under teacher forcing.

        Inputs: cell_features (rows, 64, cell_size); global_features (rows, global_size); next_boards (rows, 64); goals_left and next_goals_left (rows,).
        Outputs: dict with board_logits, goal_logits, goal_nll, goal_clamped and expected_delta.
        """
        if self.goals is None:
            raise RuntimeError("this decoder was built without the categorical goal head (goal_head=False)")
        board_logits, final_hidden = self.board.teacher_forced(cell_features, global_features, next_boards)
        goal_logits = self.goals.logits(global_features, final_hidden, goals_left)
        goal_nll, clamped = self.goals.nll(goal_logits, goals_left, next_goals_left)
        return {
            "board_logits": board_logits,
            "goal_logits": goal_logits,
            "goal_nll": goal_nll,
            "goal_clamped": clamped,
            "expected_delta": self.goals.expected_delta(goal_logits),
        }

    @torch.no_grad()
    def sample(
        self,
        *,
        cell_features: torch.Tensor,
        global_features: torch.Tensor,
        goals_left: torch.Tensor,
        colour_support: torch.Tensor,
        stochastic: bool,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        """Generate a match-free next board, then the goal decrement given that board.

        Inputs: cell_features (rows, 64, cell_size); global_features (rows, global_size); goals_left and colour_support (rows,); stochastic flag; generator.
        Outputs: dict with boards, goals_left, goal_delta, goal_logits, retries, fallback and deadlocked.
        """
        if self.goals is None:
            raise RuntimeError("this decoder was built without the categorical goal head (goal_head=False)")
        generated = self.board.generate(cell_features, global_features, colour_support=colour_support, stochastic=stochastic, generator=generator)
        goal_logits = self.goals.logits(global_features, generated["final_hidden"], goals_left)
        delta = self.goals.sample(goal_logits, stochastic=stochastic, generator=generator)
        return {
            "boards": generated["boards"],
            "goals_left": (goals_left.long() - delta).clamp_min(0),
            "goal_delta": delta,
            "goal_logits": goal_logits,
            "retries": generated["retries"],
            "fallback": generated["fallback"],
            "deadlocked": generated["deadlocked"],
        }

    def parameter_counts(self) -> dict[str, int]:
        """Count trainable parameters of the two components.

        Inputs: none.
        Outputs: dict with board_decoder and goal_head counts.
        """
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {"board_decoder": count(self.board), "goal_head": count(self.goals) if self.goals is not None else 0}


__all__ = [
    "DECODER_KINDS",
    "AutoregressiveBoardDecoder",
    "CategoricalGoalHead",
    "SpecialKindHead",
    "StructuredDecoderConfig",
    "StructuredStateDecoder",
    "constrained_sequential_decode",
    "decode_boards",
    "masked_uniform_boards",
    "run_completion_mask",
    "support_mask",
    "torch_legal_mask",
]
