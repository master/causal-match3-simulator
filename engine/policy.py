"""The action mechanism: how ``K`` and ``S_t`` produce ``A_t``.

The player follows a staged bounded-rational policy over legal swaps. Search
controls candidate recall, pattern skill controls evaluation reliability and
noise, planning weights a one-move setup feature, and strategy balances goal
progress against generic clears before softmax choice.

This is the edge that makes ``K`` a confounder of ``E -> R`` rather than merely
a cause of the treatment. If skill reached the outcome only through the served
difficulty, no adjustment would be needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib

import numpy as np

from .board import (
    apply_swap,
    immediate_effect,
    legal_moves,
    local_match_cells,
    resolve_move,
)
from .spec import EMPTY, Action, LevelContext, PlayerSkill, State

#: Weight on total tiles cleared, relative to goal tiles cleared.
W_TOTAL = 0.25

SEARCH_BASE_LOGIT = -0.50
SEARCH_SKILL_SLOPE = 5.00
SEARCH_SALIENCE_SLOPE = 0.10
PATTERN_NOISE_BASE = 2.00
PATTERN_NOISE_SLOPE = 2.00


@dataclass(frozen=True)
class MoveFeatures:
    """Deterministic candidate features available before random refill draws."""

    total_cleared: np.ndarray
    goal_cleared: np.ndarray
    setup_value: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.total_cleared),
            len(self.goal_cleared),
            len(self.setup_value),
        }
        if len(lengths) != 1:
            raise ValueError("move feature arrays must have the same length")

    @property
    def n_moves(self) -> int:
        return len(self.total_cleared)


@dataclass(frozen=True)
class ActionDiagnostics:
    """Observable consequences of the four staged skill mechanisms."""

    candidate_count: int
    noticed_count: int
    pattern_noise_scale: float
    selected_total_cleared: float
    selected_goal_cleared: float
    selected_setup_value: float

    @property
    def candidate_recall(self) -> float:
        return self.noticed_count / max(1, self.candidate_count)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.std())
    if scale < 1e-8:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


@lru_cache(maxsize=None)
def _length_three_windows(
    height: int, width: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    windows: list[tuple[tuple[int, int], ...]] = []
    for row in range(height):
        for col in range(width - 2):
            windows.append(tuple((row, col + offset) for offset in range(3)))
    for col in range(width):
        for row in range(height - 2):
            windows.append(tuple((row + offset, col) for offset in range(3)))
    return tuple(windows)


def _near_match_potential(
    board: np.ndarray,
    excluded: set[tuple[int, int]],
    goal_colour: int,
) -> float:
    """Count uncompleted length-three patterns left outside the immediate clear."""
    height, width = board.shape
    score = 0.0
    for cells in _length_three_windows(height, width):
        if any(cell in excluded for cell in cells):
            continue
        first, second, third = (int(board[cell]) for cell in cells)
        if first == EMPTY or second == EMPTY or third == EMPTY:
            continue
        pair = None
        if first == second != third:
            pair = first
        elif first == third != second:
            pair = first
        elif second == third != first:
            pair = second
        if pair is not None:
            score += 1.0 if pair == goal_colour else 0.10
    return score


def move_feature_table(
    board: np.ndarray,
    moves: list[Action],
    goal_colour: int,
    state: State | None = None,
) -> MoveFeatures:
    """Compute immediate and deterministic setup features for legal candidates."""
    total = np.empty(len(moves), dtype=np.float64)
    goal = np.empty(len(moves), dtype=np.float64)
    setup = np.empty(len(moves), dtype=np.float64)
    for index, action in enumerate(moves):
        swapped = apply_swap(board, action)
        matched = local_match_cells(swapped, action.cells)
        total[index], goal[index] = immediate_effect(
            board,
            action,
            goal_colour,
            specials=None if state is None else state.specials,
        )
        setup[index] = (
            deterministic_rollout_value(state, action)
            if state is not None
            else _near_match_potential(swapped, matched, goal_colour)
        )
    return MoveFeatures(total, goal, setup)


def deterministic_rollout_value(state: State, action: Action) -> float:
    """Hypothetical cascade and next move under a board-derived refill stream."""
    payload = state.board.tobytes() + state.specials.tobytes() + bytes(
        (action.row, action.col, action.drow, action.dcol)
    )
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    n_colours = int(state.board.max()) + 1
    level = LevelContext(
        height=state.board.shape[0],
        width=state.board.shape[1],
        n_colours=n_colours,
        name="planning-rollout",
    )

    def draw(count: int, tag: str) -> np.ndarray:
        return rng.integers(0, n_colours, size=count, dtype=np.int8)

    next_state, transition = resolve_move(state.copy(), action, level, draw)
    next_moves = legal_moves(next_state.board)
    next_best = 0.0
    for next_action in next_moves:
        total_cleared, goal_cleared = immediate_effect(
            next_state.board,
            next_action,
            state.goal_colour,
            specials=next_state.specials,
        )
        next_best = max(next_best, goal_cleared + 0.10 * total_cleared)
    return transition.goal_cleared + 0.50 * next_best


def notice_probabilities(features: MoveFeatures, search_skill: float) -> np.ndarray:
    """Probability that search notices each candidate, with salience assistance."""
    salience = (
        1.5 * _standardize(features.goal_cleared)
        + 0.35 * _standardize(features.total_cleared)
        + 0.15 * _standardize(features.setup_value)
    )
    logits = (
        SEARCH_BASE_LOGIT
        + SEARCH_SKILL_SLOPE * search_skill
        + SEARCH_SALIENCE_SLOPE * salience
    )
    return np.clip(1.0 / (1.0 + np.exp(-logits)), 0.02, 0.995)


def distractor_fallback_probabilities(features: MoveFeatures) -> np.ndarray:
    """Fallback salience when deliberate search finds no candidate."""
    non_goal = np.maximum(0.0, features.total_cleared - features.goal_cleared)
    scores = _standardize(non_goal) - 2.0 * _standardize(features.goal_cleared)
    shifted = 3.0 * (scores - scores.max())
    weights = np.exp(shifted)
    return weights / weights.sum()


def pattern_noise_scale(pattern_skill: float) -> float:
    """Evaluation-error scale, decreasing smoothly with pattern skill."""
    return float(
        np.clip(
            PATTERN_NOISE_BASE * np.exp(-PATTERN_NOISE_SLOPE * pattern_skill),
            0.15,
            5.00,
        )
    )


def pattern_choice_sharpness(pattern_skill: float) -> float:
    """How decisively evaluated move quality controls the final choice."""
    return 0.20 + 5.00 * sigmoid(4.00 * pattern_skill)


def pattern_reliability(pattern_skill: float) -> float:
    """Probability-equivalent weight on true rather than distractor patterns."""
    return sigmoid(4.00 * pattern_skill)


def planning_weight(planning_skill: float) -> float:
    """Weight placed on deterministic future-potential features."""
    return 0.05 + 3.00 * sigmoid(4.00 * planning_skill)


def goal_weight(strategy_skill: float) -> float:
    """Weight placed on goal progress relative to generic clearing."""
    return 0.05 + 2.506 * sigmoid(4.00 * strategy_skill)


def generic_clear_weight(strategy_skill: float) -> float:
    """Low-strategy preference for flashy clears unrelated to the goal."""
    return 0.10 + 2.40 * sigmoid(-5.00 * strategy_skill)


def staged_action_probs(
    features: MoveFeatures,
    player: PlayerSkill,
    noticed: np.ndarray,
    evaluation_noise: np.ndarray,
) -> np.ndarray:
    """Combine the four skill mechanisms into a distribution over candidates."""
    noticed = np.asarray(noticed, dtype=bool)
    noise = np.asarray(evaluation_noise, dtype=np.float64)
    if noticed.shape != (features.n_moves,) or noise.shape != (features.n_moves,):
        raise ValueError("noticed and evaluation_noise must align with candidates")
    if not np.any(noticed):
        raise ValueError("at least one candidate must be noticed")

    true_goal = _standardize(features.goal_cleared)
    distractor = _standardize(
        np.maximum(0.0, features.total_cleared - features.goal_cleared)
    )
    reliability = pattern_reliability(player.pattern)
    perceived_goal = reliability * true_goal + (1.0 - reliability) * distractor
    scores = (
        goal_weight(player.strategy) * perceived_goal
        + generic_clear_weight(player.strategy)
        * _standardize(features.total_cleared)
        + planning_weight(player.planning) * _standardize(features.setup_value)
        + noise
    )
    weights = np.zeros(features.n_moves, dtype=np.float64)
    shifted = pattern_choice_sharpness(player.pattern) * (
        scores[noticed] - np.max(scores[noticed])
    )
    weights[noticed] = np.exp(shifted)
    return weights / weights.sum()


def move_scores(
    board: np.ndarray,
    moves: list[Action],
    goal_colour: int,
    w_total: float = W_TOTAL,
) -> np.ndarray:
    """Score each legal swap by its immediate effect.

    Deliberately one-ply: a player sees the match in front of them, not the
    cascade it sets off.
    """
    scores = np.empty(len(moves), dtype=np.float64)
    for i, action in enumerate(moves):
        cleared, goal = immediate_effect(board, action, goal_colour)
        scores[i] = goal + w_total * cleared
    return scores


__all__ = [
    "ActionDiagnostics",
    "MoveFeatures",
    "goal_weight",
    "generic_clear_weight",
    "distractor_fallback_probabilities",
    "deterministic_rollout_value",
    "move_scores",
    "move_feature_table",
    "notice_probabilities",
    "pattern_noise_scale",
    "pattern_choice_sharpness",
    "pattern_reliability",
    "planning_weight",
    "sigmoid",
    "staged_action_probs",
]
