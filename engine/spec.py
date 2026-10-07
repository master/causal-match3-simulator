"""Variable definitions for the match-3 structural causal model.

Each dataclass here corresponds to one node (or node group) in the DAG:

    L  LevelContext   non-difficulty context; parameterises the transition kernel
    D  Difficulty     baseline tier; sets the move budget and the goal colour
    K  PlayerSkill    latent multidimensional player skill
    E  float          served difficulty, in logits
    S  State          board plus the two counters that make D and E front-loaded
    A  Action         a swap of two orthogonally adjacent cells
    R  int            1 if the quota was cleared within the move budget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

EMPTY = -1
NO_SPECIAL = 0
HORIZONTAL_STRIPE = 1
VERTICAL_STRIPE = 2


@dataclass(frozen=True)
class LevelContext:
    """``L`` -- everything about a level that is not a difficulty setting.

    These fields parameterise the *transition kernel*: the colour alphabet and
    the refill distribution are consulted at every step, not just at ``t=0``.
    """

    height: int = 8
    width: int = 8
    n_colours: int = 5
    #: Unnormalised spawn weights per colour; ``None`` means uniform.
    spawn_weights: tuple[float, ...] | None = None
    name: str = "default"
    #: Relative demand for (search, pattern, planning, strategy) skill.
    skill_demands: tuple[float, ...] = (0.35, 0.30, 0.20, 0.15)

    def weights(self) -> np.ndarray:
        if self.spawn_weights is None:
            w = np.ones(self.n_colours, dtype=np.float64)
        else:
            w = np.asarray(self.spawn_weights, dtype=np.float64)
            if w.shape != (self.n_colours,):
                raise ValueError(
                    f"spawn_weights has length {w.shape[0]}, expected {self.n_colours}"
                )
        return w / w.sum()

    def colour_entropy(self) -> float:
        """Entropy of the spawn distribution, the scalar Lily's Garden logs."""
        w = self.weights()
        return float(-(w * np.log(w)).sum())

    def demand_weights(self) -> np.ndarray:
        """Normalised level demand vector used to project multidimensional skill."""
        q = np.asarray(self.skill_demands, dtype=np.float64)
        if q.shape != (len(SKILL_NAMES),):
            raise ValueError(
                f"skill_demands has length {q.shape[0]}, expected {len(SKILL_NAMES)}"
            )
        if np.any(q < 0) or not np.any(q > 0):
            raise ValueError("skill_demands must be non-negative with a positive sum")
        return q / q.sum()


@dataclass(frozen=True)
class Difficulty:
    """``D`` -- the baseline tier the studio assigns to a level.

    ``goal_count`` here is the *nominal* ask for the tier. What a given player is
    actually served is set by ``E``, and lands in ``S_0.goals_left``.
    """

    move_budget: int = 20
    goal_colour: int = 1
    goal_count: int = 30
    #: The tier's difficulty in logits, on the same scale as ``E``.
    baseline: float = 0.0

    def baseline_logit(self, level: "LevelContext" | None = None) -> float:
        return self.baseline


SKILL_NAMES: tuple[str, ...] = (
    "search",
    "pattern",
    "planning",
    "strategy",
)

#: Population correlation on the standardised MIRT-style skill scale. The
#: dimensions are related but not interchangeable; all marginal variances are 1.
SKILL_COVARIANCE = np.array(
    [
        [1.00, 0.45, 0.25, 0.20],
        [0.45, 1.00, 0.40, 0.30],
        [0.25, 0.40, 1.00, 0.50],
        [0.20, 0.30, 0.50, 1.00],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PlayerSkill:
    """``K`` -- four standardised, correlated gameplay competencies.

    Coordinates are population z-scores in the order given by
    :data:`SKILL_NAMES`. ``label`` is only for deterministic examples and video
    captions; the population itself is continuous and has no segments.
    """

    values: tuple[float, ...]
    label: str = "sampled"

    def __post_init__(self) -> None:
        if len(self.values) != len(SKILL_NAMES):
            raise ValueError(
                f"skill vector has length {len(self.values)}, expected {len(SKILL_NAMES)}"
            )
        if not np.all(np.isfinite(self.values)):
            raise ValueError("skill coordinates must be finite")

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(SKILL_NAMES, map(float, self.values)))

    @property
    def search(self) -> float:
        return float(self.values[0])

    @property
    def pattern(self) -> float:
        return float(self.values[1])

    @property
    def planning(self) -> float:
        return float(self.values[2])

    @property
    def strategy(self) -> float:
        return float(self.values[3])

    def effective_for(self, level: LevelContext) -> float:
        """Level-specific skill projection, standardised to unit variance."""
        q = level.demand_weights()
        scale = float(np.sqrt(q @ SKILL_COVARIANCE @ q))
        return float(q @ self.as_array() / scale)


#: Named points in skill space for examples and visual comparisons. These are
#: not population classes and are never sampled by :func:`sample_K`.
SKILL_PROFILES: tuple[PlayerSkill, ...] = (
    PlayerSkill((-1.0, -1.0, -1.0, -1.0), "developing"),
    PlayerSkill((1.4, 0.8, -0.6, -0.2), "visual-specialist"),
    PlayerSkill((-0.4, 0.2, 1.5, 0.9), "planner"),
    PlayerSkill((1.0, 1.0, 1.0, 1.0), "balanced-expert"),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Pre-registered definition and gates for the landmark churn query."""

    landmark_attempt: int = 20
    warmup_churn_scale: float = 0.005
    e_grid: tuple[float, ...] = tuple(-2.0 + 0.25 * index for index in range(17))
    mastery_target: float = 0.35
    minimum_recommendation_gap: float = 1.0
    minimum_churn_contrast: float = 0.02
    minimum_shoulder_contrast: float = 0.01
    minimum_ess_fraction: float = 0.20
    maximum_normalized_weight: float = 0.01
    calibration_seeds: tuple[int, ...] = (1103, 2207, 3301)
    validation_seeds: tuple[int, ...] = (4409, 5519, 6637, 7753, 8861)

    def __post_init__(self) -> None:
        grid = np.asarray(self.e_grid, dtype=np.float64)
        if self.landmark_attempt < 1:
            raise ValueError("landmark_attempt must be positive")
        if not 0.0 <= self.warmup_churn_scale <= 1.0:
            raise ValueError("warmup_churn_scale must lie in [0, 1]")
        if grid.ndim != 1 or len(grid) < 3 or np.any(np.diff(grid) <= 0):
            raise ValueError("e_grid must be a strictly increasing vector")
        if not 0.0 < self.mastery_target < 1.0:
            raise ValueError("mastery_target must lie in (0, 1)")
        if self.minimum_recommendation_gap <= 0:
            raise ValueError("minimum_recommendation_gap must be positive")
        if self.minimum_churn_contrast <= 0:
            raise ValueError("minimum_churn_contrast must be positive")
        if self.minimum_shoulder_contrast <= 0:
            raise ValueError("minimum_shoulder_contrast must be positive")
        if not 0.0 < self.minimum_ess_fraction <= 1.0:
            raise ValueError("minimum_ess_fraction must lie in (0, 1]")
        if not 0.0 < self.maximum_normalized_weight <= 1.0:
            raise ValueError("maximum_normalized_weight must lie in (0, 1]")
        if set(self.calibration_seeds) & set(self.validation_seeds):
            raise ValueError("calibration and validation seeds must be disjoint")


BENCHMARK_CONFIG = BenchmarkConfig()


@dataclass(frozen=True)
class Action:
    """``A_t`` -- swap ``(row, col)`` with its neighbour one step down or right."""

    row: int
    col: int
    drow: int
    dcol: int

    @property
    def cells(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.row, self.col), (self.row + self.drow, self.col + self.dcol)

    def __str__(self) -> str:  # pragma: no cover - display only
        (r, c), (r2, c2) = self.cells
        return f"({r},{c})<->({r2},{c2})"


@dataclass
class State:
    """``S_t`` -- the board, goal colour, and counters carried through play."""

    board: np.ndarray
    moves_left: int
    goals_left: int
    goal_colour: int
    t: int = 0
    specials: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.specials is None:
            self.specials = np.full(
                self.board.shape, NO_SPECIAL, dtype=np.int8
            )
        else:
            specials = np.asarray(self.specials, dtype=np.int8)
            if specials.shape != self.board.shape:
                raise ValueError("specials must align with the board")
            if np.any(
                ~np.isin(
                    specials,
                    (NO_SPECIAL, HORIZONTAL_STRIPE, VERTICAL_STRIPE),
                )
            ):
                raise ValueError("specials contain an unknown kind")
            self.specials = specials

    def copy(self) -> "State":
        return State(
            self.board.copy(),
            self.moves_left,
            self.goals_left,
            self.goal_colour,
            self.t,
            self.specials.copy(),
        )

    @property
    def won(self) -> bool:
        return self.goals_left <= 0

    @property
    def lost(self) -> bool:
        return self.goals_left > 0 and self.moves_left <= 0

    @property
    def terminal(self) -> bool:
        return self.won or self.lost

    def as_lists(self) -> list[list[int]]:
        return self.board.astype(int).tolist()


@dataclass
class CascadeStep:
    """One clear-fall-refill round inside a single move."""

    matched: list[tuple[int, int]]
    board_before: np.ndarray
    board_after: np.ndarray
    fall: list[tuple[int, int, int, int]]
    spawned: list[tuple[int, int, int]]
    goal_cleared: int
    specials_before: np.ndarray | None = None
    specials_after: np.ndarray | None = None
    created_specials: list[tuple[int, int, int]] = field(default_factory=list)
    activated_specials: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Transition:
    """Everything that happened between ``S_t`` and ``S_{t+1}``.

    Retained in full because the renderer animates the phases, and because the
    cascade depth is a useful difficulty diagnostic.
    """

    action: Action | None
    board_before: np.ndarray
    board_swapped: np.ndarray
    specials_swapped: np.ndarray | None = None
    steps: list[CascadeStep] = field(default_factory=list)
    reshuffled: bool = False

    @property
    def goal_cleared(self) -> int:
        return sum(s.goal_cleared for s in self.steps)

    @property
    def tiles_cleared(self) -> int:
        return sum(len(s.matched) for s in self.steps)

    @property
    def cascade_depth(self) -> int:
        return len(self.steps)

    @property
    def created_specials(self) -> list[tuple[int, int, int]]:
        return [special for step in self.steps for special in step.created_specials]

    @property
    def activated_specials(self) -> list[tuple[int, int]]:
        return [special for step in self.steps for special in step.activated_specials]


def uniform_board(rng: np.random.Generator, level: LevelContext) -> np.ndarray:
    return rng.choice(
        level.n_colours, size=(level.height, level.width), p=level.weights()
    ).astype(np.int8)


def state_from(
    level: LevelContext, difficulty: Difficulty, board: np.ndarray, goals: int
) -> State:
    return State(
        board=board,
        moves_left=difficulty.move_budget,
        goals_left=goals,
        goal_colour=difficulty.goal_colour,
        t=0,
    )


__all__ = [
    "EMPTY",
    "HORIZONTAL_STRIPE",
    "NO_SPECIAL",
    "VERTICAL_STRIPE",
    "Action",
    "BENCHMARK_CONFIG",
    "BenchmarkConfig",
    "CascadeStep",
    "Difficulty",
    "LevelContext",
    "PlayerSkill",
    "SKILL_COVARIANCE",
    "SKILL_NAMES",
    "SKILL_PROFILES",
    "State",
    "Transition",
    "state_from",
    "uniform_board",
]
