"""The structural causal model, one function per node of the DAG.

Every node gets its own function taking its parents as arguments, so the graph

    L -> D,  L -> E,  D -> E,  K -> E,  D -> S_0,  E -> S_0,  L -> S_t,
    K -> A_t,  S_T -> R,  K -> X

can be read straight off the call sites in :func:`ground_truth_model`.

Two conventions hold throughout:

* Every node function accepts ``value=``. Passing it clamps the node, which is
  how interventions are expressed: ``sample_E(D, L, K, value=1.2)`` is
  ``do(E = 1.2)``.
* ``pyro.sample`` appears at stochastic SCM nodes and gameplay events: the root
  variables, noisy treatment and proxy, opening deal, refill colours, and the
  player's choice of swap. Match detection, gravity, and cascade resolution are
  deterministic given those draws and are recorded with ``pyro.deterministic``.

The load-bearing structural choice is that ``E`` is the difficulty the policy
*served*, not how hard the level *felt*. Skill reaches the outcome through
``K -> A_t``, independently of ``E``. Were skill instead absorbed into ``E``, the
two would be d-separated from ``R`` given ``E`` and there would be no confounding
to correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .board import deal, legal_moves, resolve_move
from .evidence import EVIDENCE_NAMES, EVIDENCE_Q_MATRIX, sample_evidence
from .policy import (
    ActionDiagnostics,
    distractor_fallback_probabilities,
    move_feature_table,
    notice_probabilities,
    pattern_noise_scale,
    staged_action_probs,
)
from .spec import (
    SKILL_COVARIANCE,
    Action,
    Difficulty,
    LevelContext,
    PlayerSkill,
    State,
    Transition,
    state_from,
)

# --------------------------------------------------------------------- setup --

#: Level catalogue. ``L`` is a root node, so this is its support.
LEVELS: tuple[LevelContext, ...] = (
    LevelContext(
        height=8,
        width=8,
        n_colours=5,
        name="orchard",
        skill_demands=(0.46, 0.27, 0.27, 0.00),
    ),
    LevelContext(height=8, width=8, n_colours=5, name="harbour",
                 spawn_weights=(1.0, 0.8, 1.0, 1.1, 1.1),
                 skill_demands=(0.55, 0.225, 0.225, 0.00)),
    LevelContext(
        height=8,
        width=8,
        n_colours=6,
        name="foundry",
        skill_demands=(0.45, 0.15, 0.40, 0.00),
    ),
)
LEVEL_PROBS: tuple[float, ...] = (0.5, 0.3, 0.2)

#: Baseline tiers, on the same logit scale as ``E``.
TIER_LOGITS: tuple[float, ...] = (-0.8, 0.0, 0.8)
TIER_NAMES: tuple[str, ...] = ("easy", "medium", "hard")
TIER_PROBS: tuple[float, ...] = (0.3, 0.45, 0.25)
TIER_MOVE_BUDGETS: tuple[int, ...] = (22, 20, 18)
DEFAULT_MOVE_BUDGET = 20
DEFAULT_GOAL_COLOUR = 1

#: Level-specific natural DDA strength and exploration noise, aligned with
#: ``LEVELS``. Scalar overrides remain available for controlled ablations.
DDA_GAINS: tuple[float, ...] = (4.0, 4.0, 8.0)
E_SIGMAS: tuple[float, ...] = (1.2, 1.2, 2.4)
DEFAULT_DDA_GAIN = 0.8
DEFAULT_E_SIGMA = 0.65

PROXY_NAMES = EVIDENCE_NAMES
PROXY_LOADINGS = EVIDENCE_Q_MATRIX


@dataclass
class Episode:
    """One level attempt, with every node of the DAG recorded."""

    level: LevelContext
    difficulty: Difficulty
    player: PlayerSkill
    E: float
    served_goal_count: int
    states: list[State]
    actions: list[Action]
    transitions: list[Transition]
    proxy: np.ndarray
    R: int
    tier: str = ""
    action_diagnostics: list[ActionDiagnostics] = field(default_factory=list)

    @property
    def moves_used(self) -> int:
        return len(self.actions)

    @property
    def goals_cleared(self) -> int:
        return self.served_goal_count - self.states[-1].goals_left

    @property
    def reshuffles(self) -> int:
        return sum(1 for t in self.transitions if t.reshuffled)


def _tensor(x, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _make_draw(level: LevelContext, prefix: str):
    """Build the refill sampler handed to the deterministic board code."""
    probs = _tensor(level.weights())

    def draw(n: int, tag: str) -> np.ndarray:
        site = dist.Categorical(probs=probs).expand([n]).to_event(1)
        drawn = pyro.sample(f"{prefix}/{tag}", site)
        return drawn.detach().cpu().numpy().astype(np.int8)

    return draw


# ------------------------------------------------------------------- nodes ----


def sample_L(name: str = "L", value: LevelContext | None = None) -> LevelContext:
    """``L`` -- level context. A root node; parameterises the transition kernel."""
    if value is not None:
        return value
    idx = pyro.sample(f"{name}/index", dist.Categorical(probs=_tensor(LEVEL_PROBS)))
    return LEVELS[int(idx)]


def sample_K(name: str = "K", value: PlayerSkill | None = None) -> PlayerSkill:
    """``K`` -- latent player skill. Root node, observed only through ``X``."""
    if value is not None:
        return value
    drawn = pyro.sample(
        name,
        dist.MultivariateNormal(
            torch.zeros(4), covariance_matrix=_tensor(SKILL_COVARIANCE)
        ),
    )
    return PlayerSkill(tuple(float(value) for value in drawn.detach().cpu()))


def sample_D(
    level: LevelContext,
    name: str = "D",
    value: Difficulty | None = None,
    move_budget: int | None = None,
    goal_colour: int = DEFAULT_GOAL_COLOUR,
) -> tuple[Difficulty, str]:
    """``D`` -- the baseline tier the studio assigns, given ``L``.

    The nominal goal count is read off the calibration curve, so a tier means the
    same thing across levels with different palettes. That is the ``L -> D``
    edge: the same ask is a different proposition at five colours than at six.
    """
    if value is not None:
        return value, "given"

    idx = int(pyro.sample(f"{name}/tier", dist.Categorical(probs=_tensor(TIER_PROBS))))
    baseline = TIER_LOGITS[idx]
    served_move_budget = TIER_MOVE_BUDGETS[idx] if move_budget is None else move_budget

    from .calibrate import goal_count_for_E

    nominal = goal_count_for_E(level.name, baseline)
    difficulty = Difficulty(
        move_budget=served_move_budget,
        goal_colour=goal_colour,
        goal_count=nominal,
        baseline=baseline,
    )
    return difficulty, TIER_NAMES[idx]


def sample_E(
    difficulty: Difficulty,
    level: LevelContext,
    player: PlayerSkill,
    name: str = "E",
    value: float | None = None,
    gain: float | None = None,
    sigma: float | None = None,
) -> float:
    """``E`` -- the difficulty served, given ``D``, ``L``, and ``K``.

    Stronger players are served harder levels, which is what every published DDA
    does. The plus sign is the source of the confounding: it correlates the
    treatment with a variable that independently raises the outcome.
    """
    if value is not None:
        return float(pyro.deterministic(name, _tensor(value)))
    level_index = next(
        (
            index
            for index, candidate in enumerate(LEVELS)
            if candidate.name == level.name
        ),
        None,
    )
    if gain is None:
        gain = DDA_GAINS[level_index] if level_index is not None else DEFAULT_DDA_GAIN
    if sigma is None:
        sigma = E_SIGMAS[level_index] if level_index is not None else DEFAULT_E_SIGMA
    location = difficulty.baseline + gain * player.effective_for(level)
    if sigma == 0:
        return float(pyro.deterministic(name, _tensor(location)))
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return float(pyro.sample(name, dist.Normal(_tensor(location), sigma)))


def sample_S0(
    level: LevelContext,
    difficulty: Difficulty,
    E: float,
    name: str = "S0",
    value: State | None = None,
    served: int | None = None,
) -> State:
    """``S_0`` -- opening state, given ``L``, ``D`` and ``E``.

    ``E`` sets the quota by inverting the calibration curve; ``D`` sets the move
    budget and goal colour. Both enter only here, so ``S_0`` is a sufficient
    statistic for the level configuration.
    """
    if value is not None:
        return value.copy()

    if served is None:
        from .calibrate import goal_count_for_E

        served = goal_count_for_E(level.name, E, fallback=difficulty.goal_count)

    board = deal(level, _make_draw(level, name), tag="deal")
    pyro.deterministic(f"{name}/goals", _tensor(served))
    return state_from(level, difficulty, board, served)


def _sample_action_decision(
    state: State,
    player: PlayerSkill,
    name: str | None = None,
    value: Action | None = None,
) -> tuple[Action | None, ActionDiagnostics | None]:
    """Sample ``A_t`` and retain diagnostics for the staged skill mechanisms.

    Returns ``None`` values when the board offers no legal move.
    """
    site = name or f"A/{state.t}"
    if value is not None:
        return value, None

    moves = legal_moves(state.board)
    if not moves:
        return None, None

    features = move_feature_table(
        state.board, moves, state.goal_colour, state=state
    )
    notice_probability = notice_probabilities(features, player.search)
    noticed_draw = pyro.sample(
        f"{site}/noticed",
        dist.Bernoulli(probs=_tensor(notice_probability)).to_event(1),
    )
    noticed = noticed_draw.detach().cpu().numpy().astype(bool)
    if not np.any(noticed):
        fallback = pyro.sample(
            f"{site}/fallback",
            dist.Categorical(
                probs=_tensor(distractor_fallback_probabilities(features))
            ),
        )
        noticed[int(fallback)] = True

    noise_scale = pattern_noise_scale(player.pattern)
    evaluation_noise = pyro.sample(
        f"{site}/evaluation_noise",
        dist.Normal(torch.zeros(len(moves)), noise_scale).to_event(1),
    )
    probs = staged_action_probs(
        features,
        player,
        noticed,
        evaluation_noise.detach().cpu().numpy(),
    )
    pyro.deterministic(f"{site}/noticed_count", _tensor(int(noticed.sum())))
    pyro.deterministic(f"{site}/pattern_noise_scale", _tensor(noise_scale))
    idx = pyro.sample(site, dist.Categorical(probs=_tensor(probs)))
    selected = int(idx)
    diagnostics = ActionDiagnostics(
        candidate_count=len(moves),
        noticed_count=int(noticed.sum()),
        pattern_noise_scale=noise_scale,
        selected_total_cleared=float(features.total_cleared[selected]),
        selected_goal_cleared=float(features.goal_cleared[selected]),
        selected_setup_value=float(features.setup_value[selected]),
    )
    return moves[selected], diagnostics


def sample_A(
    state: State,
    player: PlayerSkill,
    name: str | None = None,
    value: Action | None = None,
) -> Action | None:
    """``A_t`` -- the swap, given only current ``S_t`` and stable ``K``."""
    action, _ = _sample_action_decision(state, player, name=name, value=value)
    return action


def sample_S_next(
    state: State,
    action: Action,
    level: LevelContext,
    name: str | None = None,
    value: tuple[State, Transition] | None = None,
) -> tuple[State, Transition]:
    """``S_{t+1}`` -- next state, given ``S_t``, ``A_t`` and ``L``.

    The only randomness is the refill colours, drawn from ``L``'s spawn
    distribution. Matching, gravity and the cascade fixpoint are deterministic.
    """
    if value is not None:
        return value

    prefix = name or f"S/{state.t + 1}"
    nxt, transition = resolve_move(
        state, action, level, _make_draw(level, prefix)
    )
    pyro.deterministic(f"{prefix}/goals_left", _tensor(nxt.goals_left))
    return nxt, transition


def sample_X(
    player: PlayerSkill,
    level: LevelContext,
    name: str = "X",
    value: np.ndarray | None = None,
) -> np.ndarray:
    """``X`` -- typed task evidence, given multidimensional ``K`` and ``L``."""
    return sample_evidence(player, level, name=name, value=value)


def sample_R(
    terminal_state: State, name: str = "R", value: int | None = None
) -> int:
    """``R`` -- success: was the quota cleared within the move budget?

    Deterministic. Every source of randomness that produces ``R`` has already
    acted upstream, in the deal, the refills and the swaps.
    """
    outcome = int(value) if value is not None else int(terminal_state.goals_left <= 0)
    pyro.deterministic(name, _tensor(outcome))
    return outcome


# ------------------------------------------------------------------- model ----


def ground_truth_model(
    level: LevelContext | None = None,
    player: PlayerSkill | None = None,
    difficulty: Difficulty | None = None,
    E: float | None = None,
    evidence: np.ndarray | None = None,
    served_goal_count: int | None = None,
    dda_gain: float | None = None,
    e_sigma: float | None = None,
    dda_gains: tuple[float, ...] | None = None,
    e_sigmas: tuple[float, ...] | None = None,
    max_steps: int | None = None,
    action_policy: Callable[[State, PlayerSkill], Action | None] | None = None,
) -> Episode:
    """The full data-generating process for one level attempt.

    Any node can be pinned by passing it, which is how interventional
    distributions are drawn: pass ``E=`` for ``do(E = e)``, ``player=`` to
    condition on a skill vector, and so on.
    """
    L = sample_L(value=level)
    K = sample_K(value=player)
    D, tier = sample_D(L, value=difficulty)
    if dda_gain is not None and dda_gains is not None:
        raise ValueError("dda_gain and dda_gains cannot both be supplied")
    if e_sigma is not None and e_sigmas is not None:
        raise ValueError("e_sigma and e_sigmas cannot both be supplied")
    level_index = next(
        (index for index, candidate in enumerate(LEVELS) if candidate.name == L.name),
        None,
    )
    if dda_gains is not None:
        if len(dda_gains) != len(LEVELS) or level_index is None:
            raise ValueError("dda_gains must align with simulator levels")
        dda_gain = dda_gains[level_index]
    if e_sigmas is not None:
        if len(e_sigmas) != len(LEVELS) or level_index is None:
            raise ValueError("e_sigmas must align with simulator levels")
        e_sigma = e_sigmas[level_index]
    eff = sample_E(D, L, K, value=E, gain=dda_gain, sigma=e_sigma)

    state = sample_S0(L, D, eff, served=served_goal_count)
    served = state.goals_left
    states: list[State] = [state]
    actions: list[Action] = []
    transitions: list[Transition] = []
    action_diagnostics: list[ActionDiagnostics] = []

    budget = D.move_budget if max_steps is None else min(D.move_budget, max_steps)
    for _ in range(budget):
        if state.terminal:
            break
        if action_policy is None:
            action, diagnostics = _sample_action_decision(state, K)
        else:
            action = action_policy(state, K)
            diagnostics = None
        if action is None:
            break
        if action not in legal_moves(state.board):
            raise ValueError(f"action policy returned illegal move {action}")
        state, transition = sample_S_next(state, action, L)
        actions.append(action)
        transitions.append(transition)
        if diagnostics is not None:
            action_diagnostics.append(diagnostics)
        states.append(state)

    R = sample_R(states[-1])
    X = sample_X(K, L, value=evidence)

    return Episode(
        level=L,
        difficulty=D,
        player=K,
        E=eff,
        served_goal_count=served,
        states=states,
        actions=actions,
        transitions=transitions,
        proxy=X,
        R=R,
        tier=tier,
        action_diagnostics=action_diagnostics,
    )


__all__ = [
    "DDA_GAINS",
    "Episode",
    "E_SIGMAS",
    "LEVELS",
    "LEVEL_PROBS",
    "PROXY_LOADINGS",
    "PROXY_NAMES",
    "TIER_LOGITS",
    "TIER_MOVE_BUDGETS",
    "TIER_NAMES",
    "ground_truth_model",
    "sample_A",
    "sample_D",
    "sample_E",
    "sample_K",
    "sample_L",
    "sample_R",
    "sample_S0",
    "sample_S_next",
    "sample_X",
]
