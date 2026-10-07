"""Calibrate the served difficulty scale by simulation.

``E`` is *defined* by the win rate it induces for a fixed reference player:

    E  ==  -logit Pr(R = 1 | reference player)

so ``E = 0`` is a configuration the reference clears half the time. Calibration
bisects the goal count against that reference over a grid of ``E``, producing an
invertible curve per level.

Doing this by simulation rather than arithmetic is not fastidiousness. In a
production difficulty model, features derived from a play-testing agent were five
to six times more predictive than the static level attributes: realised
difficulty is not a function of the knobs.

The reference player is deliberately mid-skill. An agent that is too strong
passes everything and the easy end of the scale stops separating -- a failure
mode reported directly in the published work.
"""

from __future__ import annotations

import json
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .retention import WinPropensityModel
from .spec import SKILL_COVARIANCE, SKILL_NAMES, Difficulty, LevelContext, PlayerSkill

REFERENCE_PLAYER = PlayerSkill((0.0, 0.0, 0.0, 0.0), "reference")

#: The difficulty scale the curve is solved over. Spans the range the DDA
#: actually reaches: baseline tier plus gain times skill, plus policy noise.
E_GRID: tuple[float, ...] = (
    -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
)

TABLE_PATH = Path(__file__).with_name("calibration.json")
WIN_PROPENSITY_PATH = Path(__file__).with_name("win_propensity_calibration.json")


@dataclass
class CurvePoint:
    E: float
    goal_count: int
    pass_rate: float


@dataclass(frozen=True)
class WinPropensityData:
    """Stratified engine observations used to calibrate oracle win propensity."""

    level_indices: np.ndarray
    tier_indices: np.ndarray
    skills: np.ndarray
    served_difficulty: np.ndarray
    outcomes: np.ndarray

    def __post_init__(self) -> None:
        n_rows = len(self.outcomes)
        if np.shape(self.level_indices) != (n_rows,):
            raise ValueError("level_indices must align with outcomes")
        if np.shape(self.tier_indices) != (n_rows,):
            raise ValueError("tier_indices must align with outcomes")
        if np.shape(self.skills) != (n_rows, len(SKILL_NAMES)):
            raise ValueError("skills must have shape (observations, skills)")
        if np.shape(self.served_difficulty) != (n_rows,):
            raise ValueError("served_difficulty must align with outcomes")
        if n_rows == 0:
            raise ValueError("win-propensity data cannot be empty")
        outcomes = np.asarray(self.outcomes)
        if np.any((outcomes < 0) | (outcomes > 1)):
            raise ValueError("outcomes must be binary")

    def subset(self, rows: np.ndarray) -> "WinPropensityData":
        index = np.asarray(rows, dtype=np.int64)
        return WinPropensityData(
            level_indices=np.asarray(self.level_indices)[index],
            tier_indices=np.asarray(self.tier_indices)[index],
            skills=np.asarray(self.skills)[index],
            served_difficulty=np.asarray(self.served_difficulty)[index],
            outcomes=np.asarray(self.outcomes)[index],
        )


def save_win_propensity_data(
    data: WinPropensityData, path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        version=np.asarray([1], dtype=np.int16),
        level_indices=np.asarray(data.level_indices, dtype=np.int16),
        tier_indices=np.asarray(data.tier_indices, dtype=np.int16),
        skills=np.asarray(data.skills, dtype=np.float32),
        served_difficulty=np.asarray(data.served_difficulty, dtype=np.float32),
        outcomes=np.asarray(data.outcomes, dtype=np.float32),
    )
    return output


def load_win_propensity_data(path: str | Path) -> WinPropensityData:
    with np.load(Path(path), allow_pickle=False) as values:
        version = int(values["version"][0])
        if version != 1:
            raise ValueError(f"unsupported win-propensity data version {version}")
        return WinPropensityData(
            level_indices=values["level_indices"].astype(np.int64),
            tier_indices=values["tier_indices"].astype(np.int64),
            skills=values["skills"].astype(np.float64),
            served_difficulty=values["served_difficulty"].astype(np.float64),
            outcomes=values["outcomes"].astype(np.float64),
        )


def split_win_propensity_data(
    data: WinPropensityData,
    *,
    validation_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[WinPropensityData, WinPropensityData]:
    """Split repeated calibration rows by unique skill draw."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    _, group_index = np.unique(np.asarray(data.skills), axis=0, return_inverse=True)
    groups = np.unique(group_index)
    if len(groups) < 2:
        raise ValueError("at least two distinct skill draws are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    n_validation = min(
        len(groups) - 1,
        max(1, int(round(validation_fraction * len(groups)))),
    )
    validation_groups = shuffled[:n_validation]
    validation_mask = np.isin(group_index, validation_groups)
    return (
        data.subset(np.flatnonzero(~validation_mask)),
        data.subset(np.flatnonzero(validation_mask)),
    )


def win_propensity_design_summary(data: WinPropensityData) -> dict[str, object]:
    """Infer balanced-design dimensions for artifact provenance."""
    n_skills = len(np.unique(np.asarray(data.skills), axis=0))
    levels = np.unique(data.level_indices)
    tiers = np.unique(data.tier_indices)
    e_grid = np.unique(data.served_difficulty)
    denominator = n_skills * len(levels) * len(tiers) * len(e_grid)
    if denominator == 0 or len(data.outcomes) % denominator:
        raise ValueError("win-propensity data is not a balanced factorial design")
    return {
        "n_observations": len(data.outcomes),
        "n_skill_draws": n_skills,
        "n_levels": len(levels),
        "n_tiers": len(tiers),
        "n_replicates": len(data.outcomes) // denominator,
        "e_grid": [float(value) for value in e_grid],
    }


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def fit_win_propensity_model(
    data: WinPropensityData,
    level_names: tuple[str, ...],
    tier_names: tuple[str, ...],
    *,
    skill_directions: np.ndarray | None = None,
    max_iter: int = 200,
    seed: int = 0,
) -> WinPropensityModel:
    """Fit a constrained logistic response surface by maximum likelihood."""
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    torch.manual_seed(seed)
    dtype = torch.float64
    level = torch.as_tensor(data.level_indices, dtype=torch.long)
    tier = torch.as_tensor(data.tier_indices, dtype=torch.long)
    skills = torch.as_tensor(data.skills, dtype=dtype)
    served = torch.as_tensor(data.served_difficulty, dtype=dtype)
    outcomes = torch.as_tensor(data.outcomes, dtype=dtype)
    n_levels = len(level_names)
    n_tiers = len(tier_names)
    if int(level.min()) < 0 or int(level.max()) >= n_levels:
        raise ValueError("level index outside supplied names")
    if int(tier.min()) < 0 or int(tier.max()) >= n_tiers:
        raise ValueError("tier index outside supplied names")
    direction_tensor = None
    if skill_directions is not None:
        directions = np.asarray(skill_directions, dtype=np.float64)
        if directions.shape != (n_levels, len(SKILL_NAMES)):
            raise ValueError("skill_directions must have shape (levels, skills)")
        if np.any(directions < 0) or np.any(directions.sum(axis=1) <= 0):
            raise ValueError("skill directions must be non-negative and nonzero")
        direction_tensor = torch.as_tensor(directions, dtype=dtype)

    initial_intercepts = np.zeros((n_levels, n_tiers), dtype=np.float64)
    level_array = np.asarray(data.level_indices)
    tier_array = np.asarray(data.tier_indices)
    outcome_array = np.asarray(data.outcomes)
    for level_index in range(n_levels):
        for tier_index in range(n_tiers):
            mask = (level_array == level_index) & (tier_array == tier_index)
            successes = float(outcome_array[mask].sum())
            count = int(mask.sum())
            rate = (successes + 0.5) / (count + 1.0) if count else 0.5
            initial_intercepts[level_index, tier_index] = np.log(rate / (1 - rate))

    intercepts = torch.nn.Parameter(torch.as_tensor(initial_intercepts, dtype=dtype))
    raw_skill_shape = (
        (n_levels,) if direction_tensor is not None else (n_levels, len(SKILL_NAMES))
    )
    raw_skill = torch.nn.Parameter(
        torch.full(raw_skill_shape, _inverse_softplus(0.20), dtype=dtype)
    )
    raw_difficulty = torch.nn.Parameter(
        torch.full((n_levels,), _inverse_softplus(1.0), dtype=dtype)
    )
    parameters = [intercepts, raw_skill, raw_difficulty]
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=0.5,
        max_iter=max_iter,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        skill_coefficients = F.softplus(raw_skill)
        if direction_tensor is not None:
            skill_coefficients = skill_coefficients.unsqueeze(1) * direction_tensor
        difficulty_coefficients = F.softplus(raw_difficulty) + 1e-6
        logits = (
            intercepts[level, tier]
            + (skill_coefficients[level] * skills).sum(dim=1)
            - difficulty_coefficients[level] * served
        )
        loss = F.binary_cross_entropy_with_logits(logits, outcomes)
        loss = loss + 1e-5 * (
            skill_coefficients.square().mean()
            + difficulty_coefficients.square().mean()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted_skill_tensor = F.softplus(raw_skill)
    if direction_tensor is not None:
        fitted_skill_tensor = fitted_skill_tensor.unsqueeze(1) * direction_tensor
    fitted_skill = fitted_skill_tensor.detach().cpu().numpy()
    fitted_difficulty = F.softplus(raw_difficulty).detach().cpu().numpy() + 1e-6
    return WinPropensityModel(
        level_names=level_names,
        tier_names=tier_names,
        intercepts=tuple(
            tuple(map(float, row)) for row in intercepts.detach().cpu().numpy()
        ),
        skill_coefficients=tuple(tuple(map(float, row)) for row in fitted_skill),
        difficulty_coefficients=tuple(map(float, fitted_difficulty)),
    )


def predict_win_propensity(
    model: WinPropensityModel, data: WinPropensityData
) -> np.ndarray:
    predictions = np.empty(len(data.outcomes), dtype=np.float64)
    levels = np.asarray(data.level_indices)
    for level_index, level_name in enumerate(model.level_names):
        mask = levels == level_index
        if not np.any(mask):
            continue
        predictions[mask] = model.probabilities(
            level_name,
            np.asarray(data.tier_indices)[mask],
            np.asarray(data.skills)[mask],
            np.asarray(data.served_difficulty)[mask],
        )
    return predictions


def propensity_brier_score(
    model: WinPropensityModel, data: WinPropensityData
) -> float:
    predictions = predict_win_propensity(model, data)
    return float(np.mean((predictions - np.asarray(data.outcomes)) ** 2))


def _episode_outcome(
    payload: tuple[
        LevelContext,
        Difficulty,
        PlayerSkill,
        float,
        int | None,
        int,
    ],
) -> int:
    """Run one seeded engine episode; top-level so process pools can pickle it."""
    import pyro

    from .scm import PROXY_NAMES, ground_truth_model

    level, difficulty, player, served_difficulty, served_goal_count, seed = payload
    pyro.set_rng_seed(seed)
    return ground_truth_model(
        level=level,
        player=player,
        difficulty=difficulty,
        E=served_difficulty,
        evidence=np.zeros(len(PROXY_NAMES), dtype=np.float64),
        served_goal_count=served_goal_count,
    ).R


def _episode_goal_total(
    payload: tuple[LevelContext, int, int, int],
) -> int:
    """Run one full-budget reference trajectory and return total goal progress."""
    import pyro

    from .scm import PROXY_NAMES, ground_truth_model

    level, move_budget, goal_colour, seed = payload
    pyro.set_rng_seed(seed)
    served_goal_count = 10_000
    difficulty = Difficulty(
        move_budget=move_budget,
        goal_colour=goal_colour,
        goal_count=served_goal_count,
        baseline=0.0,
    )
    episode = ground_truth_model(
        level=level,
        player=REFERENCE_PLAYER,
        difficulty=difficulty,
        E=0.0,
        evidence=np.zeros(len(PROXY_NAMES), dtype=np.float64),
        served_goal_count=served_goal_count,
    )
    return episode.goals_cleared


def _run_episode_tasks(
    tasks: list[
        tuple[LevelContext, Difficulty, PlayerSkill, float, int | None, int]
    ],
    executor: Executor | None,
) -> list[int]:
    if executor is None:
        return [_episode_outcome(task) for task in tasks]
    return list(executor.map(_episode_outcome, tasks, chunksize=max(1, len(tasks) // 64)))


def reference_goal_totals(
    level: LevelContext,
    *,
    n: int,
    seed: int,
    move_budget: int = 20,
    goal_colour: int = 1,
    executor: Executor | None = None,
) -> np.ndarray:
    """Sample full-budget progress once for all candidate goal thresholds."""
    if n < 1:
        raise ValueError("n must be positive")
    tasks = [
        (
            level,
            move_budget,
            goal_colour,
            int(np.random.SeedSequence([seed, replicate]).generate_state(1)[0]),
        )
        for replicate in range(n)
    ]
    if executor is None:
        values = [_episode_goal_total(task) for task in tasks]
    else:
        values = list(
            executor.map(
                _episode_goal_total,
                tasks,
                chunksize=max(1, len(tasks) // 64),
            )
        )
    return np.asarray(values, dtype=np.int64)


def goal_count_from_totals(
    totals: np.ndarray,
    target: float,
    *,
    lo: int = 3,
    hi: int = 200,
) -> tuple[int, float]:
    """Select the integer quota whose empirical pass rate is nearest target."""
    values = np.asarray(totals, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("totals must be a non-empty vector")
    if not 0.0 <= target <= 1.0:
        raise ValueError("target must lie in [0, 1]")
    if lo > hi:
        raise ValueError("lo cannot exceed hi")
    candidates = np.arange(lo, hi + 1, dtype=np.int64)
    rates = np.asarray([np.mean(values >= goal) for goal in candidates])
    errors = np.abs(rates - target)
    best = int(np.flatnonzero(np.isclose(errors, errors.min()))[0])
    return int(candidates[best]), float(rates[best])


def target_rate(E: float) -> float:
    return float(1.0 / (1.0 + np.exp(E)))


def win_rate(
    level: LevelContext,
    goal_count: int,
    move_budget: int = 20,
    goal_colour: int = 1,
    n: int = 200,
    seed: int = 0,
    executor: Executor | None = None,
) -> float:
    """Reference player's clear rate on a configuration, bypassing the curve."""
    difficulty = Difficulty(move_budget, goal_colour, goal_count, baseline=0.0)
    tasks = []
    for replicate in range(n):
        episode_seed = int(
            np.random.SeedSequence([seed, replicate]).generate_state(1)[0]
        )
        tasks.append(
            (
                level,
                difficulty,
                REFERENCE_PLAYER,
                0.0,
                goal_count,
                episode_seed,
            )
        )
    return sum(_run_episode_tasks(tasks, executor)) / n


def solve_goal_count(
    level: LevelContext,
    target: float,
    move_budget: int = 20,
    goal_colour: int = 1,
    n: int = 200,
    lo: int = 3,
    hi: int = 140,
    seed: int = 0,
    cache: dict[int, float] | None = None,
    executor: Executor | None = None,
) -> tuple[int, float]:
    """Bisect on goal count for a target win rate.

    Win rate is monotone decreasing in the goal count, which is what makes
    bisection valid and the resulting curve invertible.
    """
    best: tuple[int, float] | None = None
    rates = cache if cache is not None else {}
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid not in rates:
            rates[mid] = win_rate(
                level, mid, move_budget, goal_colour, n, seed, executor
            )
        rate = rates[mid]
        if best is None or abs(rate - target) < abs(best[1] - target):
            best = (mid, rate)
        if rate > target:
            lo = mid + 1
        else:
            hi = mid - 1
    assert best is not None
    return best


def calibrate(
    levels: list[LevelContext],
    grid: tuple[float, ...] = E_GRID,
    n: int = 200,
    move_budget: int = 20,
    goal_colour: int = 1,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, dict]:
    if workers < 1:
        raise ValueError("workers must be positive")
    table: dict[str, dict] = {}
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for level_index, level in enumerate(levels):
            points: list[CurvePoint] = []
            totals = reference_goal_totals(
                level,
                n=n,
                seed=seed + 10_000 * level_index,
                move_budget=move_budget,
                goal_colour=goal_colour,
                executor=executor,
            )
            lo = 3
            for E in grid:
                goal_count, rate = goal_count_from_totals(
                    totals,
                    target_rate(E),
                    lo=lo,
                    hi=200,
                )
                points.append(CurvePoint(float(E), int(goal_count), float(rate)))
                lo = max(3, goal_count)
            table[level.name] = {
                "move_budget": move_budget,
                "goal_colour": goal_colour,
                "n": n,
                "seed": seed,
                "curve": [asdict(p) for p in points],
            }
    finally:
        if executor is not None:
            executor.shutdown()
    return table


def save(table: dict[str, dict], path: Path = TABLE_PATH) -> None:
    path.write_text(json.dumps(table, indent=2) + "\n")
    _CACHE[path.resolve()] = table


_CACHE: dict[Path, dict[str, dict]] = {}


def calibration_table_path(path: str | Path | None = None) -> Path:
    """Resolve the calibration table selected for this process."""
    return (
        Path(os.environ.get("MATCH3_CALIBRATION_PATH", TABLE_PATH))
        if path is None
        else Path(path)
    ).resolve()


def load(path: str | Path | None = None) -> dict[str, dict]:
    selected = calibration_table_path(path)
    if selected not in _CACHE:
        document = json.loads(selected.read_text()) if selected.exists() else {}
        _CACHE[selected] = document.get("table", document)
    return _CACHE[selected]


def goal_count_for_E(level_name: str, E: float, fallback: int = 30) -> int:
    """Invert the calibration curve. Linear interpolation, clamped at the ends."""
    entry = load().get(level_name)
    if not entry:
        return fallback
    curve = entry["curve"]
    xs = [p["E"] for p in curve]
    ys = [p["goal_count"] for p in curve]
    return int(round(float(np.interp(E, xs, ys))))


def validate_goal_count_table(
    table: dict[str, dict],
    levels: tuple[LevelContext, ...],
    *,
    n: int,
    seed: int,
    workers: int = 1,
) -> dict[str, object]:
    """Evaluate selected goal counts on independent common-random seeds."""
    if n < 1 or workers < 1:
        raise ValueError("n and workers must be positive")
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    reports: dict[str, list[dict[str, float | int]]] = {}
    try:
        for level_index, level in enumerate(levels):
            entry = table[level.name]
            totals = reference_goal_totals(
                level,
                n=n,
                seed=seed + 10_000 * level_index,
                move_budget=int(entry["move_budget"]),
                goal_colour=int(entry["goal_colour"]),
                executor=executor,
            )
            rows = []
            for point in entry["curve"]:
                observed = float(np.mean(totals >= int(point["goal_count"])))
                target = target_rate(float(point["E"]))
                rows.append(
                    {
                        "E": float(point["E"]),
                        "goal_count": int(point["goal_count"]),
                        "target": target,
                        "observed": observed,
                        "absolute_error": abs(observed - target),
                    }
                )
            reports[level.name] = rows
    finally:
        if executor is not None:
            executor.shutdown()
    errors = [
        float(row["absolute_error"])
        for rows in reports.values()
        for row in rows
    ]
    return {
        "n": n,
        "seed": seed,
        "mean_absolute_error": float(np.mean(errors)),
        "maximum_absolute_error": float(np.max(errors)),
        "levels": reports,
    }


def collect_win_propensity_data(
    levels: tuple[LevelContext, ...],
    *,
    n_skill_draws: int,
    e_grid: tuple[float, ...],
    n_replicates: int,
    seed: int,
    workers: int = 1,
) -> WinPropensityData:
    """Run a stratified engine design with common random seeds across E."""
    if n_skill_draws < 1 or n_replicates < 1:
        raise ValueError("skill draws and replicates must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    from .scm import TIER_LOGITS, TIER_MOVE_BUDGETS

    rng = np.random.default_rng(seed)
    skill_draws = rng.multivariate_normal(
        np.zeros(len(SKILL_NAMES)), SKILL_COVARIANCE, size=n_skill_draws
    )
    level_indices: list[int] = []
    tier_indices: list[int] = []
    skills: list[np.ndarray] = []
    served_values: list[float] = []
    tasks: list[
        tuple[LevelContext, Difficulty, PlayerSkill, float, int | None, int]
    ] = []
    for skill_index, values in enumerate(skill_draws):
        player = PlayerSkill(tuple(map(float, values)))
        for level_index, level in enumerate(levels):
            for tier_index, baseline in enumerate(TIER_LOGITS):
                difficulty = Difficulty(
                    move_budget=TIER_MOVE_BUDGETS[tier_index],
                    goal_colour=1,
                    goal_count=goal_count_for_E(level.name, baseline),
                    baseline=baseline,
                )
                for replicate in range(n_replicates):
                    episode_seed = int(
                        np.random.SeedSequence(
                            [seed, skill_index, level_index, tier_index, replicate]
                        ).generate_state(1)[0]
                    )
                    for served_difficulty in e_grid:
                        level_indices.append(level_index)
                        tier_indices.append(tier_index)
                        skills.append(values)
                        served_values.append(float(served_difficulty))
                        tasks.append(
                            (
                                level,
                                difficulty,
                                player,
                                float(served_difficulty),
                                None,
                                episode_seed,
                            )
                        )
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        outcomes = _run_episode_tasks(tasks, executor)
    finally:
        if executor is not None:
            executor.shutdown()
    return WinPropensityData(
        level_indices=np.asarray(level_indices, dtype=np.int64),
        tier_indices=np.asarray(tier_indices, dtype=np.int64),
        skills=np.asarray(skills, dtype=np.float64),
        served_difficulty=np.asarray(served_values, dtype=np.float64),
        outcomes=np.asarray(outcomes, dtype=np.float64),
    )


def save_win_propensity_model(
    model: WinPropensityModel,
    path: str | Path = WIN_PROPENSITY_PATH,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(path)
    document = {
        "version": 1,
        "model": model.to_dict(),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(document, indent=2) + "\n")


def load_win_propensity_model(
    path: str | Path = WIN_PROPENSITY_PATH,
) -> WinPropensityModel:
    path = Path(path)
    document = json.loads(path.read_text())
    if document.get("version") != 1:
        raise ValueError("unsupported win-propensity calibration version")
    return WinPropensityModel.from_dict(document["model"])


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from .scm import LEVELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="rollouts per probe")
    parser.add_argument("--win-propensity", action="store_true")
    parser.add_argument("--validate-goals", action="store_true")
    parser.add_argument("--skill-draws", type=int, default=256)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data-in", type=Path, default=None)
    parser.add_argument("--data-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.validate_goals:
        table = load()
        report = validate_goal_count_table(
            table,
            LEVELS,
            n=args.n,
            seed=100_003,
            workers=args.workers,
        )
        rendered = json.dumps(report, indent=2) + "\n"
        if args.out is None:
            print(rendered, end="")
        else:
            args.out.write_text(rendered)
            print(
                f"wrote {args.out}  "
                f"MAE={report['mean_absolute_error']:.4f} "
                f"max={report['maximum_absolute_error']:.4f}"
            )
        return

    if args.win_propensity:
        from .scm import LEVELS, TIER_NAMES

        e_grid = tuple(float(value) for value in np.arange(-2.0, 2.01, 0.5))
        data = (
            load_win_propensity_data(args.data_in)
            if args.data_in is not None
            else collect_win_propensity_data(
                LEVELS,
                n_skill_draws=args.skill_draws,
                e_grid=e_grid,
                n_replicates=args.replicates,
                seed=0,
                workers=args.workers,
            )
        )
        if args.data_out is not None:
            save_win_propensity_data(data, args.data_out)
        train_data, validation_data = split_win_propensity_data(data, seed=0)
        model = fit_win_propensity_model(
            train_data,
            tuple(level.name for level in LEVELS),
            TIER_NAMES,
            skill_directions=np.asarray(
                [
                    level.demand_weights()
                    / np.sqrt(
                        level.demand_weights()
                        @ SKILL_COVARIANCE
                        @ level.demand_weights()
                    )
                    for level in LEVELS
                ]
            ),
            max_iter=args.max_iter,
        )
        train_score = propensity_brier_score(model, train_data)
        validation_score = propensity_brier_score(model, validation_data)
        design = win_propensity_design_summary(data)
        output = args.out or WIN_PROPENSITY_PATH
        save_win_propensity_model(
            model,
            output,
            metadata={
                **design,
                "seed": 0,
                "training_brier_score": train_score,
                "validation_brier_score": validation_score,
                "skill_parameterization": "level_effective_skill",
                "data_source": str(args.data_in) if args.data_in else "engine",
            },
        )
        print(
            f"wrote {output}  ({design['n_observations']} observations, "
            f"Brier train={train_score:.4f} validation={validation_score:.4f})"
        )
        return

    table = calibrate(list(LEVELS), n=args.n, workers=args.workers)
    output = args.out or TABLE_PATH
    save(table, output)

    for name, entry in table.items():
        print(f"\n{name}")
        for point in entry["curve"]:
            print(
                f"  E={point['E']:+.1f}  goal={point['goal_count']:3d}  "
                f"win={point['pass_rate']:.2f}  (target {target_rate(point['E']):.2f})"
            )
    print(f"\nwrote {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
