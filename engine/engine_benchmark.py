"""Run and report board-engine landmark churn benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .calibrate import (
    WIN_PROPENSITY_PATH,
    calibration_table_path,
    load_win_propensity_model,
)
from .causal_queries import (
    ASSIGNMENT_SCHEDULE,
    AssignmentSchedule,
    EngineOutcomeSurface,
    LandmarkRiskSet,
    bootstrap_engine_curves,
    compare_engine_curves,
    compare_oracle_curves,
    comparison_report,
    engine_outcome_surface,
    generate_engine_landmark_cohort,
    generate_engine_warmup_panel,
    load_landmark_risk_set,
    materialize_engine_landmark_cohort,
    save_landmark_risk_set,
    save_engine_outcome_surface,
    save_engine_warmup_panel,
)
from .retention import CHURN_SCHEDULE, ChurnConfig, ChurnSchedule, MasteryConfig
from .scm import LEVELS
from .spec import BENCHMARK_CONFIG, BenchmarkConfig


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _subset_risk_set(risk_set: LandmarkRiskSet, rows: np.ndarray) -> LandmarkRiskSet:
    return LandmarkRiskSet(
        level_name=risk_set.level_name,
        skills=risk_set.skills[rows],
        tier_indices=risk_set.tier_indices[rows],
        mastery_before=risk_set.mastery_before[rows],
        assignment_locations=risk_set.assignment_locations[rows],
        assignment_sigma=risk_set.assignment_sigma,
        player_ids=risk_set.player_ids[rows],
        exogenous_seeds=risk_set.exogenous_seeds[rows],
        warmup_outcomes=risk_set.warmup_outcomes[rows],
    )


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _calibration_provenance() -> dict[str, object]:
    path = calibration_table_path()
    return {
        "path": str(path),
        "sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
        ),
    }


def _uses_completion_margin(churn_config: ChurnSchedule) -> bool:
    return any(churn_config.margin_deviation_coefficients) or bool(
        churn_config.margin_overchallenge_deviation_coefficients
        and any(churn_config.margin_overchallenge_deviation_coefficients)
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        len(left_array) < 2
        or np.std(left_array) == 0.0
        or np.std(right_array) == 0.0
    ):
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def engine_level_report(
    risk_set: LandmarkRiskSet,
    surface: EngineOutcomeSurface,
    propensity_model,
    *,
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    mastery_config: MasteryConfig = MasteryConfig(),
    churn_config: ChurnConfig | None = None,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Summarize one engine curve and its surrogate calibration check."""
    grid_benchmark = replace(benchmark, e_grid=tuple(map(float, surface.grid)))
    engine = compare_engine_curves(
        risk_set,
        surface,
        churn_config=churn_config,
        mastery_config=mastery_config,
    )
    uses_margin = bool(
        churn_config
        and (
            churn_config.margin_deviation_coefficient
            or churn_config.margin_overchallenge_deviation_coefficient
        )
    )
    surrogate = (
        None
        if uses_margin
        else compare_oracle_curves(
            risk_set,
            propensity_model,
            churn_config=churn_config,
            mastery_config=mastery_config,
            benchmark=grid_benchmark,
        )
    )
    engine_report = comparison_report(engine, grid_benchmark)
    bootstrap = (
        bootstrap_engine_curves(
            risk_set,
            surface,
            churn_config=churn_config,
            mastery_config=mastery_config,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        if n_bootstrap
        else None
    )
    if bootstrap is not None:
        causal_interval = bootstrap["causal_recommendation_contrast"]
        observational_interval = bootstrap[
            "observational_recommendation_contrast"
        ]
        engine_report["gates"]["causal_interval_excludes_zero"] = (
            causal_interval["lower"] > 0.0
        )
        engine_report["gates"]["observational_interval_excludes_zero"] = (
            observational_interval["upper"] < 0.0
        )
        engine_report["passed"] = all(engine_report["gates"].values())
    return _json_safe(
        {
            "estimator": "board_engine",
            "passed": engine_report["passed"],
            "n_players": len(risk_set.skills),
            "rollouts_per_player": surface.outcomes.shape[2],
            "n_rollouts": surface.outcomes.size,
            "outcome_method": surface.outcome_method,
            "engine": engine_report,
            "bootstrap": bootstrap,
            "surrogate_check": (
                {
                    "role": "unavailable",
                    "reason": (
                        "win-propensity surrogate has no conditional "
                        "completion-margin distribution"
                    ),
                }
                if surrogate is None
                else {
                    "role": "calibration_diagnostic_only",
                    "causal_curve_mae": float(
                        np.mean(np.abs(engine.causal - surrogate.causal))
                    ),
                    "observational_curve_mae": float(
                        np.mean(
                            np.abs(engine.observational - surrogate.observational)
                        )
                    ),
                    "causal_optimum": surrogate.causal_optimum,
                    "observational_optimum": surrogate.observational_optimum,
                    "causal": surrogate.causal.tolist(),
                    "observational": surrogate.observational.tolist(),
                }
            ),
        }
    )


def evaluate_engine_benchmark(
    propensity_model,
    *,
    n_players: int,
    seed: int,
    output_dir: str | Path,
    status: str = "pilot",
    max_engine_players: int | None = None,
    e_grid: tuple[float, ...] = BENCHMARK_CONFIG.e_grid,
    rollouts_per_player: int = 1,
    workers: int = 1,
    n_bootstrap: int = 0,
    reuse_goal_totals: bool = True,
    assignment: AssignmentSchedule = ASSIGNMENT_SCHEDULE,
    churn_config: ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> dict[str, object]:
    """Run engine warm-up and target interventions for every level."""
    if status not in {"pilot", "validation"}:
        raise ValueError("status must be 'pilot' or 'validation'")
    if max_engine_players is not None and max_engine_players < 1:
        raise ValueError("max_engine_players must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cohort = generate_engine_landmark_cohort(
        propensity_model,
        n_players=n_players,
        seed=seed,
        assignment=assignment,
        churn_config=churn_config,
        mastery_config=mastery_config,
        benchmark=benchmark,
        workers=workers,
    )
    n_engine_players = min(
        cohort.n_active_players,
        max_engine_players or cohort.n_active_players,
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, benchmark.landmark_attempt, 2])
    )
    rows = np.sort(
        rng.choice(
            cohort.n_active_players,
            size=n_engine_players,
            replace=False,
        )
    )
    levels: dict[str, object] = {}
    for level_index, full_risk_set in enumerate(cohort.risk_sets):
        risk_set = _subset_risk_set(full_risk_set, rows)
        risk_set_path = save_landmark_risk_set(
            risk_set, output / f"{risk_set.level_name}-risk-set.npz"
        )
        surface = engine_outcome_surface(
            risk_set,
            grid=e_grid,
            rollouts_per_player=rollouts_per_player,
            workers=workers,
            reuse_goal_totals=reuse_goal_totals,
            include_completion_margins=_uses_completion_margin(churn_config),
        )
        surface_path = save_engine_outcome_surface(
            surface, output / f"{risk_set.level_name}-outcomes.npz"
        )
        report = engine_level_report(
            risk_set,
            surface,
            propensity_model,
            benchmark=benchmark,
            mastery_config=mastery_config,
            churn_config=churn_config.for_level(risk_set.level_name),
            n_bootstrap=n_bootstrap,
            bootstrap_seed=int(
                np.random.SeedSequence([seed, level_index, 3]).generate_state(1)[0]
            ),
        )
        report["outcome_artifact"] = {
            "path": surface_path.name,
            "sha256": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
        }
        level = next(item for item in LEVELS if item.name == risk_set.level_name)
        effective_skill = risk_set.skills @ level.demand_weights()
        report["risk_set"] = {
            "artifact": {
                "path": risk_set_path.name,
                "sha256": hashlib.sha256(risk_set_path.read_bytes()).hexdigest(),
            },
            "mastery_mean": float(np.mean(risk_set.mastery_before)),
            "mastery_standard_deviation": float(
                np.std(risk_set.mastery_before)
            ),
            "mastery_quantiles": np.quantile(
                risk_set.mastery_before, [0.05, 0.25, 0.5, 0.75, 0.95]
            ).tolist(),
            "mastery_effective_skill_correlation": _correlation(
                risk_set.mastery_before, effective_skill
            ),
        }
        levels[risk_set.level_name] = report
    configuration = {
        "seed": seed,
        "n_players": n_players,
        "max_engine_players": max_engine_players,
        "e_grid": list(e_grid),
        "rollouts_per_player": rollouts_per_player,
        "n_bootstrap": n_bootstrap,
        "reuse_goal_totals": reuse_goal_totals,
        "calibration_table": _calibration_provenance(),
        "benchmark": asdict(benchmark),
        "mastery": asdict(mastery_config),
        "assignment": asdict(assignment),
        "churn": asdict(churn_config),
    }
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    propensity_sha256 = hashlib.sha256(
        json.dumps(propensity_model.to_dict(), sort_keys=True).encode()
    ).hexdigest()
    document = _json_safe(
        {
            "schema_version": 1,
            "status": status,
            "estimator": "board_engine",
            "risk_set_source": "board_engine",
            "passed": all(bool(report["passed"]) for report in levels.values()),
            "code_sha": _git_revision(),
            "configuration_sha256": configuration_sha256,
            "response_surface_sha256": propensity_sha256,
            "seed": seed,
            "n_initial_players": cohort.n_initial_players,
            "n_landmark_players": cohort.n_active_players,
            "survival_fraction": cohort.survival_fraction,
            "n_engine_players": n_engine_players,
            "e_grid": list(e_grid),
            "rollouts_per_player": rollouts_per_player,
            "workers": workers,
            "n_bootstrap": n_bootstrap,
            "outcome_method": (
                "goal_total_threshold" if reuse_goal_totals else "direct_per_e"
            ),
            "runtime_seconds": time.perf_counter() - started,
            "calibration_table": _calibration_provenance(),
            "benchmark": asdict(benchmark),
            "mastery": asdict(mastery_config),
            "assignment": asdict(assignment),
            "churn": asdict(churn_config),
            "levels": levels,
        }
    )
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n"
    )
    return document


def _risk_set_with_mastery_config(
    risk_set: LandmarkRiskSet,
    mastery_config: MasteryConfig,
) -> LandmarkRiskSet:
    if not risk_set.warmup_outcomes.shape[1]:
        return risk_set
    mastery = np.full(
        len(risk_set.skills), mastery_config.initial, dtype=np.float64
    )
    for attempt in range(risk_set.warmup_outcomes.shape[1]):
        mastery += mastery_config.update_rate * (
            risk_set.warmup_outcomes[:, attempt] - mastery
        )
    return LandmarkRiskSet(
        level_name=risk_set.level_name,
        skills=risk_set.skills,
        tier_indices=risk_set.tier_indices,
        mastery_before=mastery,
        assignment_locations=risk_set.assignment_locations,
        assignment_sigma=risk_set.assignment_sigma,
        player_ids=risk_set.player_ids,
        exogenous_seeds=risk_set.exogenous_seeds,
        warmup_outcomes=risk_set.warmup_outcomes,
    )


def evaluate_persisted_engine_risk_sets(
    propensity_model,
    *,
    risk_set_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    status: str = "pilot",
    max_engine_players: int | None = None,
    e_grid: tuple[float, ...] = BENCHMARK_CONFIG.e_grid,
    rollouts_per_player: int = 1,
    workers: int = 1,
    n_bootstrap: int = 0,
    reuse_goal_totals: bool = True,
    churn_config: ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> dict[str, object]:
    """Generate target interventions from persisted engine landmark records."""
    if status not in {"pilot", "validation"}:
        raise ValueError("status must be 'pilot' or 'validation'")
    if max_engine_players is not None and max_engine_players < 1:
        raise ValueError("max_engine_players must be positive")
    source = Path(risk_set_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    risk_sets = tuple(
        _risk_set_with_mastery_config(
            load_landmark_risk_set(source / f"{level.name}-risk-set.npz"),
            mastery_config,
        )
        for level in LEVELS
    )
    reference_ids = risk_sets[0].player_ids
    if any(
        not np.array_equal(risk_set.player_ids, reference_ids)
        for risk_set in risk_sets[1:]
    ):
        raise ValueError("persisted risk sets must contain the same players")
    n_engine_players = min(
        len(reference_ids), max_engine_players or len(reference_ids)
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, benchmark.landmark_attempt, 7])
    )
    rows = np.sort(
        rng.choice(len(reference_ids), size=n_engine_players, replace=False)
    )
    levels: dict[str, object] = {}
    for level_index, full_risk_set in enumerate(risk_sets):
        risk_set = _subset_risk_set(full_risk_set, rows)
        risk_path = save_landmark_risk_set(
            risk_set, output / f"{risk_set.level_name}-risk-set.npz"
        )
        surface = engine_outcome_surface(
            risk_set,
            grid=e_grid,
            rollouts_per_player=rollouts_per_player,
            workers=workers,
            reuse_goal_totals=reuse_goal_totals,
            include_completion_margins=_uses_completion_margin(churn_config),
        )
        surface_path = save_engine_outcome_surface(
            surface, output / f"{risk_set.level_name}-outcomes.npz"
        )
        report = engine_level_report(
            risk_set,
            surface,
            propensity_model,
            benchmark=benchmark,
            mastery_config=mastery_config,
            churn_config=churn_config.for_level(risk_set.level_name),
            n_bootstrap=n_bootstrap,
            bootstrap_seed=int(
                np.random.SeedSequence([seed, level_index, 8]).generate_state(1)[0]
            ),
        )
        report["risk_set_artifact"] = {
            "path": risk_path.name,
            "sha256": hashlib.sha256(risk_path.read_bytes()).hexdigest(),
        }
        report["outcome_artifact"] = {
            "path": surface_path.name,
            "sha256": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
        }
        levels[risk_set.level_name] = report
    document = _json_safe(
        {
            "schema_version": 1,
            "status": status,
            "estimator": "board_engine",
            "risk_set_source": "persisted_board_engine",
            "source_directory": str(source),
            "code_sha": _git_revision(),
            "seed": seed,
            "n_landmark_players": len(reference_ids),
            "n_engine_players": n_engine_players,
            "e_grid": list(e_grid),
            "rollouts_per_player": rollouts_per_player,
            "workers": workers,
            "n_bootstrap": n_bootstrap,
            "outcome_method": (
                "goal_total_threshold" if reuse_goal_totals else "direct_per_e"
            ),
            "runtime_seconds": time.perf_counter() - started,
            "calibration_table": _calibration_provenance(),
            "benchmark": asdict(benchmark),
            "mastery": asdict(mastery_config),
            "churn": asdict(churn_config),
            "passed": all(bool(report["passed"]) for report in levels.values()),
            "levels": levels,
        }
    )
    (output / "report.json").write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n"
    )
    return document


def rescore_engine_benchmark(
    propensity_model,
    *,
    input_dir: str | Path,
    output_path: str | Path,
    mastery_config: MasteryConfig,
    churn_config: ChurnSchedule,
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, object]:
    """Rescore persisted engine sufficient statistics without gameplay runs."""
    from .causal_queries import (
        load_engine_outcome_surface,
        load_landmark_risk_set,
    )

    input_directory = Path(input_dir)
    levels: dict[str, object] = {}
    for level_index, level in enumerate(LEVELS):
        risk_set = load_landmark_risk_set(
            input_directory / f"{level.name}-risk-set.npz"
        )
        surface = load_engine_outcome_surface(
            input_directory / f"{level.name}-outcomes.npz"
        )
        risk_set = _risk_set_with_mastery_config(risk_set, mastery_config)
        report = engine_level_report(
            risk_set,
            surface,
            propensity_model,
            benchmark=benchmark,
            mastery_config=mastery_config,
            churn_config=churn_config.for_level(level.name),
            n_bootstrap=n_bootstrap,
            bootstrap_seed=int(
                np.random.SeedSequence([seed, level_index, 5]).generate_state(1)[0]
            ),
        )
        levels[level.name] = report
    document = _json_safe(
        {
            "schema_version": 1,
            "status": "rescored",
            "estimator": "board_engine",
            "source_directory": str(input_directory),
            "code_sha": _git_revision(),
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "calibration_table": _calibration_provenance(),
            "benchmark": asdict(benchmark),
            "mastery": asdict(mastery_config),
            "churn": asdict(churn_config),
            "passed": all(bool(report["passed"]) for report in levels.values()),
            "levels": levels,
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document


def generate_engine_risk_set_artifacts(
    propensity_model,
    *,
    n_players: int,
    seed: int,
    output_dir: str | Path,
    workers: int = 1,
    assignment: AssignmentSchedule = ASSIGNMENT_SCHEDULE,
    churn_config: ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> dict[str, object]:
    """Run only pre-landmark engine histories and persist the full risk set."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    panel = generate_engine_warmup_panel(
        propensity_model,
        n_players=n_players,
        seed=seed,
        assignment=assignment,
        benchmark=benchmark,
        workers=workers,
    )
    panel_path = save_engine_warmup_panel(panel, output / "warmup-panel.npz")
    cohort = materialize_engine_landmark_cohort(
        panel,
        assignment=assignment,
        churn_config=churn_config,
        mastery_config=mastery_config,
        benchmark=benchmark,
    )
    if not cohort.n_active_players:
        raise RuntimeError("no players survived to the landmark attempt")
    levels: dict[str, object] = {}
    for risk_set in cohort.risk_sets:
        path = save_landmark_risk_set(
            risk_set, output / f"{risk_set.level_name}-risk-set.npz"
        )
        level = next(item for item in LEVELS if item.name == risk_set.level_name)
        effective_skill = risk_set.skills @ level.demand_weights()
        levels[risk_set.level_name] = {
            "artifact": {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "mastery_mean": float(np.mean(risk_set.mastery_before)),
            "mastery_standard_deviation": float(np.std(risk_set.mastery_before)),
            "mastery_quantiles": np.quantile(
                risk_set.mastery_before, [0.05, 0.25, 0.5, 0.75, 0.95]
            ).tolist(),
            "mastery_effective_skill_correlation": _correlation(
                risk_set.mastery_before, effective_skill
            ),
        }
    document = _json_safe(
        {
            "schema_version": 1,
            "status": "calibration",
            "artifact_type": "engine_landmark_risk_set",
            "code_sha": _git_revision(),
            "seed": seed,
            "n_initial_players": cohort.n_initial_players,
            "n_landmark_players": cohort.n_active_players,
            "survival_fraction": cohort.survival_fraction,
            "workers": workers,
            "runtime_seconds": time.perf_counter() - started,
            "warmup_panel": {
                "path": panel_path.name,
                "sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                "players": len(panel.player_ids),
            },
            "calibration_table": _calibration_provenance(),
            "benchmark": asdict(benchmark),
            "mastery": asdict(mastery_config),
            "assignment": asdict(assignment),
            "churn": asdict(churn_config),
            "levels": levels,
        }
    )
    (output / "risk-set-report.json").write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n"
    )
    return document


def main() -> None:  # pragma: no cover - exercised through CLI smoke runs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=64)
    parser.add_argument("--engine-players", type=int, default=None)
    parser.add_argument("--seed", type=int, default=BENCHMARK_CONFIG.calibration_seeds[0])
    parser.add_argument("--rollouts-per-player", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument(
        "--e-grid",
        type=float,
        nargs="+",
        default=list(BENCHMARK_CONFIG.e_grid),
    )
    parser.add_argument("--status", choices=("pilot", "validation"), default="pilot")
    parser.add_argument("--propensity", type=Path, default=WIN_PROPENSITY_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/engine-pilot"))
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument("--risk-set-dir", type=Path, default=None)
    parser.add_argument("--direct-per-e", action="store_true")
    parser.add_argument("--mastery-initial", type=float, default=None)
    parser.add_argument("--mastery-target", type=float, default=None)
    parser.add_argument("--mastery-update-rate", type=float, default=None)
    parser.add_argument("--churn-intercepts", type=float, nargs=3, default=None)
    parser.add_argument("--churn-curvatures", type=float, nargs=3, default=None)
    parser.add_argument(
        "--churn-overchallenge-curvatures", type=float, nargs=3, default=None
    )
    parser.add_argument(
        "--churn-margin-curvatures", type=float, nargs=3, default=None
    )
    parser.add_argument(
        "--churn-margin-overchallenge-curvatures",
        type=float,
        nargs=3,
        default=None,
    )
    parser.add_argument("--churn-margin-targets", type=float, nargs=3, default=None)
    parser.add_argument("--assignment-gains", type=float, nargs=3, default=None)
    parser.add_argument("--assignment-sigmas", type=float, nargs=3, default=None)
    args = parser.parse_args()
    if args.warmup_only and args.risk_set_dir is not None:
        parser.error("--warmup-only and --risk-set-dir cannot be combined")

    model = load_win_propensity_model(args.propensity)
    mastery_defaults = MasteryConfig()
    mastery_config = MasteryConfig(
        initial=(
            mastery_defaults.initial
            if args.mastery_initial is None
            else args.mastery_initial
        ),
        update_rate=(
            mastery_defaults.update_rate
            if args.mastery_update_rate is None
            else args.mastery_update_rate
        ),
    )
    churn_config = ChurnSchedule(
        intercepts=(
            CHURN_SCHEDULE.intercepts
            if args.churn_intercepts is None
            else tuple(args.churn_intercepts)
        ),
        deviation_coefficients=(
            CHURN_SCHEDULE.deviation_coefficients
            if args.churn_curvatures is None
            else tuple(args.churn_curvatures)
        ),
        overchallenge_deviation_coefficients=(
            CHURN_SCHEDULE.overchallenge_deviation_coefficients
            if args.churn_overchallenge_curvatures is None
            else tuple(args.churn_overchallenge_curvatures)
        ),
        mastery_target=(
            mastery_config.initial
            if args.mastery_target is None
            else args.mastery_target
        ),
        margin_deviation_coefficients=(
            CHURN_SCHEDULE.margin_deviation_coefficients
            if args.churn_margin_curvatures is None
            else tuple(args.churn_margin_curvatures)
        ),
        margin_overchallenge_deviation_coefficients=(
            CHURN_SCHEDULE.margin_overchallenge_deviation_coefficients
            if args.churn_margin_overchallenge_curvatures is None
            else tuple(args.churn_margin_overchallenge_curvatures)
        ),
        margin_targets=(
            CHURN_SCHEDULE.margin_targets
            if args.churn_margin_targets is None
            else tuple(args.churn_margin_targets)
        ),
    )
    assignment = AssignmentSchedule(
        skill_gains=(
            ASSIGNMENT_SCHEDULE.skill_gains
            if args.assignment_gains is None
            else tuple(args.assignment_gains)
        ),
        sigmas=(
            ASSIGNMENT_SCHEDULE.sigmas
            if args.assignment_sigmas is None
            else tuple(args.assignment_sigmas)
        ),
    )
    if args.warmup_only:
        report = generate_engine_risk_set_artifacts(
            model,
            n_players=args.players,
            seed=args.seed,
            output_dir=args.out,
            workers=args.workers,
            mastery_config=mastery_config,
            churn_config=churn_config,
            assignment=assignment,
        )
        print(
            f"wrote {args.out / 'risk-set-report.json'}  "
            f"players={report['n_landmark_players']}"
        )
        return
    if args.risk_set_dir is not None:
        report = evaluate_persisted_engine_risk_sets(
            model,
            risk_set_dir=args.risk_set_dir,
            output_dir=args.out,
            seed=args.seed,
            status=args.status,
            max_engine_players=args.engine_players,
            e_grid=tuple(args.e_grid),
            rollouts_per_player=args.rollouts_per_player,
            workers=args.workers,
            n_bootstrap=args.bootstrap,
            reuse_goal_totals=not args.direct_per_e,
            mastery_config=mastery_config,
            churn_config=churn_config,
        )
        print(
            f"wrote {args.out / 'report.json'}  "
            f"players={report['n_engine_players']}  passed={report['passed']}"
        )
        if args.require_pass and not report["passed"]:
            raise SystemExit(1)
        return
    report = evaluate_engine_benchmark(
        model,
        n_players=args.players,
        max_engine_players=args.engine_players,
        seed=args.seed,
        output_dir=args.out,
        status=args.status,
        e_grid=tuple(args.e_grid),
        rollouts_per_player=args.rollouts_per_player,
        workers=args.workers,
        n_bootstrap=args.bootstrap,
        reuse_goal_totals=not args.direct_per_e,
        mastery_config=mastery_config,
        churn_config=churn_config,
        assignment=assignment,
    )
    print(
        f"wrote {args.out / 'report.json'}  "
        f"players={report['n_engine_players']}  passed={report['passed']}"
    )
    if args.require_pass and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()