"""Shared pieces of the engine-reference stages: accepted configuration resolution, the original entry point wrapper, curve summaries.

Everything that touches the simulator goes through the original pinned functions: ``release.load_accepted_spec`` -> ``verify_release_inputs``
-> ``resolved_configs`` -> ``engine_benchmark.evaluate_engine_benchmark`` (primary protocol) with the accepted quota calibration selected
through ``MATCH3_CALIBRATION_PATH`` before any process pool is created. Tie rule, bootstrap and gates are the original functions, unchanged.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import os
from pathlib import Path
import time

import numpy as np

from match3_simulator.evaluation.common import CONFIGS, PACKAGE, read_json, set_accepted_calibration

ENGINE_CONFIG_PATH = CONFIGS / "engine_reference.json"


def accepted_context(regime: str = "natural") -> dict[str, object]:
    """Resolve the accepted configuration through the original functions (and assert the calibration table selection)."""
    from match3_simulator import release
    from match3_simulator.calibrate import calibration_table_path, load_win_propensity_model
    from match3_simulator.causal_queries import AssignmentSchedule

    calibration = set_accepted_calibration()
    spec = release.load_accepted_spec()
    release.verify_release_inputs(spec)
    benchmark, churn, mastery, gains, sigmas = release.resolved_configs(spec, regime)
    if calibration_table_path().resolve() != calibration.resolve():
        raise RuntimeError("calibration table path does not resolve to the accepted table")
    assignment = AssignmentSchedule(skill_gains=tuple(gains), sigmas=tuple(sigmas))
    model = load_win_propensity_model(release.WIN_PROPENSITY_PATH)
    return {"spec": spec, "benchmark": benchmark, "churn": churn, "mastery": mastery, "assignment": assignment, "gains": gains, "sigmas": sigmas, "model": model,
            "calibration_path": calibration, "accepted_sha256": release.sha256(release.ACCEPTED_SPEC_PATH)}


def _worker_calibration_path(_: int = 0) -> str:
    from match3_simulator.calibrate import calibration_table_path

    return str(calibration_table_path())


def assert_worker_calibration(workers: int) -> dict[str, object]:
    """Spawn-check: child processes (fork context, as the original pools use) resolve the accepted calibration table."""
    expected = str((PACKAGE / "accepted_calibration.json").resolve())
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        seen = set(pool.map(_worker_calibration_path, [0] * max(1, min(workers, 4))))
    if seen != {expected}:
        raise RuntimeError(f"worker calibration path mismatch: {seen} != {expected}")
    return {"workers_checked": max(1, min(workers, 4)), "path": "match3_simulator/accepted_calibration.json", "env": "MATCH3_CALIBRATION_PATH"}


def run_original_benchmark(*, output_dir: Path, seed: int, n_players: int, rollouts_per_player: int, n_bootstrap: int, workers: int, reuse_goal_totals: bool = True,
                           regime: str = "natural", status: str = "validation", e_grid: tuple[float, ...] | None = None) -> dict[str, object]:
    """Thin wrapper over the original entry point with the accepted configuration injected explicitly; returns the report document."""
    from match3_simulator.engine_benchmark import evaluate_engine_benchmark

    context = accepted_context(regime)
    worker_check = assert_worker_calibration(workers)
    started = time.perf_counter()
    report = evaluate_engine_benchmark(
        context["model"],
        n_players=n_players,
        seed=seed,
        output_dir=output_dir,
        status=status,
        e_grid=tuple(context["benchmark"].e_grid) if e_grid is None else tuple(e_grid),
        rollouts_per_player=rollouts_per_player,
        workers=workers,
        n_bootstrap=n_bootstrap,
        reuse_goal_totals=reuse_goal_totals,
        assignment=context["assignment"],
        churn_config=context["churn"],
        mastery_config=context["mastery"],
        benchmark=context["benchmark"],
    )
    report["match3_simulator.evaluation_wrapper"] = {"regime": regime, "accepted_benchmark_sha256": context["accepted_sha256"], "resolved_via": "release.load_accepted_spec -> verify_release_inputs -> resolved_configs",
                                  "calibration_env": os.environ.get("MATCH3_CALIBRATION_PATH", "").split("/")[-1], "worker_calibration_check": worker_check,
                                  "wall_seconds": time.perf_counter() - started, "assignment": asdict(context["assignment"]), "mastery": asdict(context["mastery"]), "churn": asdict(context["churn"])}
    return report


def load_surface_and_risk_set(directory: Path, level_name: str):
    from match3_simulator.causal_queries import load_engine_outcome_surface, load_landmark_risk_set

    return load_landmark_risk_set(directory / f"{level_name}-risk-set.npz"), load_engine_outcome_surface(directory / f"{level_name}-outcomes.npz")


def curves_from_surface(risk_set, surface, *, churn_level, mastery, n_bootstrap: int, bootstrap_seed: int, benchmark) -> dict[str, object]:
    """Original curve comparison + report + bootstrap on a (risk set, surface) pair; the summary table rows are derived from these arrays only."""
    from dataclasses import replace

    from match3_simulator.causal_queries import bootstrap_engine_curves, compare_engine_curves, comparison_report, select_grid_optimum

    comparison = compare_engine_curves(risk_set, surface, churn_config=churn_level, mastery_config=mastery)
    grid_benchmark = replace(benchmark, e_grid=tuple(map(float, surface.grid)))
    report = comparison_report(comparison, grid_benchmark)
    boot = bootstrap_engine_curves(risk_set, surface, churn_config=churn_level, mastery_config=mastery, n_bootstrap=n_bootstrap, seed=bootstrap_seed) if n_bootstrap else None
    if boot is not None:
        report["gates"]["causal_interval_excludes_zero"] = boot["causal_recommendation_contrast"]["lower"] > 0.0
        report["gates"]["observational_interval_excludes_zero"] = boot["observational_recommendation_contrast"]["upper"] < 0.0
        report["passed"] = all(report["gates"].values())
    grid = np.asarray(comparison.grid)
    causal = np.asarray(comparison.causal)
    observational = np.asarray(comparison.observational)
    e_star = select_grid_optimum(grid, causal)
    e_obs = select_grid_optimum(grid, observational)
    g = dict(zip(map(float, grid), map(float, causal)))
    summary = {"observational_recommendation": e_obs, "interventional_optimum": e_star, "absolute_gap": abs(e_obs - e_star), "added_churn_regret": g[e_obs] - g[e_star],
               "g_at_observational_recommendation": g[e_obs], "g_at_optimum": g[e_star], "tie_rule": "causal_queries.select_grid_optimum"}
    return {"report": report, "bootstrap": boot, "summary": summary, "grid": grid.tolist(), "causal": causal.tolist(), "observational": observational.tolist()}


def mc_standard_errors(risk_set, surface, *, churn_level, mastery) -> dict[str, object]:
    """Monte Carlo standard error of the causal curve at every grid point: sqrt(sum_p s_p^2 / R) / N (players fixed; replicate variance only)."""
    from match3_simulator.causal_queries import _engine_player_hazards
    from match3_simulator.retention import mastery_mismatch_hazard, update_mastery

    mastery_before = np.asarray(risk_set.mastery_before)[None, :, None]
    mastery_after = update_mastery(mastery_before, surface.outcomes, mastery)
    hazards = mastery_mismatch_hazard(mastery_after, churn_level, completion_margin=surface.completion_margins)  # (grid, players, replicates)
    n_players, replicates = hazards.shape[1], hazards.shape[2]
    per_player_var = hazards.var(axis=2, ddof=1) if replicates > 1 else np.zeros(hazards.shape[:2])
    se = np.sqrt(per_player_var.sum(axis=1) / replicates) / n_players
    player_means = _engine_player_hazards(risk_set, surface, churn_level, mastery)
    return {"mc_se": se.tolist(), "max_mc_se": float(se.max()), "player_sampling_se": (player_means.std(axis=1, ddof=1) / np.sqrt(n_players)).tolist(), "replicates": int(replicates), "n_players": int(n_players)}


def engine_config() -> dict[str, object]:
    return read_json(ENGINE_CONFIG_PATH)


__all__ = ["ENGINE_CONFIG_PATH", "accepted_context", "assert_worker_calibration", "curves_from_surface", "engine_config", "load_surface_and_risk_set", "mc_standard_errors", "run_original_benchmark"]
