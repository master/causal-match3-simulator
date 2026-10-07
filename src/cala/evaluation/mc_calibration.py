"""Stage 06: Monte Carlo budget calibration on a separate calibration cohort (seed from configs/seeds.json), frozen criteria, R vs 2R.

One run of the original entry point at R = r_max on the calibration cohort; every smaller candidate R is an exact prefix of that surface
(rollout seeds ``SeedSequence([exogenous_seed, r])`` do not depend on R). For each candidate R compared with 2R on every level:
MC-SE <= mc_se_max at every grid point; max |g_R - g_2R| <= curve_change_max; |Regret_R - Regret_2R| <= regret_change_max; same recommendation
or every candidate optimum's neighbour cost on the 2R surface <= curve_change_max. Final R = smallest candidate passing everything (floor 32).
Raw hazards (grid, players, replicates) of the R = r_max surface are stored for the figures.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import time

import numpy as np

from match3_simulator.evaluation.common import ROOT, Logger, assert_package_import, read_json, relpath, stage_dir, write_done, write_json
from match3_simulator.evaluation.reference_common import ENGINE_CONFIG_PATH, accepted_context, curves_from_surface, engine_config, load_surface_and_risk_set, mc_standard_errors, run_original_benchmark

STAGE = "engine_reference/mc_calibration"


def prefix_surface(surface, replicates: int):
    from match3_simulator.causal_queries import EngineOutcomeSurface

    if replicates > surface.outcomes.shape[2]:
        raise ValueError("prefix longer than the surface")
    return EngineOutcomeSurface(level_name=surface.level_name, grid=surface.grid, player_ids=surface.player_ids, rollout_seeds=surface.rollout_seeds[:, :replicates],
                                outcomes=surface.outcomes[:, :, :replicates], outcome_method=surface.outcome_method,
                                goal_totals=None if surface.goal_totals is None else surface.goal_totals[:, :replicates],
                                completion_margins=None if surface.completion_margins is None else surface.completion_margins[:, :, :replicates])


def _neighbour_costs(grid: np.ndarray, values: np.ndarray, optimum: float) -> dict[str, float | None]:
    g = dict(zip(map(float, grid), map(float, values)))
    out = {}
    for sign, label in ((-0.25, "minus"), (0.25, "plus")):
        e = round(optimum + sign, 6)
        out[label] = (g[e] - g[optimum]) if e in g else None
    return out


def evaluate_candidate(risk_set, full_surface, *, R: int, churn_level, mastery, benchmark, criteria: dict[str, float]) -> dict[str, object]:
    from match3_simulator.causal_queries import select_grid_optimum

    small = curves_from_surface(risk_set, prefix_surface(full_surface, R), churn_level=churn_level, mastery=mastery, n_bootstrap=0, bootstrap_seed=0, benchmark=benchmark)
    big = curves_from_surface(risk_set, prefix_surface(full_surface, 2 * R), churn_level=churn_level, mastery=mastery, n_bootstrap=0, bootstrap_seed=0, benchmark=benchmark)
    se = mc_standard_errors(risk_set, prefix_surface(full_surface, R), churn_level=churn_level, mastery=mastery)
    grid = np.asarray(small["grid"])
    g_small, g_big = np.asarray(small["causal"]), np.asarray(big["causal"])
    curve_change = float(np.max(np.abs(g_small - g_big)))
    regret_change = abs(small["summary"]["added_churn_regret"] - big["summary"]["added_churn_regret"])
    same_recommendation = small["summary"]["interventional_optimum"] == big["summary"]["interventional_optimum"]
    neighbours = {f"e*={opt:+.2f}": _neighbour_costs(grid, g_big, opt) for opt in {small["summary"]["interventional_optimum"], big["summary"]["interventional_optimum"]}}
    flat = all(cost is None or cost <= criteria["curve_change_max"] for block in neighbours.values() for cost in block.values())
    checks = {"mc_se": se["max_mc_se"] <= criteria["mc_se_max"], "curve_change": curve_change <= criteria["curve_change_max"], "regret_change": regret_change <= criteria["regret_change_max"],
              "recommendation": bool(same_recommendation or flat)}
    return {"R": R, "comparator": 2 * R, "passed": all(checks.values()), "checks": checks, "max_mc_se": se["max_mc_se"], "mc_se": se["mc_se"], "player_sampling_se": se["player_sampling_se"],
            "curve_change": curve_change, "regret_R": small["summary"]["added_churn_regret"], "regret_2R": big["summary"]["added_churn_regret"], "regret_change": regret_change,
            "optimum_R": small["summary"]["interventional_optimum"], "optimum_2R": big["summary"]["interventional_optimum"], "obs_R": small["summary"]["observational_recommendation"],
            "obs_2R": big["summary"]["observational_recommendation"], "same_recommendation": bool(same_recommendation), "neighbour_costs_on_2R_surface": neighbours, "flat_minimum": bool(flat),
            "causal_R": g_small.tolist(), "causal_2R": g_big.tolist(), "observational_R": small["observational"], "observational_2R": big["observational"], "grid": grid.tolist()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    assert_package_import()
    config = engine_config()
    if not config.get("frozen_at"):
        raise RuntimeError("configs/engine_reference.json must be frozen before the calibration run")
    equivalence = ROOT / "runs" / "engine_reference" / "equivalence" / "equivalence.json"
    if not equivalence.is_file() or read_json(equivalence)["status"] != "equivalent":
        raise RuntimeError("stage 05 equivalence gate has not passed")
    mc = config["mc_calibration"]
    workers = args.workers or int(config["workers"])
    directory = stage_dir(STAGE)
    log = Logger(STAGE, "mc_calibration")
    run_dir = directory / f"calibration-{mc['cohort_seed']}-R{mc['r_max']}"
    started = time.perf_counter()
    if not (run_dir / "report.json").exists():
        log(f"calibration cohort seed {mc['cohort_seed']}: {mc['n_players']} players, R={mc['r_max']}, workers {workers} (no bootstrap; curves recomputed per prefix)")
        report = run_original_benchmark(output_dir=run_dir, seed=int(mc["cohort_seed"]), n_players=int(mc["n_players"]), rollouts_per_player=int(mc["r_max"]), n_bootstrap=0, workers=workers)
        write_json(run_dir / "report.json", report)
        log(f"R={mc['r_max']} surface done in {report['match3_simulator.evaluation_wrapper']['wall_seconds']:.0f}s wall ({report['n_engine_players']} engine players)")
    else:
        report = read_json(run_dir / "report.json")
        log("calibration surface exists; evaluating candidates")
    context = accepted_context(config["regime"])
    criteria = mc["criteria"]
    candidates = [int(r) for r in mc["r_candidates"] if 2 * int(r) <= int(mc["r_max"])]
    per_level: dict[str, list[dict[str, object]]] = {}
    raw_hazards: dict[str, np.ndarray] = {}
    from match3_simulator.retention import mastery_mismatch_hazard, update_mastery

    for level_name in report["levels"]:
        risk_set, surface = load_surface_and_risk_set(run_dir, level_name)
        churn_level = context["churn"].for_level(level_name)
        rows = []
        for R in candidates:
            result = evaluate_candidate(risk_set, surface, R=R, churn_level=churn_level, mastery=context["mastery"], benchmark=context["benchmark"], criteria=criteria)
            rows.append(result)
            log(f"{level_name} R={R:3d} vs {2 * R:3d}: passed={result['passed']} mc_se={result['max_mc_se']:.4f} dcurve={result['curve_change']:.4f} dregret={result['regret_change']:.4f} "
                f"e*={result['optimum_R']:+.2f}/{result['optimum_2R']:+.2f} obs={result['obs_R']:+.2f}/{result['obs_2R']:+.2f}")
        per_level[level_name] = rows
        mastery_after = update_mastery(np.asarray(risk_set.mastery_before)[None, :, None], surface.outcomes, context["mastery"])
        raw_hazards[level_name] = mastery_mismatch_hazard(mastery_after, churn_level, completion_margin=surface.completion_margins).astype(np.float32)
        raw_hazards[f"{level_name}_player_ids"] = np.asarray(risk_set.player_ids)
        raw_hazards[f"{level_name}_grid"] = np.asarray(surface.grid)
    np.savez_compressed(directory / "raw_hazards.npz", **raw_hazards)
    passing = [R for R in candidates if all(row["passed"] for rows in per_level.values() for row in rows if row["R"] == R)]
    eligible = [R for R in passing if R >= int(mc["floor"])]
    chosen = min(eligible) if eligible else int(mc["r_max"])
    decision = {"schema_version": 1, "chosen_R": chosen, "candidates": candidates, "passing_all_levels": passing, "floor": int(mc["floor"]),
                "rule": mc["selection"], "fallback_used": not eligible, "criteria": criteria, "cohort_seed": mc["cohort_seed"], "n_engine_players": report["n_engine_players"],
                "note": "R = r_max has no 2R comparator; it is selected only when no smaller candidate passes (fallback_used)"}
    write_json(directory / "decision.json", decision)
    write_json(directory / "mc_calibration.json", {"schema_version": 1, "config": mc, "decision": decision, "levels": per_level, "surface_report": relpath(run_dir / "report.json"),
                                                   "seconds": time.perf_counter() - started})
    log(f"chosen R = {chosen} (passing: {passing}; fallback {decision['fallback_used']}) in {time.perf_counter() - started:.0f}s")
    write_done(STAGE, inputs={"engine_reference.json": ENGINE_CONFIG_PATH, "equivalence.json": equivalence}, outputs=[directory / "decision.json", directory / "mc_calibration.json", run_dir / "report.json"])


if __name__ == "__main__":
    main()
