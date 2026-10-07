"""Stage 08: corrected logged-level gold (v3) on the held-out test split, from original simulator primitives.

Separate second stage: the per-player reference later CaLA estimator work consumes. Population = test-split players with a logged attempt 20,
at their logged level and tier, both regimes; skill and mastery_before from the privileged oracle table (``--audit``); opening = the logged
attempt-20 step-0 state initialised through the original ``sample_S0(value=State)``; play = the original action decision / transition / outcome
nodes with the full state (board + specials) under the quota-invariant threshold protocol (one full-budget game per replicate); outcomes,
margins, hazards, curves, bootstrap and the tie rule are the original functions. Never mixed with the primary tables.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import time

import numpy as np

from match3_simulator.evaluation.common import CONFIGS, DATA, ROOT, Logger, assert_package_import, read_json, relpath, set_accepted_calibration, stage_dir, write_done, write_json
from match3_simulator.evaluation.dataset_checks import _episodes, _transitions
from match3_simulator.evaluation.mc_calibration import evaluate_candidate, prefix_surface
from match3_simulator.evaluation.reference_common import ENGINE_CONFIG_PATH, accepted_context, curves_from_surface, engine_config

STAGE = "engine_reference/logged_level"
CONFIG_PATH = CONFIGS / "logged_gold.json"
FULL_BUDGET_QUOTA = 10_000
_WORKER: dict[str, object] = {}


# ------------------------------------------------------------------ population ----


def logged_landmark_population(regime: str, split: str, *, audit: bool) -> list[dict[str, object]]:
    """Test/validation-split players with a logged attempt 20: logged level, tier, E, opening state, and (audit) skill + mastery_before."""
    splits = read_json(DATA / "splits.json")
    wanted = set(int(v) for v in splits[split])
    episodes, _ = _episodes(DATA, regime)
    arrays = _transitions(DATA, regime)
    key = arrays["player_id"].astype(np.int64) * 1000 + arrays["attempt_id"].astype(np.int64)
    opening_rows = np.flatnonzero((arrays["attempt_id"] == 20) & (arrays["step_id"] == 0))
    opening_by_player = {int(arrays["player_id"][r]): int(r) for r in opening_rows}
    oracle: dict[int, dict[str, str]] = {}
    if audit:
        manifest = read_json(DATA / "release-manifest.json")
        for shard in manifest["regimes"][regime]["shards"]:
            with (DATA / Path(shard["manifest"]).parent / "oracle" / "attempts.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if int(row["attempt_id"]) == 20:
                        oracle[int(row["player_id"])] = row
    population = []
    for (pid, aid), row in sorted(episodes.items()):
        if aid != 20 or pid not in wanted or row["active_before"] != "1":
            continue
        if pid not in opening_by_player:
            population.append({"player_id": pid, "status": "no_logged_opening"})
            continue
        r = opening_by_player[pid]
        item = {"player_id": pid, "status": "ok", "level": row["level"], "tier": row["tier"], "E": float(row["E"]), "served_goal_count": int(row["served_goal_count"]), "move_budget": int(row["move_budget"]),
                "R": int(row["R"]), "churn_after": int(row["churn_after"]), "board": arrays["board_before"][r].astype(np.int8), "specials": arrays["specials_before"][r].astype(np.int8),
                "moves_left": int(arrays["moves_left"][r]), "goals_left": int(arrays["goals_left"][r]), "goal_colour": int(arrays["goal_colour"][r]), "off_grid": abs(float(row["E"])) > 2.0,
                "episode_seed": None}
        if audit:
            o = oracle[pid]
            item["skill"] = tuple(float(o[f"k_{name}"]) for name in ("search", "pattern", "planning", "strategy"))
            item["mastery_before"] = float(o["mastery_before"])
        population.append(item)
    return population


# ------------------------------------------------------------------ simulator primitives ----


def initial_state_from_logged(item: dict[str, object], *, goals_left: int | None = None):
    """Build the original ``State`` from a logged opening and pass it through the original ``sample_S0(value=...)`` node."""
    from match3_simulator.scm import LEVELS, TIER_LOGITS, TIER_MOVE_BUDGETS, TIER_NAMES, sample_S0
    from match3_simulator.spec import Difficulty, State

    level = next(l for l in LEVELS if l.name == item["level"])
    tier = TIER_NAMES.index(item["tier"])
    if TIER_MOVE_BUDGETS[tier] != item["move_budget"]:
        raise ValueError("logged move budget does not match the tier")
    difficulty = Difficulty(move_budget=TIER_MOVE_BUDGETS[tier], goal_colour=item["goal_colour"], goal_count=item["served_goal_count"], baseline=TIER_LOGITS[tier])
    state = State(board=np.asarray(item["board"], dtype=np.int8).copy(), moves_left=int(item["moves_left"]), goals_left=int(item["goals_left"] if goals_left is None else goals_left),
                  goal_colour=int(item["goal_colour"]), t=0, specials=np.asarray(item["specials"], dtype=np.int8).copy())
    return level, difficulty, sample_S0(level, difficulty, float(item["E"]), value=state)


def play_from_state(state, level, difficulty, player, *, max_steps: int | None = None):
    """The original play loop (``ground_truth_model`` body after S0): action decision, transition, outcome; returns (states, actions, transitions, R)."""
    from match3_simulator.board import legal_moves
    from match3_simulator.scm import _sample_action_decision, sample_R, sample_S_next

    states = [state]
    actions = []
    transitions = []
    budget = difficulty.move_budget if max_steps is None else min(difficulty.move_budget, max_steps)
    for _ in range(budget):
        if state.terminal:
            break
        action, _ = _sample_action_decision(state, player)
        if action is None:
            break
        if action not in legal_moves(state.board):
            raise ValueError(f"action policy returned illegal move {action}")
        state, transition = sample_S_next(state, action, level)
        actions.append(action)
        transitions.append(transition)
        states.append(state)
    R = sample_R(states[-1])
    return states, actions, transitions, R


def goal_profile_from_logged(item: dict[str, object], player, rollout_seed: int) -> np.ndarray:
    """One quota-invariant full-budget game from the logged opening; cumulative goal progress after each move (padded to the budget)."""
    import pyro

    pyro.set_rng_seed(int(rollout_seed))
    level, difficulty, state = initial_state_from_logged(item, goals_left=FULL_BUDGET_QUOTA)
    _, _, transitions, _ = play_from_state(state, level, difficulty, player)
    increments = np.asarray([t.goal_cleared for t in transitions], dtype=np.int64)
    progile = np.empty(difficulty.move_budget, dtype=np.int64)
    if len(increments):
        progile[: len(increments)] = np.cumsum(increments)
        progile[len(increments):] = progile[len(increments) - 1]
    else:
        progile.fill(0)
    return progile


def _init_worker() -> None:
    set_accepted_calibration()


def _profiles_task(payload: tuple[dict[str, object], tuple[float, ...], list[int]]) -> np.ndarray:
    """Goal profiles of one player's replicates, padded with the final total to the largest tier budget (as causal_queries.engine_outcome_surface does)."""
    from match3_simulator.scm import TIER_MOVE_BUDGETS
    from match3_simulator.spec import PlayerSkill

    item, skill, seeds = payload
    player = PlayerSkill(tuple(skill))
    width = max(TIER_MOVE_BUDGETS)
    out = np.empty((len(seeds), width), dtype=np.int64)
    for i, s in enumerate(seeds):
        profile = goal_profile_from_logged(item, player, s)
        out[i, : len(profile)] = profile
        out[i, len(profile):] = profile[-1]
    return out


# ------------------------------------------------------------------ surfaces and curves ----


def rollout_seed_matrix(player_ids: np.ndarray, *, seed: int, replicates: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray([np.random.SeedSequence([seed, int(pid), 20]).generate_state(1)[0] for pid in player_ids], dtype=np.uint32)
    seeds = np.asarray([[np.random.SeedSequence([int(b), r]).generate_state(1)[0] for r in range(replicates)] for b in base], dtype=np.uint32)
    return base, seeds


def surface_from_profiles(level_name: str, grid: np.ndarray, player_ids: np.ndarray, rollout_seeds: np.ndarray, profiles: np.ndarray, move_budgets: np.ndarray):
    """EngineOutcomeSurface from (players, replicates, budget) goal profiles — the same arithmetic as the original engine_outcome_surface."""
    from match3_simulator.calibrate import goal_count_for_E
    from match3_simulator.causal_queries import EngineOutcomeSurface

    quotas = np.asarray([goal_count_for_E(level_name, float(e)) for e in grid], dtype=np.int64)
    if np.any(quotas <= 0):
        raise ValueError("calibrated goal quotas must be positive")
    goal_totals = profiles[:, :, -1]
    outcomes = (goal_totals[None, :, :] >= quotas[:, None, None]).astype(np.int8)
    margins = np.empty_like(outcomes, dtype=np.float64)
    for gi, quota in enumerate(quotas):
        reached = profiles >= quota
        won = reached.any(axis=2)
        first_step = reached.argmax(axis=2) + 1
        win_margin = (move_budgets[:, None] - first_step) / move_budgets[:, None]
        loss_margin = -(quota - goal_totals).clip(min=0) / float(quota)
        margins[gi] = np.where(won, win_margin, loss_margin)
    return EngineOutcomeSurface(level_name=level_name, grid=grid, player_ids=player_ids, rollout_seeds=rollout_seeds, outcomes=outcomes, outcome_method="goal_total_threshold",
                                goal_totals=goal_totals, completion_margins=margins), quotas


def risk_set_for(level_name: str, items: list[dict[str, object]], *, context) -> object:
    from match3_simulator.causal_queries import LandmarkRiskSet, _effective_skills
    from match3_simulator.scm import LEVELS, TIER_LOGITS, TIER_NAMES

    level = next(l for l in LEVELS if l.name == level_name)
    level_assignment = context["assignment"].for_level(level_name)
    skills = np.asarray([it["skill"] for it in items], dtype=np.float64)
    tiers = np.asarray([TIER_NAMES.index(it["tier"]) for it in items], dtype=np.int8)
    locations = np.asarray(TIER_LOGITS, dtype=np.float64)[tiers] + level_assignment.skill_gain * _effective_skills(skills, level)
    return LandmarkRiskSet(level_name=level_name, skills=skills, tier_indices=tiers, mastery_before=np.asarray([it["mastery_before"] for it in items]), assignment_locations=locations,
                           assignment_sigma=level_assignment.sigma, player_ids=np.asarray([it["player_id"] for it in items], dtype=np.int64))


def compute_gold(regime: str, split: str, *, replicates: int, seed: int, workers: int, n_bootstrap: int, log, out_dir: Path) -> dict[str, object]:
    from match3_simulator.scm import LEVELS, TIER_MOVE_BUDGETS, TIER_NAMES

    context = accepted_context(regime)
    population = logged_landmark_population(regime, split, audit=True)
    usable = [it for it in population if it["status"] == "ok"]
    log(f"{regime}/{split}: {len(usable)} players with a logged attempt 20 and opening ({len(population) - len(usable)} without), R={replicates}, seed {seed}")
    grid = np.asarray(context["benchmark"].e_grid, dtype=np.float64)
    levels_out: dict[str, object] = {}
    surfaces = {}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for level in LEVELS:
            items = [it for it in usable if it["level"] == level.name]
            if not items:
                continue
            player_ids = np.asarray([it["player_id"] for it in items], dtype=np.int64)
            base, seeds = rollout_seed_matrix(player_ids, seed=seed, replicates=replicates)
            tasks = [(it, it["skill"], [int(s) for s in seeds[i]]) for i, it in enumerate(items)]
            t0 = time.perf_counter()
            profiles = np.stack(list(pool.map(_profiles_task, tasks, chunksize=1)))
            log(f"{regime}/{split}/{level.name}: {len(items)} players x {replicates} games in {time.perf_counter() - t0:.0f}s")
            budgets = np.asarray([TIER_MOVE_BUDGETS[TIER_NAMES.index(it["tier"])] for it in items])
            surface, quotas = surface_from_profiles(level.name, grid, player_ids, seeds, profiles, budgets)
            risk_set = risk_set_for(level.name, items, context=context)
            bootstrap_seed = int(np.random.SeedSequence([seed, LEVELS.index(level), 3]).generate_state(1)[0])
            derived = curves_from_surface(risk_set, surface, churn_level=context["churn"].for_level(level.name), mastery=context["mastery"], n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed, benchmark=context["benchmark"])
            np.savez_compressed(out_dir / f"{regime}-{split}-{level.name}-surface.npz", grid=grid, player_ids=player_ids, rollout_seeds=seeds, exogenous_seeds=base, outcomes=surface.outcomes,
                                goal_totals=surface.goal_totals, completion_margins=surface.completion_margins, profiles=profiles.astype(np.int16), quotas=quotas,
                                mastery_before=risk_set.mastery_before, tier_indices=risk_set.tier_indices, logged_E=np.asarray([it["E"] for it in items]), off_grid=np.asarray([it["off_grid"] for it in items]))
            surfaces[level.name] = (risk_set, surface)
            per_player = {"player_ids": player_ids.tolist(), "logged_E": [it["E"] for it in items], "logged_tier": [it["tier"] for it in items], "off_grid_abs_E_gt_2": [bool(it["off_grid"]) for it in items],
                          "logged_R": [it["R"] for it in items], "logged_churn_after": [it["churn_after"] for it in items]}
            levels_out[level.name] = {"n_players": len(items), "replicates": replicates, "share_off_grid_abs_E_gt_2": float(np.mean([it["off_grid"] for it in items])),
                                      "quotas": quotas.tolist(), "distinct_quotas": int(len(np.unique(quotas))), "quota_collisions": [[float(grid[i]), float(grid[i + 1])] for i in range(len(grid) - 1) if quotas[i] == quotas[i + 1]],
                                      **derived, "per_player": per_player, "bootstrap_keys": sorted(derived["bootstrap"]) if derived["bootstrap"] else None}
    record = {"schema_version": 3, "protocol": "logged_gold_v3", "regime": regime, "split": split, "replicates": replicates, "seed": seed, "n_bootstrap": n_bootstrap, "grid": grid.tolist(),
              "population": {"candidates": len(population), "usable": len(usable), "without_logged_opening": len(population) - len(usable)}, "levels": levels_out, "seconds": time.perf_counter() - started,
              "privileged": True, "config": read_json(CONFIG_PATH)}
    return record, surfaces


def compare_with_cala_gold_v2(record: dict[str, object], gold_v2_path: Path) -> dict[str, object]:
    """Document differences to the earlier CaLA gold v2 on the shared players (same regime, test split)."""
    v2 = read_json(gold_v2_path)
    out = {"source_protocol": v2.get("protocol"), "source_version": v2.get("protocol_version"), "source_replicates": v2.get("replicates"), "source_specials_propagated": v2.get("specials_propagated"), "levels": {}}
    for level, block in record["levels"].items():
        src = v2["levels"].get(level, {})
        m_do = src.get("m_do") or src.get("causal")
        entry = {"v2_argmin_e": src.get("argmin_e"), "v3_optimum": block["summary"]["interventional_optimum"], "v2_observational_argmin_e": src.get("observational_argmin_e"), "v3_observational": block["summary"]["observational_recommendation"],
                 "v2_bootstrap_causal_optimum": (src.get("bootstrap") or {}).get("causal_optimum"), "v2_n_players": len(src.get("player_ids", [])) if isinstance(src.get("player_ids"), list) else src.get("n_players")}
        if m_do is not None and len(m_do) == len(block["causal"]):
            entry["max_abs_curve_difference"] = float(np.max(np.abs(np.asarray(m_do, dtype=float) - np.asarray(block["causal"]))))
        if isinstance(src.get("player_ids"), list):
            shared = sorted(set(src["player_ids"]) & set(block["per_player"]["player_ids"]))
            entry["shared_players"] = len(shared)
        out["levels"][level] = entry
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True, help="acknowledges that the privileged oracle table is read")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cala-gold-v2", nargs="*", default=[], help="paths of the historical gold v2 json files (natural / randomized) for the documented comparison")
    parser.add_argument("--replicates", type=int, default=None, help="override the stage-06 budget (debug only; recorded)")
    args = parser.parse_args(argv)
    assert_package_import()
    config = read_json(CONFIG_PATH)
    if not config.get("frozen_at"):
        raise RuntimeError("configs/logged_gold.json must be frozen first")
    workers = args.workers or int(config["workers"])
    decision = read_json(ROOT / "runs" / "engine_reference" / "mc_calibration" / "decision.json")
    R = args.replicates or int(decision["chosen_R"])
    criteria = engine_config()["mc_calibration"]["criteria"]
    directory = stage_dir(STAGE)
    log = Logger(STAGE, "logged_gold")
    # re-check R vs 2R on the validation split (natural regime) before touching the test split
    recheck_path = directory / "validation_recheck.json"
    if not recheck_path.exists():
        record, surfaces = compute_gold("natural", "validation", replicates=2 * R, seed=int(config["seed"]), workers=workers, n_bootstrap=0, log=log, out_dir=directory)
        context = accepted_context("natural")
        rows = {}
        for level_name, (risk_set, surface) in surfaces.items():
            rows[level_name] = evaluate_candidate(risk_set, surface, R=R, churn_level=context["churn"].for_level(level_name), mastery=context["mastery"], benchmark=context["benchmark"], criteria=criteria)
            log(f"validation recheck {level_name}: R={R} vs {2 * R} passed={rows[level_name]['passed']} mc_se={rows[level_name]['max_mc_se']:.4f} dcurve={rows[level_name]['curve_change']:.4f} dregret={rows[level_name]['regret_change']:.4f}")
        write_json(recheck_path, {"schema_version": 1, "split": "validation", "regime": "natural", "R": R, "comparator": 2 * R, "criteria": criteria, "levels": rows, "passed": all(r["passed"] for r in rows.values()),
                                  "population": record["population"], "note": "the stage-06 budget re-checked on the logged-level validation population; the test run uses R regardless and reports this outcome"})
    outputs = [recheck_path]
    gold_v2 = {Path(p).name: Path(p) for p in args.cala_gold_v2}
    for regime in ("natural", "randomized"):
        path = directory / f"gold_v3-{regime}-test.json"
        if path.exists():
            log(f"{relpath(path)} exists; skipping")
        else:
            record, _ = compute_gold(regime, "test", replicates=R, seed=int(config["seed"]), workers=workers, n_bootstrap=int(config["n_bootstrap"]), log=log, out_dir=directory)
            candidate = gold_v2.get(f"gold-{regime}.json")
            if candidate is not None and candidate.exists():
                record["comparison_with_cala_gold_v2"] = compare_with_cala_gold_v2(record, candidate)
            write_json(path, record)
            for level, block in record["levels"].items():
                log(f"{regime}/test/{level}: e*={block['summary']['interventional_optimum']:+.2f} obs={block['summary']['observational_recommendation']:+.2f} regret={block['summary']['added_churn_regret']:.4f} "
                    f"off-grid share {block['share_off_grid_abs_E_gt_2']:.2f}, distinct quotas {block['distinct_quotas']}/17")
        outputs.append(path)
    write_done(STAGE, inputs={"logged_gold.json": CONFIG_PATH, "engine_reference.json": ENGINE_CONFIG_PATH, "decision.json": ROOT / "runs" / "engine_reference" / "mc_calibration" / "decision.json"}, outputs=outputs)


if __name__ == "__main__":
    main()
