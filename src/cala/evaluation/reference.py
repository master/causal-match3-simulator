"""Two separate engine populations: cloned benchmark and logged HF players."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from match3_simulator.experiments.paper import read, write, sha, PACKAGE
from match3_simulator.evaluation.reference_common import (
    accepted_context,
    run_original_benchmark,
    load_surface_and_risk_set,
)
from match3_simulator.evaluation.mc_calibration import evaluate_candidate


def equivalence(out, workers=4):
    """Compare the wrapper with the original entry point before budget selection."""
    from match3_simulator.engine_benchmark import evaluate_engine_benchmark

    cfg = read(PACKAGE / "configs/engine_reference.json")["equivalence"]
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    c = accepted_context("natural")
    args = dict(
        n_players=cfg["n_players"],
        seed=cfg["seed"],
        e_grid=tuple(c["benchmark"].e_grid),
        rollouts_per_player=cfg["rollouts_per_player"],
        workers=workers,
        n_bootstrap=cfg["n_bootstrap"],
        reuse_goal_totals=True,
        assignment=c["assignment"],
        churn_config=c["churn"],
        mastery_config=c["mastery"],
        benchmark=c["benchmark"],
    )
    evaluate_engine_benchmark(
        c["model"], output_dir=out / "direct", status="validation", **args
    )
    run_original_benchmark(
        output_dir=out / "wrapper",
        seed=cfg["seed"],
        n_players=cfg["n_players"],
        rollouts_per_player=cfg["rollouts_per_player"],
        n_bootstrap=cfg["n_bootstrap"],
        workers=workers,
    )
    checks = {}
    for level in ("orchard", "harbour", "foundry"):
        _, a = load_surface_and_risk_set(out / "direct", level)
        _, b = load_surface_and_risk_set(out / "wrapper", level)
        for key in (
            "grid",
            "player_ids",
            "rollout_seeds",
            "outcomes",
            "goal_totals",
            "completion_margins",
        ):
            checks[level + "/" + key] = bool(
                np.array_equal(getattr(a, key), getattr(b, key))
            )
    if not all(checks.values()):
        raise RuntimeError("Reference wrapper equivalence failed")
    write(
        out / "equivalence.json",
        {
            "status": "equivalent",
            "checks": checks,
            "protocol_sha256": sha(PACKAGE / "configs/engine_reference.json"),
        },
    )


def benchmark(stage, *, out="runs/reference/benchmark", workers=10):
    cfg = read(PACKAGE / "configs/engine_reference.json")
    out = Path(out)
    if stage == "equivalence":
        return equivalence(out / "equivalence", workers)
    eq = read(out / "equivalence/equivalence.json")
    if eq["status"] != "equivalent" or eq["protocol_sha256"] != sha(
        PACKAGE / "configs/engine_reference.json"
    ):
        raise ValueError("Missing/current equivalence gate")
    if stage == "audit":
        from match3_simulator.engine_benchmark import evaluate_engine_benchmark

        audit = cfg["intervention_audit"]
        context = accepted_context("natural")
        target = out / "intervention_audit"
        if target.exists():
            raise FileExistsError(target)
        common = dict(
            n_players=audit["n_players"],
            seed=audit["seed"],
            max_engine_players=audit["max_engine_players"],
            rollouts_per_player=audit["rollouts_per_player"],
            workers=workers,
            n_bootstrap=0,
            assignment=context["assignment"],
            churn_config=context["churn"],
            mastery_config=context["mastery"],
            benchmark=context["benchmark"],
        )
        for reuse, name in [(True, "threshold"), (False, "direct")]:
            evaluate_engine_benchmark(
                context["model"],
                output_dir=target / name,
                reuse_goal_totals=reuse,
                **common,
            )
        comparison = {}
        for level in ("orchard", "harbour", "foundry"):
            _, a = load_surface_and_risk_set(target / "threshold", level)
            _, b = load_surface_and_risk_set(target / "direct", level)
            if not np.array_equal(a.player_ids, b.player_ids) or not np.array_equal(
                a.rollout_seeds, b.rollout_seeds
            ):
                raise ValueError("Intervention audit rows/seeds differ")
            comparison[level] = {
                "outcome_mismatch_fraction": float((a.outcomes != b.outcomes).mean()),
                "max_win_curve_difference": float(
                    abs(a.outcomes.mean((1, 2)) - b.outcomes.mean((1, 2))).max()
                ),
            }
        write(
            target / "comparison.json",
            {
                "levels": comparison,
                "primary": "threshold",
                "scope": "protocol sensitivity; not a replacement reference",
            },
        )
    elif stage == "calibrate":
        mc = cfg["mc_calibration"]
        run = out / "calibration"
        c = accepted_context("natural")
        if run.exists():
            raise FileExistsError(run)
        report = run_original_benchmark(
            output_dir=run,
            seed=mc["cohort_seed"],
            n_players=mc["n_players"],
            rollouts_per_player=mc["r_max"],
            n_bootstrap=0,
            workers=workers,
        )
        per = {}
        for level in report["levels"]:
            risk, surface = load_surface_and_risk_set(run, level)
            per[level] = [
                evaluate_candidate(
                    risk,
                    surface,
                    R=R,
                    churn_level=c["churn"].for_level(level),
                    mastery=c["mastery"],
                    benchmark=c["benchmark"],
                    criteria=mc["criteria"],
                )
                for R in mc["r_candidates"]
                if 2 * R <= mc["r_max"]
            ]
        eligible = [
            R
            for R in mc["r_candidates"]
            if R >= mc["floor"]
            and 2 * R <= mc["r_max"]
            and all(
                next(x["passed"] for x in rows if x["R"] == R) for rows in per.values()
            )
        ]
        write(
            out / "decision.json",
            {
                "R": min(eligible) if eligible else mc["r_max"],
                "fallback_used": not bool(eligible),
                "levels": per,
                "protocol_sha256": sha(PACKAGE / "configs/engine_reference.json"),
            },
        )
    else:
        decision = read(out / "decision.json")
        if decision["protocol_sha256"] != sha(
            PACKAGE / "configs/engine_reference.json"
        ):
            raise ValueError("Budget decision belongs to another protocol")
        for seed in cfg["validation"]["seeds"]:
            target = out / f"seed-{seed}"
            if target.exists():
                raise FileExistsError(target)
            report = run_original_benchmark(
                output_dir=target,
                seed=seed,
                n_players=cfg["validation"]["n_players"],
                rollouts_per_player=decision["R"],
                n_bootstrap=cfg["validation"]["n_bootstrap"],
                workers=workers,
            )
            write(target / "report.json", report)


def logged(*, root="data/release", regime, split, workers=10):
    from match3_simulator.evaluation import reference_logged_level as ref
    from match3_simulator.evaluation.gold_schema import surface_to_gold_v2
    from match3_simulator.experiments.data import Logged4
    from match3_simulator.experiments.training_utils import accepted
    from match3_simulator.world_modeling.wm1_data import load_splits

    out = Path("runs/reference/logged") / regime / split
    if out.exists():
        raise FileExistsError(out)
    if split == "test":
        # Validation precision is recorded first; failure remains visible instead
        # of silently being relabelled a passing R128 convergence result.
        read(out.parent / "validation/precision.json")
    out.mkdir(parents=True)
    ref.DATA = Path(root)
    report, _ = ref.compute_gold(
        regime,
        split,
        replicates=128,
        seed=0,
        workers=workers,
        n_bootstrap=500,
        log=print,
        out_dir=out,
    )
    write(out / "reference.json", report)
    _, mastery, _, _ = accepted(regime)
    cache = Logged4(
        "runs/data/schema2", regime, splits=load_splits(root), mastery_config=mastery
    )
    surface_to_gold_v2(
        out, regime, split, 128, out / "gold.json", logged=cache, seed=0, progress=print
    )
    precision(out, regime=regime, split=split)


def precision(surface_dir, *, regime, split, recommendations=None):
    """Nested R16/32/64/128 precision, including saved decisions if supplied."""
    from match3_simulator.evaluation.gold_schema import hazards_from_surface

    c = accepted_context(regime)
    rows = {}
    surface_dir = Path(surface_dir)
    for level in ("orchard", "harbour", "foundry"):
        with np.load(surface_dir / f"{regime}-{split}-{level}-surface.npz") as z:
            s = {k: z[k] for k in z.files}
        full = hazards_from_surface(
            s, R=128, churn=c["churn"], mastery=c["mastery"], level=level
        )["hazards_GPR"]
        curve = full.mean(2).mean(1)
        grid = s["grid"]
        items = []
        for R in (16, 32, 64, 128):
            h = full[:, :, :R]
            small = h.mean(2).mean(1)
            se = np.sqrt(h.var(2, ddof=1).sum(1) / R) / h.shape[1]
            recs = list((recommendations or {}).get(level, [])) + [
                float(grid[np.argmin(small)])
            ]
            record = {
                "R": R,
                "max_mc_se": float(se.max()),
                "mc_se_pass": bool(se.max() <= 0.005),
                "has_doubling": 2 * R <= 128,
            }
            if 2 * R <= 128:
                big = full[:, :, : 2 * R].mean(2).mean(1)
                indices = [int(np.flatnonzero(np.isclose(grid, e))[0]) for e in recs]
                change = max(
                    abs((small[i] - small.min()) - (big[i] - big.min()))
                    for i in indices
                )
                same = int(np.argmin(small)) == int(np.argmin(big))
                costs = []
                for i in {int(np.argmin(small)), int(np.argmin(big))}:
                    costs.extend(
                        big[j] - big[i] for j in (i - 1, i + 1) if 0 <= j < len(grid)
                    )
                record.update(
                    curve_change=float(abs(small - big).max()),
                    regret_change=float(change),
                    recommendation_pass=bool(same or all(v <= 0.005 for v in costs)),
                )
                record["passed"] = (
                    record["mc_se_pass"]
                    and record["curve_change"] <= 0.005
                    and change <= 0.005
                    and record["recommendation_pass"]
                )
            items.append(record)
        rows[level] = items
    write(
        surface_dir / "precision.json",
        {
            "headline_R": 128,
            "rule": "highest fixed computed budget, not selected by arm ranking",
            "levels": rows,
            "note": "R128 has no R256 comparator; passing a lower-budget doubling is not asserted automatically.",
        },
    )
