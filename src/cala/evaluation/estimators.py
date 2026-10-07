"""Paper estimators on frozen test predictions; no outcomes enter recommendations."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from match3_simulator.experiments.paper import read, write, sha
from match3_simulator.experiments.data_base import LANDMARK, Logged
from match3_simulator.experiments.training_utils import accepted
from match3_simulator.experiments.propensity import Propensity
from match3_simulator.experiments.guard import SerializedPredictions, phase
from match3_simulator.experiments.estimators import (
    dr_curve,
    curve_metrics,
    individual_recommendations,
    wm_mc_curve,
)
from match3_simulator.experiments.bootstrap import whole_player_bootstrap
from match3_simulator.world_modeling.wm1_data import load_splits

LEVELS = ("orchard", "harbour", "foundry")


def aligned_gold(gold, pids, levels, grid):
    if not np.array_equal(grid, np.asarray(gold["grid"])):
        raise ValueError("Gold/prediction grids differ")
    if len(set(map(int, pids))) != len(pids):
        raise ValueError("Duplicate players")
    out = {}
    for li, name in enumerate(LEVELS):
        sel = np.flatnonzero(levels == li)
        g = gold["levels"][name]
        if set(map(int, pids[sel])) != set(map(int, g["player_ids"])):
            raise ValueError(f"Gold/prediction population differs for {name}")
        index = {int(p): i for i, p in enumerate(g["player_ids"])}
        hazards = np.asarray(g["player_hazards"], float)[
            :, [index[int(p)] for p in pids[sel]]
        ]
        if not np.isfinite(hazards).all():
            raise ValueError("Nonfinite gold")
        out[name] = (sel, hazards)
    return out


def summarize(
    mu,
    mu_logged,
    Y,
    E,
    g_logged,
    g_grid,
    pids,
    levels,
    grid,
    gold,
    *,
    seed,
    routes=("WM-MC", "WM-DR"),
    bootstrap=2000,
):
    """Use the original estimator v2 formulas and player-resampling functions."""
    ref = aligned_gold(gold, pids, levels, grid)
    report = {"levels": {}, "grid": grid.tolist(), "seed": seed}
    per_player = {
        "player_ids": pids,
        "level": levels,
        "grid": grid,
        "mu_grid": mu,
        "g_grid": g_grid,
    }
    for name, (sel, hazards) in ref.items():
        m = np.asarray(gold["levels"][name]["m_do"], float)
        if not np.allclose(m, hazards.mean(axis=1), rtol=0, atol=1e-12):
            raise ValueError("Inconsistent gold population mean")
        ind = individual_recommendations(mu[sel], g_grid[sel], hazards, grid)
        rec = ind["recommendation_index"]
        row = {"n_players": len(sel), "gold_m_do": m.tolist(), "routes": {}}
        # Estimated curves are evaluated against the same complete level population.
        for route in routes:
            d = None
            if route == "WM-MC":
                terms = mu[sel]
            else:
                d = dr_curve(
                    mu[sel], mu_logged[sel], Y[sel], E[sel], g_logged[sel], grid
                )
                terms = d["terms"]
            curve = terms.mean(axis=0)
            row["routes"][route] = {
                "curve": curve.tolist(),
                **curve_metrics(curve, m, grid),
                "bootstrap": whole_player_bootstrap(
                    terms, m, grid, n_bootstrap=bootstrap, seed=seed
                ),
                "mean_individual_regret": ind["mean_individual_regret"],
                "individual_support_fraction": ind["support_fraction"],
                "population_density_support": float(np.mean(g_grid[sel] > 0.02)),
                "individual_policy": "support-masked plug-in surface; no target outcome",
            }
            if d is not None:
                row["routes"][route]["residual_diagnostics"] = {
                    "mean_residual": float(d["residual"].mean()),
                    "support": d["support"],
                }
            per_player[f"{name}__{route}__terms"] = terms
        per_player[f"{name}__gold"] = hazards
        per_player[f"{name}__rows"] = sel
        per_player[f"{name}__recommendation"] = rec
        per_player[f"{name}__regret"] = ind["individual_regret"]
        report["levels"][name] = row
    report["aggregate"] = {}
    for route in routes:
        v = [report["levels"][name] for name in LEVELS]
        report["aggregate"][route] = {
            "weighted_curve_mae": sum(
                x["n_players"] * x["routes"][route]["integrated_abs_error"] for x in v
            )
            / len(pids),
            "mean_population_regret": float(
                np.mean([x["routes"][route]["regret"] for x in v])
            ),
            "pooled_individual_regret": float(
                np.mean(
                    np.concatenate([per_player[f"{name}__regret"] for name in LEVELS])
                )
            ),
        }
    return report, per_player


def evaluate(
    *,
    route,
    out,
    gold_path,
    arm_run,
    regime,
    seed,
    root="data/release",
    cell=None,
    device="cpu",
):
    """Direct-DR uses original current-policy contexts; WM uses the saved cell.

    Direct-DR does not generate WM rollouts. Both routes serialize predictions
    before the test-outcome lock is opened and refuse stale output directories.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    dev = torch.device(device)
    gold = read(gold_path)
    run = Path(arm_run)
    training = read(run / "training.json")
    _, mastery, _, _ = accepted(regime)
    if training["regime"] != regime or int(training["seed"]) != seed:
        raise ValueError("Arm seed/regime mismatch")
    if route == "Direct-DR":
        from match3_simulator.evaluation.nuisances import (
            direct_dr_predictions,
            _W_for,
        )

        logged = Logged(
            "runs/data/schema1",
            regime,
            splits=load_splits(root),
            mastery_config=mastery,
        )
        idx = logged.active_at(LANDMARK, "test")
        target = logged.target_rows(idx, LANDMARK)
        pids = logged.player_ids[idx]
        levels = np.asarray(target["level"])
        E = np.asarray(target["E"], float)
        grid = np.asarray(gold["grid"], float)
        aligned_gold(gold, pids, levels, grid)
        with phase("evaluate"):
            pred = direct_dr_predictions(
                str(run),
                logged,
                idx,
                grid,
                device=dev,
                seed=seed,
                epochs=400,
                cache=out,
            )
            W = pred["W_test"]
            mu = pred["mu_grid_test"]
            mu_logged = pred["mu_logged_test"]
        prop = Propensity.from_json(read(run / "propensity.json"))
        g_logged = prop.density(E, W, levels)
        g_grid = prop.density(np.tile(grid, (len(pids), 1)), W, levels, floor=False)
        routes = ("Direct-DR",)
        source = {
            "arm_checkpoint_sha256": sha(run / "best.pt"),
            "direct_oof_brier": float(pred["oof_brier"]),
        }
    else:
        from match3_simulator.experiments.data import Logged4
        from match3_simulator.evaluation.nuisances import support_for

        meta = read(Path(cell) / "cell.json")
        spec = meta["spec"]
        if (
            spec["population"] != "test"
            or spec["regime"] != regime
            or spec["seed"] != seed
        ):
            raise ValueError("Test cell identity mismatch")
        if sha(run / "best.pt") != meta["arm_checkpoint_sha256"]:
            raise ValueError("Arm checkpoint differs from saved predictions")
        with np.load(Path(cell) / "arrays.npz") as a:
            pids = a["player_ids"]
            levels = a["level"].astype(int)
            grid = a["grid"].astype(float)
            E = a["E_logged"].astype(float)
            churn = a["churn"].astype(float)
            served = a["served"].astype(float)
        G = len(grid)
        if churn.ndim != 4 or churn.shape[1] * churn.shape[2] != 128:
            raise ValueError("WM evaluation requires 128 trajectories")
        if served.shape != (len(pids), G + 1) or not np.allclose(served[:, -1], E):
            raise ValueError("Missing or misaligned logged-difficulty candidate")
        if not np.allclose(served[:, :G], grid[None, :]):
            raise ValueError("Candidate ordering mismatch")
        mu_all = churn.mean(axis=(1, 2))
        mu, mu_logged = mu_all[:, :G], mu_all[:, G]
        logged = Logged4(
            "runs/data/schema2",
            regime,
            splits=load_splits(root),
            mastery_config=mastery,
        )
        positions = {int(p): i for i, p in enumerate(logged.player_ids)}
        idx = np.array([positions[int(p)] for p in pids])
        target = logged.target_rows(idx, LANDMARK)
        if not np.array_equal(levels, target["level"]) or not np.allclose(
            E, target["E"], atol=1e-5
        ):
            raise ValueError("Cell/cache alignment failure")
        aligned_gold(gold, pids, levels, grid)
        g_grid, W = support_for(str(run), logged, idx, grid, device=dev)
        prop = Propensity.from_json(read(run / "propensity.json"))
        g_logged = prop.density(E, W, levels)
        routes = ("WM-MC", "WM-DR")
        source = {
            "cell": str(cell),
            "cell_sha256": sha(Path(cell) / "arrays.npz"),
            "spec": spec,
            "arm_checkpoint_sha256": meta["arm_checkpoint_sha256"],
        }
    if not all(np.isfinite(x).all() for x in (mu, mu_logged, g_logged, g_grid)):
        raise ValueError("Nonfinite prediction/density")
    path = out / "predictions.npz"
    np.savez_compressed(
        path,
        player_ids=pids,
        level=levels,
        grid=grid,
        mu_grid=mu,
        mu_logged=mu_logged,
        g_logged=g_logged,
        g_grid=g_grid,
        W=W,
        E_logged=E,
    )
    token = SerializedPredictions.of(path)
    with phase("correct"):
        observed = logged.lock.unlock(token)
    pos = {int(p): i for i, p in enumerate(observed.player_ids)}
    Y = np.array([observed["churn_after"][pos[int(p)]] for p in pids], float)
    if not np.isfinite(Y).all() or not set(np.unique(Y)) <= {0.0, 1.0}:
        raise ValueError("Invalid churn outcome")
    report, arrays = summarize(
        mu,
        mu_logged,
        Y,
        E,
        g_logged,
        g_grid,
        pids,
        levels,
        grid,
        gold,
        seed=seed,
        routes=routes,
    )
    report.update(
        regime=regime,
        arm=training["arm"],
        policy=(
            "special"
            if training.get("arm_config", {}).get("policy_specials")
            else "current"
        ),
        gold_sha256=sha(gold_path),
        gold_replicates=gold["replicates"],
        predictions_sha256=token.sha256,
        source=source,
    )
    np.savez_compressed(out / "analysis_arrays.npz", **arrays)
    write(out / "report.json", report)
    return report
