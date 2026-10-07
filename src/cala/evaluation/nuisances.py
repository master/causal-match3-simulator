"""Frozen outcome and propensity helpers used by the paper estimators."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from match3_simulator.experiments.cell_utils import arm_inputs_for, load_trained_arm
from match3_simulator.experiments.contexts import Arm
from match3_simulator.experiments.data_base import LANDMARK, Logged
from match3_simulator.experiments.direct_dr import cross_fitted_outcome_model
from match3_simulator.experiments.guard import EvaluationOnly, phase
from match3_simulator.experiments.propensity import Propensity
from match3_simulator.experiments.training_utils import accepted, arm_W
from match3_simulator.experiments.cells import arm_inputs_for4, load_trained_arm4
from match3_simulator.experiments.data import Logged4

@torch.no_grad()
def _W_for(arm_run: str, logged: Logged, idx: np.ndarray, *, device: torch.device) -> np.ndarray:
    churn, mastery, _, _ = accepted(logged.regime)
    model, training = load_trained_arm(arm_run, device=device, churn=churn, mastery=mastery)
    token = EvaluationOnly("oracle_k") if model.arm is Arm.ORACLE_K else None
    if token is not None:
        token.__enter__()
    try:
        with phase("evaluate"):
            inputs = arm_inputs_for(model, training, logged)
            return arm_W(model, logged, inputs, idx, device=device)
    finally:
        if token is not None:
            token.__exit__(None, None, None)


def direct_dr_predictions(arm_run: str, logged: Logged, test_idx: np.ndarray, grid: np.ndarray, *, device: torch.device, seed: int, epochs: int = 400, cache: Path | None = None) -> dict[str, object]:
    """Cross-fitted logged-outcome model mu_obs on train + validation landmark rows, predicted on the test players (cached per arm run and population hash)."""
    key = f"direct_dr-{len(test_idx)}-{int(test_idx.sum())}-{len(grid)}.npz"
    path = (Path(arm_run) / key) if cache is None else cache / key
    if path.exists():
        with np.load(path) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    tv = np.concatenate([logged.active_at(LANDMARK, "train"), logged.active_at(LANDMARK, "validation")])
    W = _W_for(arm_run, logged, tv, device=device)
    W_test = _W_for(arm_run, logged, test_idx, device=device)
    tgt, out = logged.target_rows(tv, LANDMARK), logged.outcome_rows(tv, LANDMARK)
    tgt_test = logged.target_rows(test_idx, LANDMARK)
    res = cross_fitted_outcome_model(W, tgt["E"], out["C"], tgt["level"], W_test=W_test, E_test=tgt_test["E"], level_test=tgt_test["level"], grid=grid, seed=seed, device=device, epochs=epochs)
    payload = {"mu_grid_test": res["mu_grid_test"], "mu_logged_test": res["mu_logged_test"], "oof_brier": np.asarray(res["oof_brier"]), "W_test": W_test}
    np.savez_compressed(path.with_name(path.name + ".partial"), **payload)
    os.replace(str(path.with_name(path.name + ".partial")) + ".npz", path)
    return payload


@torch.no_grad()
def support_for(arm_run: str, logged: Logged4, idx: np.ndarray, grid: np.ndarray, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Unfloored propensity density g(e | W_i) on the grid with the arm's frozen propensity model; W = (B_i, posterior mean / point / features). Outputs: (g_grid (P, G), W)."""
    churn, mastery, _, _ = accepted(logged.regime)
    model, training = load_trained_arm4(arm_run, device=device, churn=churn, mastery=mastery)
    from match3_simulator.experiments.contexts import Arm
    from match3_simulator.experiments.guard import EvaluationOnly
    token = EvaluationOnly("oracle_k") if model.arm is Arm.ORACLE_K else None   # reference arm, evaluation only (2026-10-01; see arm_inputs_for4)
    if token is not None:
        token.__enter__()
    try:
        with phase("evaluate"):
            inputs = arm_inputs_for4(model, training, logged, allow_oracle=token is not None)
            W = arm_W(model, logged, inputs, idx, device=device)
    finally:
        if token is not None:
            token.__exit__(None, None, None)
    prop = Propensity.from_json(json.loads((Path(arm_run) / "propensity.json").read_text()))
    level = logged.arrays["attempt_level"][idx, LANDMARK - 1].astype(np.int64)
    return prop.density(np.tile(grid, (len(idx), 1)), W, level, floor=False), W

