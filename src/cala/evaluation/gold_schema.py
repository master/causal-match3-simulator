"""Validation and test engine reference adapter for logged-level surfaces with R replicates.

The seed scheme is SeedSequence([0, pid, 20]) -> [base, r]. The adapter writes the gold-v2 JSON schema. Hazards use
``mastery_mismatch_hazard(update_mastery(mastery_before, outcomes), churn.for_level, completion_margin=margins)`` averaged over replicates. The surfaces
carry the oracle ``mastery_before``. The adapter asserts that it matches the logged replay (max |diff| <= 1e-9) before writing.
``check_prefix_identity`` verifies that the first R' replicates of a new surface set are bit-identical to an existing R' surface set.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from match3_simulator.causal_queries import select_grid_optimum
from match3_simulator.retention import mastery_mismatch_hazard, update_mastery
from match3_simulator.scm import LEVELS
from match3_simulator.experiments.data_base import LANDMARK
from match3_simulator.experiments.training_utils import accepted

LEVEL_NAMES = tuple(l.name for l in LEVELS)
PREFIX_KEYS = ("outcomes", "completion_margins", "goal_totals", "profiles", "rollout_seeds", "player_ids", "quotas", "mastery_before", "exogenous_seeds", "tier_indices", "logged_E", "grid")


def surface_file(surface_dir: str | Path, regime: str, split: str, level: str) -> Path:
    return Path(surface_dir) / f"{regime}-{split}-{level}-surface.npz"


def hazards_from_surface(surface: dict[str, np.ndarray], *, R: int, churn, mastery, level: str) -> dict[str, np.ndarray]:
    """Per-player hazards (G, P), full (G, P, R), m_do (G,), win / margin summaries for the first R replicates — ``hazards_for``'s arithmetic."""
    outcomes = np.asarray(surface["outcomes"])[:, :, :R]
    margins = np.asarray(surface["completion_margins"])[:, :, :R]
    if outcomes.shape[2] != R:
        raise ValueError(f"surface has {outcomes.shape[2]} replicates < requested {R}")
    mastery_before = np.asarray(surface["mastery_before"], np.float64)
    full = mastery_mismatch_hazard(update_mastery(mastery_before[None, :, None], outcomes, mastery), churn.for_level(level), completion_margin=margins)
    hazards = full.mean(axis=2)
    return {"hazards_GP": hazards, "hazards_GPR": full, "m_do": hazards.mean(axis=1), "win_GP": outcomes.mean(axis=2), "margin_GP": margins.mean(axis=2), "win": outcomes.mean(axis=2).mean(axis=1),
            "mean_margin": margins.mean(axis=2).mean(axis=1), "player_margin_sd": margins.std(axis=2), "player_hazard_sd": full.std(axis=2), "outcomes": outcomes, "margins": margins}


def surface_to_gold_v2(surface_dir: str | Path, regime: str, split: str, R: int, out: str | Path, *, logged=None, seed: int = 0, mastery_tolerance: float = 1e-9, progress=None) -> dict[str, object]:
    """Write ``out`` (gold-v2 schema; top level grid / key / replicates / protocol_version, per level player_ids, player_hazards (G, P), m_do, win,
    mean_margin, player_win, player_margin, quotas, tier, argmin_e, player_hazard_sd, surface_sha256). Outputs: the gold dict. Refuses to overwrite."""
    emit = progress or (lambda _: None)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    churn, mastery, _, _ = accepted(regime)
    levels_out: dict[str, object] = {}
    all_pids: list[int] = []
    shas: dict[str, str] = {}
    grid_ref = None
    started = time.perf_counter()
    for level in LEVEL_NAMES:
        path = surface_file(surface_dir, regime, split, level)
        if not path.exists():
            levels_out[level] = None
            continue
        shas[level] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as z:
            s = {k: np.asarray(z[k]) for k in z.files}
        grid = np.asarray(s["grid"], np.float64)
        if grid_ref is None:
            grid_ref = grid
        elif not np.array_equal(grid, grid_ref):
            raise ValueError("grid differs between level surfaces")
        pids = np.asarray(s["player_ids"], np.int64)
        if logged is not None:  # the surfaces carry the oracle mastery_before; it must equal the logged replay entering attempt 20
            idx = np.asarray([int(np.flatnonzero(logged.player_ids == p)[0]) for p in pids])
            replay = logged.mastery[idx, LANDMARK - 1]
            diff = float(np.max(np.abs(replay - np.asarray(s["mastery_before"], np.float64))))
            if diff > mastery_tolerance:
                raise RuntimeError(f"{path.name}: oracle mastery_before differs from the logged replay (max |diff| = {diff:.3e} > {mastery_tolerance})")
            lvl = logged.arrays["attempt_level"][idx, LANDMARK - 1]
            if not np.all(lvl == LEVEL_NAMES.index(level)):
                raise RuntimeError(f"{path.name}: player level mismatch with the cache")
        h = hazards_from_surface(s, R=R, churn=churn, mastery=mastery, level=level)
        m_do = h["m_do"]
        levels_out[level] = {"n_players": int(len(pids)), "player_ids": [int(x) for x in pids], "tier": [int(x) for x in s["tier_indices"]], "quotas": [int(q) for q in s["quotas"]], "m_do": m_do.tolist(),
                             "argmin_e": float(select_grid_optimum(grid, m_do)), "win": h["win"].tolist(), "mean_margin": h["mean_margin"].tolist(), "player_hazards": h["hazards_GP"].tolist(),
                             "player_win": h["win_GP"].tolist(), "player_margin": h["margin_GP"].tolist(), "player_margin_sd": h["player_margin_sd"].tolist(), "player_hazard_sd": h["player_hazard_sd"].tolist(),
                             "mc_se_max": float((np.sqrt(h["hazards_GPR"].var(axis=2, ddof=1).sum(axis=1) / R) / len(pids)).max()), "logged_E": np.asarray(s["logged_E"], np.float64).tolist(),
                             "surface": str(path), "surface_sha256": shas[level], "surface_replicates_available": int(np.asarray(s["outcomes"]).shape[2]), "mastery_before_source": "oracle table (asserted equal to the logged replay)" if logged is not None else "oracle table"}
        all_pids += [int(x) for x in pids]
        emit(f"{regime}/{split}/{level}: {len(pids)} players x R={R}: argmin {levels_out[level]['argmin_e']:+.2f}, m_do range {m_do.min():.3f}-{m_do.max():.3f}, MC-SE max {levels_out[level]['mc_se_max']:.4f}")
    if grid_ref is None:
        raise FileNotFoundError(f"no surfaces for {regime}/{split} under {surface_dir}")
    key = hashlib.sha256(json.dumps({"regime": regime, "split": split, "grid": grid_ref.tolist(), "R": int(R), "seed": int(seed), "player_ids": all_pids, "surfaces": shas}, sort_keys=True).encode()).hexdigest()
    gold = {"protocol": "logged-level engine reference (original engine; threshold protocol) adapted to the gold-v2 schema", "protocol_version": 3, "adapter": "match3_simulator.evaluation.gold_schema.surface_to_gold_v2",
            "specials_propagated": True, "regime": regime, "split": split, "landmark_attempt": LANDMARK, "grid": grid_ref.tolist(), "replicates": int(R), "seed": int(seed), "seed_scheme": "SeedSequence([seed, player_id, 20]) -> base; SeedSequence([base, r])",
            "n_players": len(all_pids), "player_ids": all_pids, "key": key, "surface_dir": str(surface_dir), "hazard_arithmetic": "mastery_mismatch_hazard(update_mastery(mastery_before, outcomes), churn.for_level, completion_margin=margins).mean(replicates)",
            "levels": levels_out, "seconds": time.perf_counter() - started, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + f".partial-{os.getpid()}")
    partial.write_text(json.dumps(gold, indent=1, allow_nan=False) + "\n")
    os.replace(partial, out)
    emit(f"gold -> {out} (key {key[:16]}, {len(all_pids)} players)")
    return gold


def check_prefix_identity(new_dir: str | Path, old_dir: str | Path, regime: str, split: str, *, keys: tuple[str, ...] = PREFIX_KEYS) -> dict[str, object]:
    """The first R_old replicates of every new surface must be bit-identical to the old surfaces. Outputs: report dict with ``passed``."""
    report: dict[str, object] = {"new_dir": str(new_dir), "old_dir": str(old_dir), "regime": regime, "split": split, "levels": {}, "passed": True}
    for level in LEVEL_NAMES:
        new_p, old_p = surface_file(new_dir, regime, split, level), surface_file(old_dir, regime, split, level)
        if not old_p.exists() or not new_p.exists():
            report["levels"][level] = {"status": "missing", "new": new_p.exists(), "old": old_p.exists()}
            report["passed"] = False
            continue
        with np.load(new_p, allow_pickle=False) as zn, np.load(old_p, allow_pickle=False) as zo:
            R_old = int(zo["outcomes"].shape[2])
            res = {"R_old": R_old, "R_new": int(zn["outcomes"].shape[2]), "keys": {}}
            for k in keys:
                if k not in zo.files or k not in zn.files:
                    res["keys"][k] = "absent"
                    continue
                a, b = np.asarray(zo[k]), np.asarray(zn[k])
                if k in ("outcomes", "completion_margins"):
                    b = b[:, :, :R_old]
                elif k in ("goal_totals", "rollout_seeds", "profiles"):
                    b = b[:, :R_old]
                ok = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
                res["keys"][k] = bool(ok)
                if not ok:
                    report["passed"] = False
        report["levels"][level] = res
    return report


__all__ = ["LEVEL_NAMES", "PREFIX_KEYS", "check_prefix_identity", "hazards_from_surface", "surface_file", "surface_to_gold_v2"]
