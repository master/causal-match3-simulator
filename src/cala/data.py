"""
This is the main CaLA data module. It uses the special-tile grids and transition information that the
release already carries per logged step (``specials_before``, ``goals_left_next``, ``specials_after``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.learned_model.tokens import N_CELLS
from match3_simulator.retention import MasteryConfig
from match3_simulator.world_modeling.wm1_data import Splits, _open_payload, load_splits, shard_dirs
from match3_simulator.experiments.data_base import LANDMARK, MAX_ATTEMPTS, MAX_STEPS, REGIMES, SPLIT_CODES, Logged, build_arrays, cache_paths

CACHE_SCHEMA4 = 2
EXTRA_STEP_KEYS = ("specials", "goal_delta", "specials_after_count")


def build_arrays4(root: str | Path, regime: str, *, splits: Splits, device: str = "cpu", progress=None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Schema-1 arrays (unchanged code path) + per-step specials grid, goal delta and post-move special count. Outputs: (public, locked, header)."""
    emit = progress or (lambda _: None)
    public, locked, header = build_arrays(root, regime, splits=splits, device=device, progress=progress)
    started = time.perf_counter()
    player_ids = public["player_ids"]
    pos = {int(p): i for i, p in enumerate(player_ids)}
    n = len(player_ids)
    shape = (n, MAX_ATTEMPTS, MAX_STEPS)
    specials = np.zeros((*shape, N_CELLS), np.int8)
    goal_delta = np.zeros(shape, np.int16)
    after_count = np.zeros(shape, np.int8)
    total = 0
    for shard in shard_dirs(Path(root), regime):
        manifest = json.loads((shard / "manifest.json").read_text())
        with _open_payload(shard / manifest["logged_artifacts"]["transitions"]["path"]) as npz:
            pi = np.asarray([pos[int(p)] for p in npz["player_id"]])
            aj = np.asarray(npz["attempt_id"]).astype(np.int64) - 1
            st = np.asarray(npz["step_id"]).astype(np.int64)
            sb = np.asarray(npz["specials_before"]).reshape(len(pi), N_CELLS)
            sa = np.asarray(npz["specials_after"]).reshape(len(pi), N_CELLS)
            specials[pi, aj, st] = sb
            goal_delta[pi, aj, st] = np.asarray(npz["goals_left"]).astype(np.int32) - np.asarray(npz["goals_left_next"]).astype(np.int32)
            after_count[pi, aj, st] = (sa != 0).sum(axis=1).astype(np.int8)
            total += len(pi)
    if total != header["transitions"]:
        raise RuntimeError("schema-2 pass read a different number of transitions than the schema-1 pass")
    test_active = (public["split"] == SPLIT_CODES["test"]) & public["attempt_present"][:, LANDMARK - 1]
    extras = {"specials": specials, "goal_delta": goal_delta, "specials_after_count": after_count}
    for key, arr in extras.items():
        locked[f"step_{key}"] = arr[test_active, LANDMARK - 1].copy()
        arr[test_active, LANDMARK - 1, 1:] = 0  # the opening (step 0) stays readable; openings carry no specials by construction
        public[f"step_{key}"] = arr
    step_mask = public["step_step_mask"]  # already excludes the locked landmark steps of test players
    if np.any(goal_delta[~step_mask] != 0) or np.any(specials[~step_mask] != 0) or np.any(after_count[~step_mask] != 0):
        raise RuntimeError("schema-2 arrays populated outside the (locked) step mask")
    header = {**header, "cache_schema": CACHE_SCHEMA4, "schema1_header": {k: v for k, v in header.items()}, "extra_step_keys": list(EXTRA_STEP_KEYS), "schema2_seconds": time.perf_counter() - started,
              "opening_specials_all_zero": bool((specials[:, :, 0].sum()) == 0)}
    emit(f"{regime}: schema-2 extras built ({total} transitions, {time.perf_counter() - started:.0f}s); opening specials all zero = {header['opening_specials_all_zero']}")
    return public, locked, header


def build_cache4(root: str | Path, regime: str, out_dir: str | Path, *, device: str = "cpu", progress=None) -> Path:
    public_path, locked_path = cache_paths(out_dir, regime)
    if public_path.exists():
        return public_path
    splits = load_splits(root)
    public, locked, header = build_arrays4(root, regime, splits=splits, device=device, progress=progress)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((locked_path, locked), (public_path, public)):
        partial = path.with_name(path.name + f".partial-{os.getpid()}")
        np.savez_compressed(partial, header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8), **payload)
        os.replace(str(partial) + ".npz", path)
    return public_path


def verify_superset(new_cache: str | Path, old_cache: str | Path) -> dict[str, object]:
    """Every array of the frozen schema-1 cache must be byte-identical in the schema-2 cache. Outputs: report dict (raises on mismatch)."""
    with np.load(old_cache, allow_pickle=False) as old, np.load(new_cache, allow_pickle=False) as new:
        old_keys = [k for k in old.files if k != "header"]
        mismatched = [k for k in old_keys if k not in new.files or not np.array_equal(np.asarray(old[k]), np.asarray(new[k]))]
        extra = sorted(set(new.files) - set(old.files))
    if mismatched:
        raise RuntimeError(f"schema-2 cache differs from the frozen cache on {mismatched}")
    return {"old": str(old_cache), "new": str(new_cache), "shared_keys_identical": len(old_keys), "extra_keys": extra}


class Logged4(Logged):
    """``Logged`` + specials / transition information in the strict-prefix batches and the target transitions. Requires a schema-2 cache."""

    def __init__(self, cache_dir: str | Path, regime: str, *, splits: Splits, mastery_config: MasteryConfig):
        super().__init__(cache_dir, regime, splits=splits, mastery_config=mastery_config)
        if self.header.get("cache_schema") != CACHE_SCHEMA4:
            raise RuntimeError(f"{cache_dir}/{regime}: Logged4 needs a schema-{CACHE_SCHEMA4} cache")

    def prefix_batch(self, idx: np.ndarray, target_attempt: np.ndarray | int, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
        """Schema-1 keys (identical to ``Logged.prefix_batch``) + specials (B, 19, T, 64), goal_delta / specials_after (B, 19, T) float, goal_colours (B, 19) long; all masked to the strict prefix."""
        out = super().prefix_batch(idx, target_attempt, device)
        idx = np.asarray(idx, dtype=np.int64)
        n_ep = MAX_ATTEMPTS - 1
        ep_mask = out["episode_mask"].cpu().numpy()
        sm = out["step_mask"].cpu().numpy()
        a = self.arrays
        specials = a["step_specials"][idx, :n_ep].astype(np.int64) * sm[:, :, :, None]
        out["specials"] = torch.as_tensor(specials, device=device)
        out["goal_delta"] = torch.as_tensor(a["step_goal_delta"][idx, :n_ep].astype(np.float32) * sm, device=device)
        out["specials_after"] = torch.as_tensor(a["step_specials_after_count"][idx, :n_ep].astype(np.float32) * sm, device=device)
        out["goal_colours"] = torch.as_tensor(np.where(ep_mask, a["attempt_goal_colour"][idx, :n_ep].astype(np.int64), 0), device=device)
        return out

    def transition_rows(self, idx: np.ndarray, target_attempt: np.ndarray | int) -> dict[str, np.ndarray]:
        out = super().transition_rows(idx, target_attempt)
        idx = np.asarray(idx, dtype=np.int64)
        target = np.broadcast_to(np.asarray(target_attempt, dtype=np.int64), idx.shape)
        j = target - 1
        sm = self.arrays["step_step_mask"][idx, j]
        r, t = np.nonzero(sm)
        out["specials"] = self.arrays["step_specials"][idx[r], j[r], t].astype(np.int64)
        return out


def stripe_state_mask(specials: np.ndarray | torch.Tensor) -> np.ndarray:
    """True for states with at least one special tile on the board. Inputs: (M, 64). Outputs: (M,) bool."""
    s = specials.cpu().numpy() if isinstance(specials, torch.Tensor) else np.asarray(specials)
    return (s != 0).any(axis=1)


__all__ = ["CACHE_SCHEMA4", "EXTRA_STEP_KEYS", "Logged4", "build_arrays4", "build_cache4", "stripe_state_mask", "verify_superset"]
