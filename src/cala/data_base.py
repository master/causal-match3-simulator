"""CaLA data: the local release as padded per-player arrays, strict-prefix batches for the encoder, decision-time base
context B_i, attempt-20 openings, the accepted-rule mastery replay, handcrafted history features and the test-outcome lock.

    python -m match3_simulator.experiments.data_base build --regime natural --out runs/cala/data
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from match3_simulator.learned_model.tokens import N_CELLS
from match3_simulator.retention import MasteryConfig, update_mastery
from match3_simulator.scm import LEVELS, TIER_MOVE_BUDGETS, TIER_NAMES
from match3_simulator.world_modeling.wm1_data import OracleAccess, Splits, _open_payload, load_splits, shard_dirs
from match3_simulator.world_modeling.wm1_verify import PAYLOAD_COMMIT
from match3_simulator.experiments.guard import TestOutcomeLock

LANDMARK = 20
MAX_ATTEMPTS = 20
MAX_STEPS = max(TIER_MOVE_BUDGETS)
REGIMES: tuple[str, ...] = ("natural", "randomized")
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
LEVEL_INDEX = {level.name: i for i, level in enumerate(LEVELS)}
TIER_INDEX = {name: i for i, name in enumerate(TIER_NAMES)}
FORBIDDEN_COLUMNS = frozenset({"mastery_before", "mastery_after", "oracle_win_probability", "churn_probability", "k_search", "k_pattern", "k_planning", "k_strategy"})
EVIDENCE_COLUMNS: tuple[str, ...] = ("x_search_latency", "x_candidate_recall", "x_hint_count", "x_pattern_error_rate", "x_immediate_pattern_precision", "x_distractor_resistance",
                                     "x_lookahead_choice_rate", "x_setup_value_z", "x_cascade_preparation", "x_goal_clear_share", "x_goals_per_move", "x_moves_left_efficiency")
ATTEMPT_FLOAT_COLUMNS: tuple[str, ...] = ("E", "completion_margin", "cascade_depth_mean", "tiles_per_move", "candidate_recall_mean", "pattern_noise_scale_mean", "selected_setup_value_mean",
                                          "selected_goal_cleared_mean")
ATTEMPT_INT_COLUMNS: tuple[str, ...] = ("R", "churn_after", "moves_used", "goals_cleared", "reshuffles", "striped_tiles_created", "striped_tiles_activated", "served_goal_count", "move_budget",
                                        "active_before")
#: attempt-20 fields of test players that reveal the outcome (locked); E, level, tier, served_goal_count, move_budget, active_before stay readable (treatment / design)
LOCKED_ATTEMPT_COLUMNS: tuple[str, ...] = ("R", "completion_margin", "churn_after", "moves_used", "goals_cleared", "reshuffles", "cascade_depth_mean", "tiles_per_move",
                                           "striped_tiles_created", "striped_tiles_activated", "candidate_recall_mean", "pattern_noise_scale_mean", "selected_setup_value_mean",
                                           "selected_goal_cleared_mean", "evidence")
CACHE_SCHEMA = 1
BASE_CONTEXT_WIDTH = len(TIER_NAMES) + len(LEVELS) + 1


class DataError(RuntimeError):
    """Malformed or inconsistent release rows."""


def _float(value: str) -> float:
    return float(value) if value not in ("", "nan", "None") else 0.0


def read_attempts(shard_dir: Path) -> list[dict[str, object]]:
    """Read one shard's logged attempt table (episodes.csv), refusing oracle columns. Inputs: shard dir. Outputs: list of parsed rows."""
    path = shard_dir / "episodes.csv"
    if any(part == "oracle" for part in path.parts):
        raise OracleAccess(str(path))
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        forbidden = FORBIDDEN_COLUMNS & set(reader.fieldnames or ())
        if forbidden:
            raise OracleAccess(f"logged table carries oracle fields {sorted(forbidden)}")
        for row in reader:
            parsed: dict[str, object] = {"player_id": int(row["player_id"]), "attempt_id": int(row["attempt_id"]), "level": LEVEL_INDEX[row["level"]], "tier": TIER_INDEX[row["tier"]]}
            for name in ATTEMPT_FLOAT_COLUMNS:
                parsed[name] = _float(row[name])
            for name in ATTEMPT_INT_COLUMNS:
                parsed[name] = int(float(row[name]))
            parsed["evidence"] = [_float(row[name]) for name in EVIDENCE_COLUMNS]
            rows.append(parsed)
    return rows


def build_arrays(root: str | Path, regime: str, *, splits: Splits, device: str = "cpu", progress=None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Materialise the padded per-player arrays of one regime from the shards.

    Inputs: dataset root; regime; Splits; torch device for the batched legal-mask / immediate-clear features.
    Outputs: (public arrays, locked test attempt-20 arrays, header).
    """
    emit = progress or (lambda _: None)
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    root = Path(root)
    started = time.perf_counter()
    attempts: list[dict[str, object]] = []
    trans: dict[str, list[np.ndarray]] = {}
    for shard in shard_dirs(root, regime):
        attempts.extend(read_attempts(shard))
        manifest = json.loads((shard / "manifest.json").read_text())
        with _open_payload(shard / manifest["logged_artifacts"]["transitions"]["path"]) as npz:
            if int(npz["schema_version"][0]) != 3:
                raise DataError("CaLA needs transition schema 3")
            for key in ("board_before", "action_index", "moves_left", "goals_left", "goal_colour", "player_id", "attempt_id", "step_id"):
                trans.setdefault(key, []).append(np.asarray(npz[key]))
    t = {k: np.concatenate(v) for k, v in trans.items()}
    player_ids = np.asarray(sorted({int(r["player_id"]) for r in attempts}), dtype=np.int64)
    pos = {int(p): i for i, p in enumerate(player_ids)}
    n = len(player_ids)
    split_code = np.full(n, -1, dtype=np.int8)
    for name, code in SPLIT_CODES.items():
        split_code[np.isin(player_ids, sorted(splits.of(name)))] = code
    if np.any(split_code < 0):
        raise DataError("players outside every split")
    shape = (n, MAX_ATTEMPTS)
    a: dict[str, np.ndarray] = {"present": np.zeros(shape, bool), "level": np.zeros(shape, np.int8), "tier": np.zeros(shape, np.int8), "goal_colour": np.zeros(shape, np.int8),
                                "evidence": np.zeros((*shape, len(EVIDENCE_COLUMNS)), np.float32)}
    for name in ATTEMPT_FLOAT_COLUMNS:
        a[name] = np.zeros(shape, np.float32)
    for name in ATTEMPT_INT_COLUMNS:
        a[name] = np.zeros(shape, np.int32)
    for row in attempts:
        i, j = pos[int(row["player_id"])], int(row["attempt_id"]) - 1
        if not 0 <= j < MAX_ATTEMPTS:
            raise DataError(f"attempt {j + 1} outside 1..{MAX_ATTEMPTS}")
        a["present"][i, j] = True
        a["level"][i, j], a["tier"][i, j] = row["level"], row["tier"]
        a["evidence"][i, j] = row["evidence"]
        for name in ATTEMPT_FLOAT_COLUMNS + ATTEMPT_INT_COLUMNS:
            a[name][i, j] = row[name]
    # attempts are contiguous from 1 (churn is absorbing)
    counts = a["present"].sum(axis=1)
    if not np.array_equal(a["present"], np.arange(MAX_ATTEMPTS)[None, :] < counts[:, None]):
        raise DataError("attempts are not a contiguous prefix from 1")
    # steps
    s_shape = (n, MAX_ATTEMPTS, MAX_STEPS)
    s = {"boards": np.zeros((*s_shape, N_CELLS), np.int8), "actions": np.zeros(s_shape, np.int16), "moves_left": np.zeros(s_shape, np.int8), "goals_left": np.zeros(s_shape, np.int16),
         "step_mask": np.zeros(s_shape, bool)}
    pi = np.asarray([pos[int(p)] for p in t["player_id"]])
    aj = t["attempt_id"].astype(np.int64) - 1
    st = t["step_id"].astype(np.int64)
    if st.max() >= MAX_STEPS:
        raise DataError("more steps than the largest move budget")
    s["boards"][pi, aj, st] = t["board_before"].reshape(len(pi), N_CELLS)
    s["actions"][pi, aj, st] = t["action_index"]
    s["moves_left"][pi, aj, st] = t["moves_left"]
    s["goals_left"][pi, aj, st] = t["goals_left"]
    s["step_mask"][pi, aj, st] = True
    a["goal_colour"][pi, aj] = t["goal_colour"]
    if not np.array_equal(s["step_mask"].any(axis=2), a["present"] & (a["moves_used"] > 0)):
        raise DataError("transition rows do not match the attempt table")
    emit(f"{regime}: {n} players, {len(attempts)} attempts, {len(pi)} transitions read ({time.perf_counter() - started:.0f}s); action features…")
    legal_count, chosen_total, chosen_goal = action_features(s["boards"], s["actions"], a["goal_colour"], s["step_mask"], device=device)
    s["legal_count"], s["chosen_total_clear"], s["chosen_goal_clear"] = legal_count, chosen_total, chosen_goal
    # lock the test players' attempt-20 outcomes
    test_active = (split_code == SPLIT_CODES["test"]) & a["present"][:, LANDMARK - 1]
    locked: dict[str, np.ndarray] = {"player_id": player_ids[test_active]}
    for name in LOCKED_ATTEMPT_COLUMNS:
        locked[name] = a[name][test_active, LANDMARK - 1].copy()
        a[name][test_active, LANDMARK - 1] = 0
    for name in ("boards", "actions", "moves_left", "goals_left", "step_mask", "legal_count", "chosen_total_clear", "chosen_goal_clear"):
        locked[f"step_{name}"] = s[name][test_active, LANDMARK - 1].copy()
        s[name][test_active, LANDMARK - 1, 1:] = 0  # the opening (step 0) stays readable
    header = {"cache_schema": CACHE_SCHEMA, "regime": regime, "payload_commit": PAYLOAD_COMMIT, "splits_sha256": splits.sha256, "players": int(n), "attempts": len(attempts),
              "transitions": int(len(pi)), "test_players_active_at_landmark": int(test_active.sum()), "locked_columns": list(LOCKED_ATTEMPT_COLUMNS), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "seconds": time.perf_counter() - started}
    public = {"player_ids": player_ids, "split": split_code, **{f"attempt_{k}": v for k, v in a.items()}, **{f"step_{k}": v for k, v in s.items()}}
    return public, locked, header


@torch.no_grad()
def action_features(boards: np.ndarray, actions: np.ndarray, goal_colour: np.ndarray, step_mask: np.ndarray, *, device: str = "cpu", chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step legal-move count and the immediate total / goal clears of the chosen swap (handcrafted action-quality features).

    Inputs: boards (P, A, T, 64); actions (P, A, T); goal_colour (P, A); step_mask (P, A, T). Outputs: three (P, A, T) arrays (0 where masked).
    """
    from match3_simulator.learned_model.action_policy import ContinuousActionPolicy
    from match3_simulator.world_modeling.decoder import torch_legal_mask

    rows = np.flatnonzero(step_mask.reshape(-1))
    flat_boards = boards.reshape(-1, N_CELLS)
    flat_actions = actions.reshape(-1)
    colours = np.broadcast_to(goal_colour[:, :, None], step_mask.shape).reshape(-1)
    legal = np.zeros(step_mask.size, np.int16)
    total = np.zeros(step_mask.size, np.int8)
    goal = np.zeros(step_mask.size, np.int8)
    policy = ContinuousActionPolicy().to(device)
    dev = torch.device(device)
    for start in range(0, len(rows), chunk):
        idx = rows[start : start + chunk]
        b = torch.as_tensor(flat_boards[idx].astype(np.int64), device=dev)
        c = torch.as_tensor(colours[idx].astype(np.int64), device=dev)
        legal[idx] = torch_legal_mask(b).sum(dim=1).cpu().numpy()
        feats = policy.immediate_action_features(b, c)  # (n, 128, 2)
        chosen = feats[torch.arange(len(idx), device=dev), torch.as_tensor(flat_actions[idx].astype(np.int64), device=dev)]
        total[idx] = chosen[:, 0].cpu().numpy().astype(np.int8)
        goal[idx] = chosen[:, 1].cpu().numpy().astype(np.int8)
    return legal.reshape(step_mask.shape), total.reshape(step_mask.shape), goal.reshape(step_mask.shape)


def cache_paths(out_dir: str | Path, regime: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    return out / f"{regime}.npz", out / f"{regime}.test_outcomes.npz"


def build_cache(root: str | Path, regime: str, out_dir: str | Path, *, device: str = "cpu", progress=None) -> Path:
    """Build (once) the per-regime cache and the separate locked test-outcome file. Outputs: the public cache path."""
    public_path, locked_path = cache_paths(out_dir, regime)
    if public_path.exists():
        return public_path
    splits = load_splits(root)
    public, locked, header = build_arrays(root, regime, splits=splits, device=device, progress=progress)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((locked_path, locked), (public_path, public)):
        partial = path.with_name(path.name + f".partial-{os.getpid()}")
        np.savez_compressed(partial, header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8), **payload)
        os.replace(str(partial) + ".npz", path)
    return public_path


class Logged:
    """Padded per-player arrays of one regime (public part) plus the test-outcome lock.

    Attributes (P players, A = 20 attempts, T = 22 steps): player_ids, split, attempt_* (P, A[, 12]), step_* (P, A, T[, 64]), mastery (P, A + 1) with
    mastery[:, a - 1] the experience state entering attempt a under the accepted rule replayed over logged R.
    """

    def __init__(self, cache_dir: str | Path, regime: str, *, splits: Splits, mastery_config: MasteryConfig):
        public_path, locked_path = cache_paths(cache_dir, regime)
        with _open_payload(public_path) as npz:
            self.header = json.loads(bytes(np.asarray(npz["header"])).decode())
            if self.header["splits_sha256"] != splits.sha256 or self.header["payload_commit"] != PAYLOAD_COMMIT or self.header["regime"] != regime:
                raise DataError(f"{public_path}: cache header does not match (splits / payload / regime)")
            self.arrays = {k: np.asarray(npz[k]) for k in npz.files if k != "header"}
        self.regime = regime
        self.splits = splits
        self.mastery_config = mastery_config
        self.player_ids: np.ndarray = self.arrays["player_ids"]
        self.split: np.ndarray = self.arrays["split"]
        self.present: np.ndarray = self.arrays["attempt_present"]
        self.n_attempts = self.present.sum(axis=1)
        self.mastery = self._replay_mastery()
        self._lock: TestOutcomeLock | None = None
        self._locked_path = locked_path

    def __getattr__(self, name: str) -> np.ndarray:
        arrays = self.__dict__.get("arrays")
        if arrays is not None and name in arrays:
            return arrays[name]
        raise AttributeError(name)

    def _replay_mastery(self) -> np.ndarray:
        m = np.zeros((len(self.player_ids), MAX_ATTEMPTS + 1), dtype=np.float64)
        m[:, 0] = self.mastery_config.initial
        r = self.arrays["attempt_R"]
        test20 = (self.split == SPLIT_CODES["test"])
        for j in range(MAX_ATTEMPTS):
            updated = update_mastery(m[:, j], np.clip(r[:, j], 0, 1), self.mastery_config)
            valid = self.present[:, j] & ~((j == LANDMARK - 1) & test20)  # the locked landmark outcome of test players never enters
            m[:, j + 1] = np.where(valid, updated, m[:, j])
        return m

    # --- populations -------------------------------------------------------------------------------------------------
    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == SPLIT_CODES[split])

    def active_at(self, attempt: int, split: str | None = None) -> np.ndarray:
        """Player indices active at (i.e. logged with) ``attempt``, optionally restricted to a split."""
        mask = self.present[:, attempt - 1].copy()
        if split is not None:
            mask &= self.split == SPLIT_CODES[split]
        return np.flatnonzero(mask)

    @property
    def lock(self) -> TestOutcomeLock:
        if self._lock is None:
            with _open_payload(self._locked_path) as npz:
                columns = {k: np.asarray(npz[k]) for k in npz.files if k not in ("header", "player_id")}
                self._lock = TestOutcomeLock(columns, player_ids=np.asarray(npz["player_id"]))
        return self._lock

    # --- strict-prefix encoder batches ------------------------------------------------------------------------------------
    def prefix_batch(self, idx: np.ndarray, target_attempt: np.ndarray | int, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
        """Encoder inputs over attempts < target_attempt (never the target or later).

        Inputs: player indices (B,); target attempt per row (B,) or scalar in 1..20; device.
        Outputs: dict boards (B, 19, T, 64), actions, moves_left, goals_left, step_mask (B, 19, T), levels, tiers, served_difficulty, outcomes, margins, episode_mask (B, 19),
        proxies (B, 19, 12).
        """
        idx = np.asarray(idx, dtype=np.int64)
        target = np.broadcast_to(np.asarray(target_attempt, dtype=np.int64), idx.shape)
        if np.any((target < 1) | (target > MAX_ATTEMPTS)):
            raise ValueError("target attempt must lie in 1..20")
        n_ep = MAX_ATTEMPTS - 1
        ep_mask = (np.arange(n_ep)[None, :] < (target - 1)[:, None]) & self.present[idx, :n_ep]
        a = self.arrays
        sel = lambda key: a[key][idx, :n_ep]
        out = {"boards": sel("step_boards").astype(np.int64), "actions": sel("step_actions").astype(np.int64), "moves_left": sel("step_moves_left").astype(np.int64),
               "goals_left": sel("step_goals_left").astype(np.int64), "step_mask": sel("step_step_mask") & ep_mask[:, :, None], "levels": sel("attempt_level").astype(np.int64),
               "tiers": sel("attempt_tier").astype(np.int64), "served_difficulty": sel("attempt_E").astype(np.float32), "outcomes": sel("attempt_R").astype(np.float32),
               "margins": sel("attempt_completion_margin").astype(np.float32), "proxies": sel("attempt_evidence").astype(np.float32), "episode_mask": ep_mask}
        for key in ("levels", "tiers", "served_difficulty", "outcomes", "margins"):
            out[key] = np.where(ep_mask, out[key], 0)
        out["proxies"] = out["proxies"] * ep_mask[:, :, None]
        return {k: torch.as_tensor(v, device=device) for k, v in out.items()}

    def base_context(self, idx: np.ndarray, target_attempt: np.ndarray | int) -> np.ndarray:
        """Decision-time context B_i = tier one-hot (3) + level one-hot (3) + mastery entering the target attempt (1). Outputs: (B, 7) float32."""
        idx = np.asarray(idx, dtype=np.int64)
        target = np.broadcast_to(np.asarray(target_attempt, dtype=np.int64), idx.shape)
        tier = self.arrays["attempt_tier"][idx, target - 1]
        level = self.arrays["attempt_level"][idx, target - 1]
        out = np.zeros((len(idx), BASE_CONTEXT_WIDTH), dtype=np.float32)
        out[np.arange(len(idx)), tier] = 1.0
        out[np.arange(len(idx)), len(TIER_NAMES) + level] = 1.0
        out[:, -1] = self.mastery[idx, target - 1]
        return out

    def target_rows(self, idx: np.ndarray, target_attempt: np.ndarray | int) -> dict[str, np.ndarray]:
        """Per-row treatment / design fields of the target attempt (never test outcomes): level, tier, E, goal_colour, move_budget, served quota, opening board, mastery before."""
        idx = np.asarray(idx, dtype=np.int64)
        target = np.broadcast_to(np.asarray(target_attempt, dtype=np.int64), idx.shape)
        a = self.arrays
        j = target - 1
        return {"player_id": self.player_ids[idx], "level": a["attempt_level"][idx, j].astype(np.int64), "tier": a["attempt_tier"][idx, j].astype(np.int64), "E": a["attempt_E"][idx, j].astype(np.float64),
                "goal_colour": a["attempt_goal_colour"][idx, j].astype(np.int64), "move_budget": a["attempt_move_budget"][idx, j].astype(np.int64), "served_goal_count": a["attempt_served_goal_count"][idx, j].astype(np.int64),
                "board": a["step_boards"][idx, j, 0].astype(np.int64), "mastery_before": self.mastery[idx, j], "attempt": target.copy()}

    def outcome_rows(self, idx: np.ndarray, target_attempt: np.ndarray | int) -> dict[str, np.ndarray]:
        """Logged outcomes of the target attempt for train / validation rows (R, Q, C, moves_used, goals_cleared, evidence, action summaries). Refuses test attempt-20 rows."""
        idx = np.asarray(idx, dtype=np.int64)
        target = np.broadcast_to(np.asarray(target_attempt, dtype=np.int64), idx.shape)
        if np.any((self.split[idx] == SPLIT_CODES["test"]) & (target == LANDMARK)):
            from match3_simulator.experiments.guard import GuardViolation

            raise GuardViolation("attempt-20 outcomes of test players are locked")
        a = self.arrays
        j = target - 1
        sm = a["step_step_mask"][idx, j]
        steps = np.maximum(sm.sum(axis=1), 1)
        return {"R": a["attempt_R"][idx, j].astype(np.float32), "Q": a["attempt_completion_margin"][idx, j].astype(np.float32), "C": a["attempt_churn_after"][idx, j].astype(np.float32),
                "moves_used": a["attempt_moves_used"][idx, j].astype(np.float32), "goals_cleared": a["attempt_goals_cleared"][idx, j].astype(np.float32), "evidence": a["attempt_evidence"][idx, j].astype(np.float32),
                "mean_legal": (a["step_legal_count"][idx, j] * sm).sum(axis=1) / steps, "mean_goal_clear": (a["step_chosen_goal_clear"][idx, j] * sm).sum(axis=1) / steps}

    def transition_rows(self, idx: np.ndarray, target_attempt: np.ndarray | int) -> dict[str, np.ndarray]:
        """Flattened logged transitions of the target attempt for policy training: boards (M, 64), actions, moves_left, goals_left, goal_colour, row (M,) index into idx."""
        idx = np.asarray(idx, dtype=np.int64)
        target = np.broadcast_to(np.asarray(target_attempt, dtype=np.int64), idx.shape)
        a = self.arrays
        j = target - 1
        sm = a["step_step_mask"][idx, j]  # (B, T)
        r, t = np.nonzero(sm)
        return {"boards": a["step_boards"][idx[r], j[r], t].astype(np.int64), "actions": a["step_actions"][idx[r], j[r], t].astype(np.int64), "moves_left": a["step_moves_left"][idx[r], j[r], t].astype(np.int64),
                "goals_left": a["step_goals_left"][idx[r], j[r], t].astype(np.int64), "goal_colour": a["attempt_goal_colour"][idx[r], j[r]].astype(np.int64), "row": r}


# --- handcrafted features ------------------------------------------------------------------------------------------------

SERIES: tuple[str, ...] = ("attempt_E", "attempt_R", "attempt_completion_margin", "attempt_moves_used", "attempt_goals_cleared", "attempt_cascade_depth_mean", "attempt_striped_tiles_created",
                           "attempt_striped_tiles_activated", "attempt_candidate_recall_mean", "attempt_selected_goal_cleared_mean")
DECAY = 0.8


def handcrafted_feature_names() -> list[str]:
    names = ["n_attempts", "n_wins", *[f"n_level_{l.name}" for l in LEVELS]]
    for key in SERIES:
        names += [f"{key[8:]}_recency_mean", f"{key[8:]}_slope"]
    names += ["mean_legal_moves", "mean_chosen_total_clear", "mean_chosen_goal_clear", *[f"{c}_mean" for c in EVIDENCE_COLUMNS]]
    return names


def raw_handcrafted(logged: Logged, idx: np.ndarray, target_attempt: np.ndarray | int) -> np.ndarray:
    """Prespecified history features over attempts < target (counts, decay-0.8 recency means, OLS slopes, action-quality means, evidence means). Outputs: (B, F)."""
    idx = np.asarray(idx, dtype=np.int64)
    target = np.broadcast_to(np.asarray(target_attempt, dtype=np.int64), idx.shape)
    n_ep = MAX_ATTEMPTS - 1
    mask = ((np.arange(n_ep)[None, :] < (target - 1)[:, None]) & logged.present[idx, :n_ep]).astype(np.float64)
    count = mask.sum(axis=1)
    a = logged.arrays
    feats = [count, (a["attempt_R"][idx, :n_ep] * mask).sum(axis=1)]
    for li in range(len(LEVELS)):
        feats.append(((a["attempt_level"][idx, :n_ep] == li) & (mask > 0)).sum(axis=1).astype(np.float64))
    pos = np.arange(n_ep, dtype=np.float64)[None, :]
    last = (target - 2)[:, None]  # index of the most recent prefix attempt
    weights = np.where(mask > 0, DECAY ** np.clip(last - pos, 0, None), 0.0)
    wsum = np.maximum(weights.sum(axis=1), 1e-12)
    x_mean = (pos * mask).sum(axis=1) / np.maximum(count, 1)
    x_var = (((pos - x_mean[:, None]) ** 2) * mask).sum(axis=1)
    for key in SERIES:
        y = a[key][idx, :n_ep].astype(np.float64)
        feats.append((y * weights).sum(axis=1) / wsum)
        y_mean = (y * mask).sum(axis=1) / np.maximum(count, 1)
        cov = (((pos - x_mean[:, None]) * (y - y_mean[:, None])) * mask).sum(axis=1)
        feats.append(np.where(x_var > 0, cov / np.maximum(x_var, 1e-12), 0.0))
    sm = a["step_step_mask"][idx, :n_ep] & (mask[:, :, None] > 0)
    steps = np.maximum(sm.sum(axis=(1, 2)), 1)
    for key in ("step_legal_count", "step_chosen_total_clear", "step_chosen_goal_clear"):
        feats.append((a[key][idx, :n_ep] * sm).sum(axis=(1, 2)) / steps)
    ev = a["attempt_evidence"][idx, :n_ep].astype(np.float64)
    ev_mean = (ev * mask[:, :, None]).sum(axis=1) / np.maximum(count, 1)[:, None]
    return np.concatenate([np.stack(feats, axis=1), ev_mean], axis=1).astype(np.float32)


@dataclass(frozen=True)
class HandcraftedFeatures:
    """Standardisation fitted on training rows only; ``transform`` maps raw features to z-scores (constant features -> 0)."""

    mean: np.ndarray
    scale: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def fit(cls, raw_train: np.ndarray) -> "HandcraftedFeatures":
        mean = raw_train.mean(axis=0)
        scale = raw_train.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32), tuple(handcrafted_feature_names()))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - self.mean) / self.scale).astype(np.float32)

    @property
    def width(self) -> int:
        return int(len(self.mean))

    def to_json(self) -> dict[str, object]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "names": list(self.names)}

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> "HandcraftedFeatures":
        return cls(np.asarray(raw["mean"], np.float32), np.asarray(raw["scale"], np.float32), tuple(raw["names"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", default="data/release")
    build.add_argument("--regime", choices=REGIMES, default="natural")
    build.add_argument("--out", default="runs/cala/data")
    build.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    path = build_cache(args.root, args.regime, args.out, device=args.device, progress=print)
    print(f"{path} ready ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()


__all__ = ["BASE_CONTEXT_WIDTH", "EVIDENCE_COLUMNS", "HandcraftedFeatures", "LANDMARK", "LOCKED_ATTEMPT_COLUMNS", "Logged", "MAX_ATTEMPTS", "MAX_STEPS", "REGIMES", "SPLIT_CODES",
           "action_features", "build_arrays", "build_cache", "cache_paths", "handcrafted_feature_names", "raw_handcrafted", "read_attempts"]
