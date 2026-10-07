"""WM data pipeline: split guards, the per-shard derived cache (settled-pair cleared masks, residual, terminal, stripe flags) and the
padded-episode store the training loop and the evaluators read.

    python -m match3_simulator.world_modeling.wm1_data build --root data/release --out runs/wm1/data/natural [--target settled]
    python -m match3_simulator.world_modeling.wm1_data verify --cache runs/wm1/data/natural
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from match3_simulator.learned_model.tokens import BOARD_WIDTH, N_CELLS
from match3_simulator.spec import NO_SPECIAL
from match3_simulator.world_modeling.settled_mask import settled_mask
from match3_simulator.world_modeling.wm1_verify import DEFAULT_IDENTITY, ReleaseIdentity, sha256_file

TARGETS: tuple[str, ...] = ("settled", "engine")
REGIME = "natural"
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
LEARNER_SPLITS: tuple[str, ...] = ("train", "validation")
EPISODE_ID_STRIDE = 100_000
CACHE_SCHEMA = 2
DERIVED_KEYS = ("cleared_mask", "m_per_column", "derived_goal", "residual_goal", "terminal", "mask_valid", "stripe_present", "stripe_activated", "stripe_created",
                "round0_activations", "inferred_activations", "consistent", "reshuffled_attempt")
SHIPPED_KEYS = ("board_before", "board_after", "specials_before", "specials_after", "action_index", "moves_left", "moves_left_next", "goals_left", "goals_left_next",
                "goal_colour", "level", "tier", "served_difficulty", "episode_id", "step_id", "player_id", "attempt_id")


class TestSplitAccess(RuntimeError):
    """Raised when a test-split player id is requested anywhere in the WM-1 pipeline."""


class OracleAccess(RuntimeError):
    """Raised when a path under an oracle directory would be opened."""


@dataclass(frozen=True)
class TestSplitUnlock:
    """Explicit, recorded permission to materialise the held-out test split (final evaluation only, after every selection decision)."""

    reason: str
    run_id: str
    config_sha: str

    def __post_init__(self) -> None:
        if not self.reason.strip() or not self.run_id.strip() or len(self.config_sha) < 12:
            raise ValueError("TestSplitUnlock needs a reason, a run id and the sha256 of the frozen evaluation configuration")

    def record(self) -> dict[str, str]:
        return {"reason": self.reason, "run_id": self.run_id, "config_sha": self.config_sha}


@dataclass(frozen=True)
class Splits:
    train: frozenset[int]
    validation: frozenset[int]
    test: frozenset[int]
    sha256: str

    def of(self, name: str) -> frozenset[int]:
        return getattr(self, name)


def load_splits(root: str | Path, identity: ReleaseIdentity = DEFAULT_IDENTITY) -> Splits:
    """Read splits.json (hash-checked against the registered release identity).

    Inputs: dataset root; release identity. Outputs: Splits (the test ids are kept to refuse them, or to select them under an unlock).
    """
    path = Path(root) / "splits.json"
    digest = sha256_file(path)
    if digest != identity.hashes["splits.json"]:
        raise RuntimeError(f"splits.json hash {digest} != {identity.hashes['splits.json']}")
    raw = json.loads(path.read_text())
    return Splits(frozenset(int(i) for i in raw["train"]), frozenset(int(i) for i in raw["validation"]), frozenset(int(i) for i in raw["test"]), digest)


def assert_not_test(player_ids: Iterable[int], splits: Splits) -> None:
    """Raise TestSplitAccess when any id belongs to the test split.

    Inputs: player ids; Splits. Outputs: none.
    """
    offending = sorted(set(int(i) for i in player_ids) & splits.test)
    if offending:
        raise TestSplitAccess(f"test-split player ids requested: {offending[:10]}{'...' if len(offending) > 10 else ''}")


def _open_payload(path: str | Path):
    """np.load of a logged artifact; refuses oracle paths.

    Inputs: file path. Outputs: NpzFile (caller closes).
    """
    path = Path(path)
    if any(part == "oracle" for part in path.parts):
        raise OracleAccess(f"refusing to open oracle artifact {path}")
    return np.load(path, allow_pickle=False)


def read_episode_table(shard_dir: Path) -> dict[tuple[int, int], dict[str, object]]:
    """Read the logged attempt table episodes.csv of a shard (never the oracle table).

    Inputs: shard directory. Outputs: (player_id, attempt_id) -> row dict with ints parsed for the fields WM-1 uses.
    """
    path = shard_dir / "episodes.csv"
    if any(part == "oracle" for part in path.parts):
        raise OracleAccess(str(path))
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            forbidden = {"mastery_before", "mastery_after", "oracle_win_probability", "churn_probability", "k_search"} & set(row)
            if forbidden:
                raise OracleAccess(f"logged table carries oracle fields {forbidden}")
            key = (int(row["player_id"]), int(row["attempt_id"]))
            out[key] = {"reshuffles": int(row["reshuffles"]), "striped_tiles_created": int(row["striped_tiles_created"]), "striped_tiles_activated": int(row["striped_tiles_activated"]),
                        "goals_cleared": int(row["goals_cleared"]), "moves_used": int(row["moves_used"]), "level": row["level"], "tier": row["tier"], "E": float(row["E"]), "R": int(row["R"]),
                        "served_goal_count": int(row["served_goal_count"]), "move_budget": int(row["move_budget"])}
    return out


def shard_dirs(root: str | Path, regime: str = REGIME) -> list[Path]:
    """Inputs: dataset root; regime. Outputs: sorted shard directories listed in release-manifest.json for the regime."""
    manifest = json.loads((Path(root) / "release-manifest.json").read_text())
    return [Path(root) / Path(shard["manifest"]).parent for shard in manifest["regimes"][regime]["shards"]]


def _derive_rows(arrays: dict[str, np.ndarray], attempts: dict[tuple[int, int], dict[str, object]]) -> dict[str, np.ndarray]:
    """Settled-pair reconstruction of every row (target='settled')."""
    n = len(arrays["action_index"])
    out = {"cleared_mask": np.zeros((n, N_CELLS), dtype=np.int8), "m_per_column": np.zeros((n, BOARD_WIDTH), dtype=np.int8), "derived_goal": np.zeros(n, dtype=np.int16),
           "residual_goal": np.zeros(n, dtype=np.int16), "terminal": np.zeros(n, dtype=bool), "mask_valid": np.zeros(n, dtype=bool), "stripe_present": np.zeros(n, dtype=bool),
           "stripe_activated": np.zeros(n, dtype=np.int8), "stripe_created": np.zeros(n, dtype=bool), "round0_activations": np.zeros(n, dtype=np.int8),
           "inferred_activations": np.zeros(n, dtype=np.int8), "consistent": np.zeros(n, dtype=bool), "reshuffled_attempt": np.zeros(n, dtype=bool)}
    boards = arrays["board_before"].reshape(n, N_CELLS)
    for i in range(n):
        result = settled_mask(arrays["board_before"][i], arrays["specials_before"][i], int(arrays["action_index"][i]), arrays["board_after"][i], arrays["specials_after"][i])
        row = attempts[(int(arrays["player_id"][i]), int(arrays["attempt_id"][i]))]
        reshuffled = row["reshuffles"] > 0
        derived = result.derived_goal(boards[i], int(arrays["goal_colour"][i]))
        delta = int(arrays["goals_left"][i]) - int(arrays["goals_left_next"][i])
        out["cleared_mask"][i] = result.mask
        out["m_per_column"][i] = result.m_per_column
        out["derived_goal"][i] = derived
        out["residual_goal"][i] = delta - derived
        out["terminal"][i] = int(arrays["goals_left_next"][i]) <= 0 or int(arrays["moves_left_next"][i]) <= 0
        out["mask_valid"][i] = (not reshuffled) and result.consistent
        out["stripe_present"][i] = bool((arrays["specials_before"][i] != NO_SPECIAL).any()) or result.created is not None
        out["stripe_activated"][i] = result.n_activations
        out["stripe_created"][i] = result.created is not None
        out["round0_activations"][i] = len(result.activations_round0)
        out["inferred_activations"][i] = len(result.activations_inferred)
        out["consistent"][i] = result.consistent
        out["reshuffled_attempt"][i] = reshuffled
    return out


def build_shard_cache(shard_dir: str | Path, out: str | Path, splits: Splits, *, target: str = "settled", shard_index: int | None = None,
                      identity: ReleaseIdentity = DEFAULT_IDENTITY, keep: tuple[str, ...] = LEARNER_SPLITS, unlock: TestSplitUnlock | None = None) -> Path:
    """Materialise the WM-1 cache of one shard: shipped fields of the kept splits' players plus the derived targets.

    Inputs: shard directory; output npz path; Splits; target ('settled' default; 'engine' is the flagged ablation, see wm1_engine_mask);
    shard index (parsed from the directory name when omitted); release identity; splits to keep (train + validation by default; keeping
    "test" requires a TestSplitUnlock, recorded in the header).
    Outputs: the written path (written as <out>.partial then renamed; never overwrites an existing file).
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    keep = tuple(keep)
    if any(name not in SPLIT_CODES for name in keep) or not keep:
        raise ValueError(f"keep must name splits among {tuple(SPLIT_CODES)}")
    if "test" in keep and unlock is None:
        raise TestSplitAccess("materialising the test split requires an explicit TestSplitUnlock")
    if "test" not in keep and unlock is not None:
        raise ValueError("an unlock was given but the test split is not kept")
    shard_dir = Path(shard_dir)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    index = int(shard_dir.name.split("-")[-1]) if shard_index is None else int(shard_index)
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    started = time.perf_counter()
    attempts = read_episode_table(shard_dir)
    with _open_payload(shard_dir / manifest["logged_artifacts"]["transitions"]["path"]) as npz:
        if int(npz["schema_version"][0]) != 3:
            raise RuntimeError("WM-1 needs transition schema 3 (specials present)")
        arrays = {key: np.asarray(npz[key]) for key in SHIPPED_KEYS}
    players = np.unique(arrays["player_id"])
    kept_sets = frozenset().union(*(splits.of(name) for name in keep))
    keep_ids = set(int(p) for p in players) & kept_sets
    dropped_test = sorted(set(int(p) for p in players) & splits.test) if "test" not in keep else []
    keep_rows = np.isin(arrays["player_id"], sorted(keep_ids))
    arrays = {key: values[keep_rows] for key, values in arrays.items()}
    if "test" not in keep:
        assert_not_test(arrays["player_id"], splits)
    if target == "settled":
        derived = _derive_rows(arrays, attempts)
    else:
        from match3_simulator.world_modeling.wm1_engine_mask import derive_rows_engine

        derived = derive_rows_engine(arrays, attempts, shard_dir=shard_dir)
    split_code = np.full(len(arrays["player_id"]), -1, dtype=np.int8)
    for name in keep:
        split_code[np.isin(arrays["player_id"], sorted(splits.of(name)))] = SPLIT_CODES[name]
    if np.any(split_code < 0):
        raise RuntimeError("a kept row belongs to no kept split")
    attempt_rows = {f"attempt_{name}": np.asarray([attempts[(int(p), int(a))][name] for p, a in zip(arrays["player_id"], arrays["attempt_id"])], dtype=np.int32)
                    for name in ("reshuffles", "striped_tiles_created", "striped_tiles_activated", "goals_cleared", "moves_used")}
    header = {"cache_schema": CACHE_SCHEMA, "target": target, "shard_index": index, "shard_manifest_sha256": sha256_file(manifest_path), "splits_sha256": splits.sha256,
              "payload_commit": identity.payload_commit, "release_identity": identity.name, "release_manifest_sha256": identity.hashes["release-manifest.json"],
              "splits_kept": list(keep), "unlock": unlock.record() if unlock is not None else None,
              "transitions_sha256": manifest["logged_artifacts"]["transitions"]["sha256"], "rows": int(len(arrays["action_index"])),
              "dropped_test_players": len(dropped_test), "kept_players": len(keep_ids), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": time.perf_counter() - started}
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + f".partial-{os.getpid()}")
    np.savez_compressed(partial, header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8), split=split_code,
                        episode_global_id=(index * EPISODE_ID_STRIDE + arrays["episode_id"].astype(np.int64)), **arrays, **derived, **attempt_rows)
    os.replace(str(partial) + ".npz", out)
    return out


def _build_one(payload: tuple) -> str:
    shard_dir, out, splits, target, index, identity, keep, unlock = payload
    return str(build_shard_cache(shard_dir, out, splits, target=target, shard_index=index, identity=identity, keep=keep, unlock=unlock))


def build_cache(root: str | Path, out_dir: str | Path, *, target: str = "settled", workers: int = 10, identity: ReleaseIdentity = DEFAULT_IDENTITY,
                keep: tuple[str, ...] = LEARNER_SPLITS, unlock: TestSplitUnlock | None = None, regime: str = REGIME) -> list[Path]:
    """Build every shard cache of one regime that does not exist yet (process pool).

    Inputs: dataset root; output directory; target; workers; release identity; splits to keep; unlock (test split only); regime. Outputs: list of cache paths.
    """
    splits = load_splits(root, identity)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    outputs = []
    for shard in shard_dirs(root, regime):
        index = int(shard.name.split("-")[-1])
        out = out_dir / f"shard-{index:03d}.wm1.npz"
        outputs.append(out)
        if not out.exists():
            tasks.append((str(shard), str(out), splits, target, index, identity, tuple(keep), unlock))
    if tasks:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
                list(pool.map(_build_one, tasks))
        else:
            for task in tasks:
                _build_one(task)
    return outputs


def _header(npz) -> dict[str, object]:
    return json.loads(bytes(np.asarray(npz["header"])).decode())


class EpisodeStore:
    """Concatenated transition rows of one split with per-episode offsets; rows are step-sorted within an episode (asserted at load).

    Batches are padded to the longest selected episode and carry train.BATCH_KEYS plus specials, next_specials, terminal, mask_valid, residual_goal.
    """

    def __init__(self, cache_dir: str | Path, split: str, *, splits: Splits, target: str = "settled", identity: ReleaseIdentity = DEFAULT_IDENTITY,
                 unlock: TestSplitUnlock | None = None):
        if split not in SPLIT_CODES:
            raise ValueError(f"split must be one of {tuple(SPLIT_CODES)}")
        if split == "test" and unlock is None:
            raise TestSplitAccess("EpisodeStore(split='test') requires an explicit TestSplitUnlock")
        if split != "test" and unlock is not None:
            raise ValueError("an unlock was given for a learner split")
        self.split = split
        self.unlock = unlock
        self.cache_dir = Path(cache_dir)
        paths = sorted(self.cache_dir.glob("shard-*.wm1.npz"))
        if not paths:
            raise FileNotFoundError(f"no WM-1 shard caches under {self.cache_dir}")
        parts: dict[str, list[np.ndarray]] = {}
        self.headers = []
        for path in paths:
            with _open_payload(path) as npz:
                header = _header(npz)
                if header.get("cache_schema") != CACHE_SCHEMA:
                    raise RuntimeError(f"{path}: cache schema {header.get('cache_schema')} != {CACHE_SCHEMA} (rebuild the cache with the release identity)")
                if (header["target"] != target or header["splits_sha256"] != splits.sha256 or header["payload_commit"] != identity.payload_commit
                        or header.get("release_identity") != identity.name):
                    raise RuntimeError(f"{path}: cache header does not match (target/splits/payload/release identity)")
                if split not in header.get("splits_kept", []):
                    raise RuntimeError(f"{path}: cache does not hold the {split} split (kept: {header.get('splits_kept')})")
                if split == "test" and header.get("unlock") is None:
                    raise TestSplitAccess(f"{path}: test rows present without a recorded unlock")
                self.headers.append(header)
                keep = np.asarray(npz["split"]) == SPLIT_CODES[split]
                for key in npz.files:
                    if key == "header":
                        continue
                    values = np.asarray(npz[key])
                    parts.setdefault(key, []).append(values[keep] if values.shape[:1] == keep.shape else values)
        self.arrays = {key: np.concatenate(values) for key, values in parts.items()}
        if split != "test":
            assert_not_test(np.unique(self.arrays["player_id"]), splits)
        expected = splits.of(split)
        if not set(int(p) for p in np.unique(self.arrays["player_id"])) <= expected:
            raise RuntimeError("store contains players outside its split")
        n = len(self.arrays["action_index"])
        ids = self.arrays["episode_global_id"]
        order = np.lexsort((self.arrays["step_id"], ids))
        for key, values in self.arrays.items():
            self.arrays[key] = values[order]
        ids = self.arrays["episode_global_id"]
        boundaries = np.flatnonzero(np.diff(ids)) + 1
        self.offsets = np.concatenate(([0], boundaries, [n]))
        self.lengths = np.diff(self.offsets)
        self.episode_ids = ids[self.offsets[:-1]]
        steps = self.arrays["step_id"]
        for start, length in zip(self.offsets[:-1], self.lengths):
            if not np.array_equal(steps[start : start + length], np.arange(length)):
                raise RuntimeError("episode rows are not contiguous zero-based steps")
        self.n_players = int(len(np.unique(self.arrays["player_id"])))
        for key in ("board_before", "board_after", "specials_before", "specials_after", "cleared_mask"):
            self.arrays[key] = self.arrays[key].reshape(n, -1)
        self._pad = {key: np.concatenate((values, np.zeros_like(values[:1]))) for key, values in self.arrays.items()}  # index -1 = padding row

    def __len__(self) -> int:
        return int(len(self.episode_ids))

    @property
    def n_rows(self) -> int:
        return int(len(self.arrays["action_index"]))

    def rows_of(self, episode_indices: np.ndarray) -> np.ndarray:
        """Inputs: episode positions (B,). Outputs: (B, T) row indices with -1 padding."""
        episode_indices = np.asarray(episode_indices, dtype=np.int64)
        lengths = self.lengths[episode_indices]
        t = int(lengths.max())
        grid = self.offsets[episode_indices][:, None] + np.arange(t)[None, :]
        return np.where(np.arange(t)[None, :] < lengths[:, None], grid, -1)

    def batch(self, episode_indices: np.ndarray, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
        """Pad the selected episodes into the world-model batch contract.

        Inputs: episode positions (B,) (positions in this store, not global ids); device.
        Outputs: dict of tensors (B, T, ...) / (B, T) / (B,) as documented in train.BATCH_KEYS plus the WM-1 keys.
        """
        rows = self.rows_of(episode_indices)
        first = rows[:, 0]
        p = self._pad
        long = lambda key: torch.as_tensor(p[key][rows].astype(np.int64), device=device)
        residual = p["residual_goal"][rows].astype(np.int64)
        return {
            "boards": long("board_before"), "next_boards": long("board_after"), "specials": long("specials_before"), "next_specials": long("specials_after"),
            "actions": long("action_index"), "goal_colours": long("goal_colour"), "moves_left": long("moves_left"), "goals_left": long("goals_left"),
            "next_moves_left": long("moves_left_next"), "next_goals_left": long("goals_left_next"), "cleared_masks": long("cleared_mask"),
            "refill_goal_cleared": torch.as_tensor(np.clip(residual, 0, None), device=device), "residual_goal": torch.as_tensor(residual, device=device),
            "terminal": torch.as_tensor(p["terminal"][rows], device=device), "mask_valid": torch.as_tensor(p["mask_valid"][rows] & (rows >= 0), device=device),
            "levels": torch.as_tensor(p["level"][first].astype(np.int64), device=device), "tiers": torch.as_tensor(p["tier"][first].astype(np.int64), device=device),
            "served_difficulty": torch.as_tensor(p["served_difficulty"][first].astype(np.float32), device=device), "step_mask": torch.as_tensor(rows >= 0, device=device),
            # strata keys for the evaluators (not consumed by the models: dropped by train.model_batch)
            "stripe_present": torch.as_tensor(p["stripe_present"][rows] & (rows >= 0), device=device), "stripe_activated": torch.as_tensor(p["stripe_activated"][rows].astype(np.int64), device=device),
            "player_ids": torch.as_tensor(p["player_id"][first].astype(np.int64), device=device), "attempt_ids": torch.as_tensor(p["attempt_id"][first].astype(np.int64), device=device),
        }

    def summary(self) -> dict[str, object]:
        a = self.arrays
        return {"split": self.split, "players": self.n_players, "episodes": len(self), "transitions": self.n_rows, "mask_valid_transitions": int(a["mask_valid"].sum()),
                "reshuffled_attempt_transitions": int(a["reshuffled_attempt"].sum()), "inconsistent_transitions": int((~a["consistent"]).sum()),
                "stripe_present_transitions": int(a["stripe_present"].sum()), "stripe_activated_transitions": int((a["stripe_activated"] > 0).sum()),
                "terminal_transitions": int(a["terminal"].sum()), "mean_episode_length": float(self.lengths.mean()), "max_episode_length": int(self.lengths.max()),
                "levels": {int(k): int(v) for k, v in zip(*np.unique(a["level"][self.offsets[:-1]], return_counts=True))},
                "residual_histogram": {int(k): int(v) for k, v in zip(*np.unique(a["residual_goal"][a["mask_valid"]], return_counts=True))}}


def verify_store_invariants(store: EpisodeStore, *, min_activation_agreement: float = 0.995) -> dict[str, object]:
    """Check the identities the settled-pair target must satisfy on every row (raises on a hard violation, reports the soft rates).

    Hard (every mask-valid transition): derived + residual == logged delta; per-column cleared count == m_per_column; a changed board has a
    non-empty mask; sum of created stripes per attempt == striped_tiles_created; sum of logged deltas per attempt == goals_cleared;
    mask_valid excludes exactly the reshuffled attempts' rows and the inconsistent rows.
    Soft (reported): per-attempt inferred activations == striped_tiles_activated (rate must be >= min_activation_agreement).
    """
    a = store.arrays
    n = store.n_rows
    boards = a["board_before"].astype(np.int64)
    goal_cells = boards == a["goal_colour"].astype(np.int64)[:, None]
    derived = (a["cleared_mask"].astype(np.int64) * goal_cells).sum(axis=1)
    if not np.array_equal(derived, a["derived_goal"].astype(np.int64)):
        raise RuntimeError("derived_goal != mask-derived count")
    delta = a["goals_left"].astype(np.int64) - a["goals_left_next"].astype(np.int64)
    if not np.array_equal(derived + a["residual_goal"].astype(np.int64), delta):
        raise RuntimeError("derived + residual != logged goal delta")
    # per-column cleared count (columns after the swap) == m_per_column
    from match3_simulator.learned_model.tokens import CELL1_TOKEN, CELL2_TOKEN

    cols = np.tile(np.arange(N_CELLS) % BOARD_WIDTH, (n, 1))
    idx = np.arange(n)
    c1, c2 = CELL1_TOKEN[a["action_index"]], CELL2_TOKEN[a["action_index"]]
    cols[idx, c1], cols[idx, c2] = cols[idx, c2].copy(), cols[idx, c1].copy()
    per_col = np.zeros((n, BOARD_WIDTH), dtype=np.int64)
    for col in range(BOARD_WIDTH):
        per_col[:, col] = ((cols == col) & (a["cleared_mask"] == 1)).sum(axis=1)
    valid = a["mask_valid"]
    if not np.array_equal(per_col[valid], a["m_per_column"][valid].astype(np.int64)):
        raise RuntimeError("per-column cleared count != m_per_column on a mask-valid row")
    changed = (a["board_before"] != a["board_after"]).any(axis=1)
    empty = (a["cleared_mask"].sum(axis=1) == 0)
    if np.any(changed & empty & valid):
        raise RuntimeError("changed board with an empty mask")
    if not np.array_equal(valid, ~a["reshuffled_attempt"] & a["consistent"]):
        raise RuntimeError("mask_valid must equal (not reshuffled) & consistent")
    # per-attempt identities
    key = a["player_id"].astype(np.int64) * 1000 + a["attempt_id"].astype(np.int64)
    uniq, inverse = np.unique(key, return_inverse=True)
    created_sum = np.bincount(inverse, weights=a["stripe_created"].astype(np.float64)).astype(np.int64)
    created_logged = np.zeros(len(uniq), dtype=np.int64)
    created_logged[inverse] = a["attempt_striped_tiles_created"]
    if not np.array_equal(created_sum, created_logged):
        raise RuntimeError("sum of created stripes per attempt != striped_tiles_created")
    delta_sum = np.bincount(inverse, weights=delta.astype(np.float64)).astype(np.int64)
    goals_logged = np.zeros(len(uniq), dtype=np.int64)
    goals_logged[inverse] = a["attempt_goals_cleared"]
    if not np.array_equal(delta_sum, goals_logged):
        raise RuntimeError("sum of goal deltas per attempt != goals_cleared")
    activated_sum = np.bincount(inverse, weights=a["stripe_activated"].astype(np.float64)).astype(np.int64)
    activated_logged = np.zeros(len(uniq), dtype=np.int64)
    activated_logged[inverse] = a["attempt_striped_tiles_activated"]
    reshuffled_attempt = np.zeros(len(uniq), dtype=bool)
    reshuffled_attempt[inverse] = a["reshuffled_attempt"]
    agreement = float(np.mean(activated_sum[~reshuffled_attempt] == activated_logged[~reshuffled_attempt]))
    if agreement < min_activation_agreement:
        raise RuntimeError(f"inferred activations agree with striped_tiles_activated on only {agreement:.4%} of attempts")
    return {"rows": n, "attempts": int(len(uniq)), "mask_valid_rate": float(valid.mean()), "activation_agreement_per_attempt": agreement,
            "reshuffled_attempts": int(reshuffled_attempt.sum()), "inconsistent_rows": int((~a["consistent"]).sum()), "changed_rows": int(changed.sum()),
            "residual_min": int(a["residual_goal"][valid].min()), "residual_max": int(a["residual_goal"][valid].max()),
            "residual_zero_rate": float((a["residual_goal"][valid] == 0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", default="data/release")
    build.add_argument("--out", default="runs/wm1/data/natural")
    build.add_argument("--target", choices=TARGETS, default="settled")
    build.add_argument("--workers", type=int, default=10)
    build.add_argument("--regime", default=REGIME)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", default="data/release")
    verify.add_argument("--cache", default="runs/wm1/data/natural")
    verify.add_argument("--target", choices=TARGETS, default="settled")
    verify.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.command == "build":
        started = time.perf_counter()
        paths = build_cache(args.root, args.out, target=args.target, workers=args.workers, regime=args.regime)
        print(f"built {len(paths)} shard caches under {args.out} in {time.perf_counter() - started:.0f}s")
    else:
        splits = load_splits(args.root)
        report = {}
        for split in ("train", "validation"):
            store = EpisodeStore(args.cache, split, splits=splits, target=args.target)
            report[split] = {"summary": store.summary(), "invariants": verify_store_invariants(store)}
            print(json.dumps(report[split], indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()


__all__ = ["CACHE_SCHEMA", "DERIVED_KEYS", "EPISODE_ID_STRIDE", "EpisodeStore", "LEARNER_SPLITS", "OracleAccess", "SPLIT_CODES", "Splits", "TARGETS", "TestSplitAccess",
           "TestSplitUnlock", "assert_not_test", "build_cache", "build_shard_cache", "load_splits", "read_episode_table", "shard_dirs", "verify_store_invariants"]
