"""Generate and audit the accepted Wrong Move reference datasets."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

from .board import creates_match
from .calibrate import load_win_propensity_model
from .learned_model.tokens import action_to_index
from .retention import (
    ChurnSchedule,
    MasteryConfig,
    PlayerTrajectory,
    WinPropensityModel,
    simulate_player_trajectory,
)
from .scm import LEVELS, TIER_LOGITS, TIER_NAMES
from .simulate import write_player_dataset
from .spec import Action, BENCHMARK_CONFIG, BenchmarkConfig, PlayerSkill


PACKAGE_ROOT = Path(__file__).resolve().parent
ACCEPTED_SPEC_PATH = PACKAGE_ROOT / "accepted_benchmark.json"
ACCEPTED_CALIBRATION_PATH = PACKAGE_ROOT / "accepted_calibration.json"
WIN_PROPENSITY_PATH = PACKAGE_ROOT / "win_propensity_calibration.json"
DEFAULT_OUTPUT = Path("data/releases/wrong-move-reference-v1")
REQUIRED_LOGGED_FIELDS = {
    "player_id",
    "attempt_id",
    "level",
    "tier",
    "E",
    "R",
    "completion_margin",
    "churn_after",
    "striped_tiles_created",
    "striped_tiles_activated",
}
FORBIDDEN_LOGGED_FIELDS = {
    "mastery_before",
    "mastery_after",
    "oracle_win_probability",
    "churn_probability",
    "k_search",
    "k_pattern",
    "k_planning",
    "k_strategy",
}
REQUIRED_ORACLE_FIELDS = {
    "player_id",
    "attempt_id",
    "mastery_before",
    "mastery_after",
    "completion_margin",
    "oracle_win_probability",
    "churn_probability",
    "k_search",
    "k_pattern",
    "k_planning",
    "k_strategy",
}

_WORKER_CONTEXT: tuple[
    WinPropensityModel,
    int,
    int,
    BenchmarkConfig,
    ChurnSchedule,
    MasteryConfig,
    tuple[float, ...],
    tuple[float, ...],
] | None = None


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision() -> str | None:
    git = "/usr/bin/git" if Path("/usr/bin/git").is_file() else "git"
    result = subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_accepted_spec(path: str | Path = ACCEPTED_SPEC_PATH) -> dict[str, object]:
    spec = json.loads(Path(path).read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported accepted benchmark schema")
    if spec.get("status") != "accepted_reference_benchmark":
        raise ValueError("benchmark configuration is not accepted")
    if spec["acceptance"]["retuned_after_validation"] is not False:
        raise ValueError("accepted configuration must remain unretuned")
    natural = spec["regimes"]["natural"]
    randomized = spec["regimes"]["randomized"]
    expected_sigmas = [
        math.hypot(float(gain), float(sigma))
        for gain, sigma in zip(
            natural["skill_gains"], natural["sigmas"], strict=True
        )
    ]
    if randomized["skill_gains"] != [0.0, 0.0, 0.0] or not np.allclose(
        randomized["sigmas"], expected_sigmas, atol=1e-14, rtol=0.0
    ):
        raise ValueError("randomized assignment does not preserve marginal variance")
    return spec


def resolved_configs(
    spec: dict[str, object], regime: str
) -> tuple[
    BenchmarkConfig,
    ChurnSchedule,
    MasteryConfig,
    tuple[float, ...],
    tuple[float, ...],
]:
    if regime not in spec["regimes"]:
        raise ValueError(f"unknown assignment regime {regime}")
    candidate = spec["candidate"]
    values = [candidate["levels"][level.name] for level in LEVELS]
    mastery = MasteryConfig(
        initial=float(candidate["experience_initial"]),
        update_rate=float(candidate["experience_update_rate"]),
    )
    churn = ChurnSchedule(
        level_names=tuple(level.name for level in LEVELS),
        intercepts=tuple(float(value["intercept"]) for value in values),
        deviation_coefficients=tuple(
            float(value["experience_curvature_easy"]) for value in values
        ),
        overchallenge_deviation_coefficients=tuple(
            float(value["experience_curvature_overchallenge"])
            for value in values
        ),
        mastery_target=float(candidate["experience_target"]),
        margin_deviation_coefficients=tuple(
            float(value["margin_curvature_easy"]) for value in values
        ),
        margin_overchallenge_deviation_coefficients=tuple(
            float(value["margin_curvature_overchallenge"])
            for value in values
        ),
        margin_targets=tuple(float(value["margin_target"]) for value in values),
    )
    assignment = spec["regimes"][regime]
    gains = tuple(float(value) for value in assignment["skill_gains"])
    sigmas = tuple(float(value) for value in assignment["sigmas"])
    return BENCHMARK_CONFIG, churn, mastery, gains, sigmas


def verify_release_inputs(
    spec: dict[str, object],
    calibration_path: str | Path = ACCEPTED_CALIBRATION_PATH,
) -> None:
    source = spec["source"]
    calibration = Path(calibration_path)
    if sha256(calibration) != source["quota_calibration_sha256"]:
        raise RuntimeError("accepted quota calibration hash mismatch")
    if sha256(WIN_PROPENSITY_PATH) != source["win_propensity_sha256"]:
        raise RuntimeError("win propensity model hash mismatch")
    for name, expected in source["mechanics_sha256"].items():
        if sha256(PACKAGE_ROOT / name) != expected:
            raise RuntimeError(f"accepted mechanics hash mismatch for {name}")


def _initialize_worker(
    model: WinPropensityModel,
    seed: int,
    max_attempts: int,
    benchmark: BenchmarkConfig,
    churn: ChurnSchedule,
    mastery: MasteryConfig,
    gains: tuple[float, ...],
    sigmas: tuple[float, ...],
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (
        model,
        seed,
        max_attempts,
        benchmark,
        churn,
        mastery,
        gains,
        sigmas,
    )


def _simulate_worker(player_id: int) -> PlayerTrajectory:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("release worker was not initialized")
    model, seed, max_attempts, benchmark, churn, mastery, gains, sigmas = (
        _WORKER_CONTEXT
    )
    return simulate_player_trajectory(
        model,
        player_id=player_id,
        seed=seed,
        max_attempts=max_attempts,
        benchmark=benchmark,
        churn_config=churn,
        mastery_config=mastery,
        dda_gains=gains,
        e_sigmas=sigmas,
    )


def simulate_player_ids(
    player_ids: Iterable[int],
    model: WinPropensityModel,
    *,
    seed: int,
    max_attempts: int,
    benchmark: BenchmarkConfig,
    churn: ChurnSchedule,
    mastery: MasteryConfig,
    gains: tuple[float, ...],
    sigmas: tuple[float, ...],
    workers: int,
) -> list[PlayerTrajectory]:
    ids = [int(player_id) for player_id in player_ids]
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        _initialize_worker(
            model,
            seed,
            max_attempts,
            benchmark,
            churn,
            mastery,
            gains,
            sigmas,
        )
        return [_simulate_worker(player_id) for player_id in ids]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(
            model,
            seed,
            max_attempts,
            benchmark,
            churn,
            mastery,
            gains,
            sigmas,
        ),
    ) as executor:
        return list(executor.map(_simulate_worker, ids, chunksize=1))


def create_splits(player_ids: list[int], seed: int) -> dict[str, object]:
    shuffled = np.random.default_rng(seed).permutation(player_ids)
    n_players = len(shuffled)
    n_validation = int(round(0.15 * n_players))
    n_test = int(round(0.15 * n_players))
    n_train = n_players - n_validation - n_test
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("release split requires at least one player per partition")
    return {
        "schema_version": 1,
        "seed": seed,
        "unit": "player",
        "train": sorted(int(value) for value in shuffled[:n_train]),
        "validation": sorted(
            int(value) for value in shuffled[n_train : n_train + n_validation]
        ),
        "test": sorted(int(value) for value in shuffled[n_train + n_validation :]),
    }


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _write_regime(
    root: Path,
    regime: str,
    player_ids: list[int],
    *,
    spec: dict[str, object],
    spec_sha256: str,
    model: WinPropensityModel,
    seed: int,
    max_attempts: int,
    shard_size: int,
    workers: int,
    include_transitions: bool,
) -> dict[str, object]:
    benchmark, churn, mastery, gains, sigmas = resolved_configs(spec, regime)
    entries = []
    regime_root = root / regime
    for shard_index, start in enumerate(range(0, len(player_ids), shard_size)):
        shard_ids = player_ids[start : start + shard_size]
        trajectories = simulate_player_ids(
            shard_ids,
            model,
            seed=seed,
            max_attempts=max_attempts,
            benchmark=benchmark,
            churn=churn,
            mastery=mastery,
            gains=gains,
            sigmas=sigmas,
            workers=workers,
        )
        shard_root = regime_root / f"shard-{shard_index:03d}"
        manifest_path = write_player_dataset(
            trajectories,
            shard_root,
            seed=seed,
            max_attempts=max_attempts,
            include_transitions=include_transitions,
            benchmark=benchmark,
            churn_config=churn,
            mastery_config=mastery,
            dda_gains=gains,
            e_sigmas=sigmas,
            provenance={
                "accepted_benchmark_sha256": spec_sha256,
                "regime": regime,
                "shard_index": shard_index,
                "player_ids": shard_ids,
            },
        )
        manifest = json.loads(manifest_path.read_text())
        entries.append(
            {
                "index": shard_index,
                "player_id_min": min(shard_ids),
                "player_id_max": max(shard_ids),
                "players": len(shard_ids),
                "manifest": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": sha256(manifest_path),
                "counts": manifest["counts"],
            }
        )
        print(
            f"{regime} shard {shard_index + 1}/"
            f"{math.ceil(len(player_ids) / shard_size)} complete",
            flush=True,
        )
    return {
        "assignment": {"skill_gains": list(gains), "sigmas": list(sigmas)},
        "transitions_included": include_transitions,
        "shards": entries,
        "counts": {
            key: sum(int(entry["counts"][key]) for entry in entries)
            for key in ("players", "attempts", "transitions", "churned_players")
        },
    }


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _artifact_path(shard_root: Path, manifest: dict[str, object], name: str) -> Path:
    for group in ("logged_artifacts", "oracle_artifacts"):
        if name in manifest[group]:
            artifact = manifest[group][name]
            path = shard_root / artifact["path"]
            if sha256(path) != artifact["sha256"]:
                raise RuntimeError(f"artifact hash mismatch for {path}")
            return path
    raise KeyError(name)


def _audit_regime(
    root: Path,
    regime: str,
    report: dict[str, object],
    expected_players: set[int],
) -> tuple[dict[str, object], dict[int, tuple[float, ...]]]:
    attempts_by_player: dict[int, list[int]] = {}
    player_skills: dict[int, tuple[float, ...]] = {}
    level_values = {
        level.name: {
            "difficulty": [],
            "assignment_residual": [],
            "effective_skill": [],
            "completion": [],
            "churn": [],
            "created": [],
            "activated": [],
        }
        for level in LEVELS
    }
    total_attempts = 0
    total_transitions = 0
    semantic_legal_actions = 0
    missing_values = 0

    for entry in report["shards"]:
        manifest_path = root / entry["manifest"]
        if sha256(manifest_path) != entry["manifest_sha256"]:
            raise RuntimeError(f"shard manifest hash mismatch for {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        shard_root = manifest_path.parent
        configuration_sha = hashlib.sha256(
            json.dumps(manifest["configuration"], sort_keys=True).encode()
        ).hexdigest()
        if configuration_sha != manifest["configuration_sha256"]:
            raise RuntimeError(
                f"configuration hash mismatch for {manifest_path}"
            )
        if manifest["configuration"]["assignment"] != report["assignment"]:
            raise RuntimeError(
                f"assignment configuration mismatch for {manifest_path}"
            )
        logged_path = _artifact_path(shard_root, manifest, "attempts")
        oracle_path = _artifact_path(shard_root, manifest, "attempt_state")
        with oracle_path.open(newline="") as handle:
            oracle_rows = list(csv.DictReader(handle))
        if not oracle_rows or not REQUIRED_ORACLE_FIELDS <= set(oracle_rows[0]):
            raise RuntimeError(f"oracle schema mismatch for {oracle_path}")
        oracle_by_key = {}
        for row in oracle_rows:
            key = (int(row["player_id"]), int(row["attempt_id"]))
            if key in oracle_by_key:
                raise RuntimeError(f"duplicate oracle attempt {key}")
            oracle_by_key[key] = row
            skill = tuple(
                float(row[f"k_{name}"])
                for name in ("search", "pattern", "planning", "strategy")
            )
            previous = player_skills.setdefault(key[0], skill)
            if previous != skill:
                raise RuntimeError(f"player skill changed for {key[0]}")

        with logged_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"logged schema is empty for {logged_path}")
            fields = set(reader.fieldnames)
            if not REQUIRED_LOGGED_FIELDS <= fields or FORBIDDEN_LOGGED_FIELDS & fields:
                raise RuntimeError(f"logged schema mismatch for {logged_path}")
            logged_rows = list(reader)
        if len(logged_rows) != len(oracle_rows):
            raise RuntimeError(f"logged/oracle row mismatch for {shard_root}")
        for row in logged_rows:
            missing_values += sum(value == "" for value in row.values())
            player_id = int(row["player_id"])
            attempt_id = int(row["attempt_id"])
            key = (player_id, attempt_id)
            if key not in oracle_by_key:
                raise RuntimeError(f"logged attempt lacks oracle row {key}")
            attempts = attempts_by_player.setdefault(player_id, [])
            if attempt_id in attempts:
                raise RuntimeError(f"duplicate logged attempt {key}")
            attempts.append(attempt_id)
            level_name = row["level"]
            if level_name not in level_values or row["tier"] not in TIER_NAMES:
                raise RuntimeError(f"unsupported level or tier in {key}")
            outcome = int(row["R"])
            churn = int(row["churn_after"])
            margin = float(row["completion_margin"])
            if outcome not in (0, 1) or churn not in (0, 1):
                raise RuntimeError(f"non-binary outcome in {key}")
            if not -1.0 <= margin <= 1.0:
                raise RuntimeError(f"completion margin outside support in {key}")
            if (outcome and margin < 0.0) or (not outcome and margin > 0.0):
                raise RuntimeError(f"completion margin sign mismatch in {key}")
            skill = PlayerSkill(player_skills[player_id])
            level = next(value for value in LEVELS if value.name == level_name)
            difficulty = float(row["E"])
            baseline = float(TIER_LOGITS[TIER_NAMES.index(row["tier"])])
            values = level_values[level_name]
            values["difficulty"].append(difficulty)
            values["assignment_residual"].append(difficulty - baseline)
            values["effective_skill"].append(skill.effective_for(level))
            values["completion"].append(outcome)
            values["churn"].append(churn)
            values["created"].append(int(row["striped_tiles_created"]))
            values["activated"].append(int(row["striped_tiles_activated"]))
        total_attempts += len(logged_rows)

        if "transitions" in manifest["logged_artifacts"]:
            transitions_path = _artifact_path(shard_root, manifest, "transitions")
            with np.load(transitions_path, allow_pickle=False) as arrays:
                rows = len(arrays["action_index"])
                if arrays["board_before"].shape != (rows, 8, 8):
                    raise RuntimeError(f"transition board shape mismatch in {shard_root}")
                if arrays["specials_before"].shape != (rows, 8, 8):
                    raise RuntimeError(f"transition specials shape mismatch in {shard_root}")
                if np.any((arrays["specials_before"] < 0) | (arrays["specials_before"] > 2)):
                    raise RuntimeError(f"special code outside support in {shard_root}")
                actions = arrays["action"]
                expected_indices = (
                    (actions[:, 0] * 8 + actions[:, 1]) * 2 + actions[:, 2]
                )
                if not np.array_equal(expected_indices, arrays["action_index"]):
                    raise RuntimeError(f"action index mismatch in {shard_root}")
                player_ids = arrays["player_id"]
                attempt_ids = arrays["attempt_id"]
                steps = arrays["step_id"]
                same_episode = (player_ids[1:] == player_ids[:-1]) & (
                    attempt_ids[1:] == attempt_ids[:-1]
                )
                if np.any(steps[1:][same_episode] != steps[:-1][same_episode] + 1):
                    raise RuntimeError(f"non-contiguous transition steps in {shard_root}")
                if np.any(steps[1:][~same_episode] != 0) or (rows and steps[0] != 0):
                    raise RuntimeError(f"transition step did not reset in {shard_root}")
                for board, raw_action, action_index in zip(
                    arrays["board_before"], actions, arrays["action_index"], strict=True
                ):
                    action = Action(*(int(value) for value in raw_action))
                    if action_to_index(action) != int(action_index) or not creates_match(
                        board, action
                    ):
                        raise RuntimeError(f"illegal logged action in {shard_root}")
                semantic_legal_actions += rows
                total_transitions += rows

    if set(attempts_by_player) != expected_players:
        raise RuntimeError(f"{regime} player IDs do not match release plan")
    for player_id, attempts in attempts_by_player.items():
        if sorted(attempts) != list(range(1, max(attempts) + 1)):
            raise RuntimeError(f"non-contiguous attempts for player {player_id}")
    if missing_values:
        raise RuntimeError(f"{regime} contains {missing_values} missing values")
    expected_counts = report["counts"]
    if total_attempts != expected_counts["attempts"]:
        raise RuntimeError(f"{regime} attempt count differs from manifests")
    if report["transitions_included"] and total_transitions != expected_counts[
        "transitions"
    ]:
        raise RuntimeError(f"{regime} transition count differs from manifests")

    level_report = {}
    for level_name, values in level_values.items():
        difficulty = np.asarray(values["difficulty"], dtype=np.float64)
        if len(difficulty) == 0:
            level_report[level_name] = {"attempts": 0}
            continue
        residual = np.asarray(values["assignment_residual"], dtype=np.float64)
        effective = np.asarray(values["effective_skill"], dtype=np.float64)
        level_report[level_name] = {
            "attempts": len(difficulty),
            "difficulty_mean": float(difficulty.mean()),
            "difficulty_std": float(difficulty.std()),
            "difficulty_min": float(difficulty.min()),
            "difficulty_max": float(difficulty.max()),
            "outside_calibrated_support_fraction": float(
                np.mean((difficulty < -6.0) | (difficulty > 6.0))
            ),
            "assignment_residual_mean": float(residual.mean()),
            "assignment_residual_std": float(residual.std()),
            "skill_assignment_correlation": _correlation(
                values["effective_skill"], values["assignment_residual"]
            ),
            "completion_rate": float(np.mean(values["completion"])),
            "churn_events": int(np.sum(values["churn"])),
            "striped_tiles_created": int(np.sum(values["created"])),
            "striped_tiles_activated": int(np.sum(values["activated"])),
        }
    lengths = np.asarray(list(map(len, attempts_by_player.values())))
    return (
        {
            "players": len(attempts_by_player),
            "attempts": total_attempts,
            "transitions": int(expected_counts["transitions"]),
            "transition_rows_materialized": total_transitions,
            "semantic_legal_actions_checked": semantic_legal_actions,
            "missing_values": missing_values,
            "players_active_at_attempt_20": int(np.sum(lengths >= 20)),
            "attempts_per_player_quantiles": {
                str(quantile): float(np.quantile(lengths, quantile))
                for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "levels": level_report,
        },
        player_skills,
    )


def audit_release(
    root: Path,
    release_manifest: dict[str, object],
    splits: dict[str, object],
) -> dict[str, object]:
    split_sets = {name: set(splits[name]) for name in ("train", "validation", "test")}
    if any(split_sets[left] & split_sets[right] for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )):
        raise RuntimeError("player splits overlap")
    expected_players = set.union(*split_sets.values())
    regimes = {}
    skills = {}
    for regime in ("natural", "randomized"):
        regimes[regime], skills[regime] = _audit_regime(
            root,
            regime,
            release_manifest["regimes"][regime],
            expected_players,
        )
    if skills["natural"] != skills["randomized"]:
        raise RuntimeError("natural and randomized regimes do not share players")

    matching = {}
    enforce = len(expected_players) >= 500
    for level in (level.name for level in LEVELS):
        natural = regimes["natural"]["levels"][level]
        randomized = regimes["randomized"]["levels"][level]
        if natural["attempts"] == 0 or randomized["attempts"] == 0:
            matching[level] = {
                "checks_enforced": False,
                "status": "not_evaluated_no_rows",
            }
            continue
        mean_difference = abs(
            natural["assignment_residual_mean"]
            - randomized["assignment_residual_mean"]
        )
        std_ratio = randomized["assignment_residual_std"] / natural[
            "assignment_residual_std"
        ]
        checks = {
            "natural_assignment_depends_on_skill": (
                natural["skill_assignment_correlation"] > 0.5
            ),
            "randomized_assignment_independent_of_skill": (
                abs(randomized["skill_assignment_correlation"]) < 0.05
            ),
            "marginal_mean_matched": mean_difference < 0.10,
            "marginal_standard_deviation_matched": 0.90 < std_ratio < 1.10,
        }
        if enforce and not all(checks.values()):
            raise RuntimeError(f"assignment matching QC failed for {level}: {checks}")
        matching[level] = {
            "mean_difference": float(mean_difference),
            "standard_deviation_ratio": float(std_ratio),
            "checks_enforced": enforce,
            "checks": checks,
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "split_counts": {name: len(values) for name, values in split_sets.items()},
        "player_disjoint_splits": True,
        "matched_player_skills": True,
        "regimes": regimes,
        "assignment_matching": matching,
    }


def generate_release(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    spec_path: str | Path = ACCEPTED_SPEC_PATH,
    calibration_path: str | Path = ACCEPTED_CALIBRATION_PATH,
    players: int | None = None,
    max_attempts: int | None = None,
    shard_size: int | None = None,
    workers: int = 1,
    include_transitions: bool = True,
) -> Path:
    output_path = Path(output).resolve()
    partial = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output_path} or {partial}")
    spec_file = Path(spec_path).resolve()
    calibration_file = Path(calibration_path).resolve()
    spec = load_accepted_spec(spec_file)
    verify_release_inputs(spec, calibration_file)
    dataset = spec["dataset"]
    n_players = int(dataset["players"] if players is None else players)
    attempts = int(dataset["max_attempts"] if max_attempts is None else max_attempts)
    shard = int(dataset["shard_size"] if shard_size is None else shard_size)
    if n_players < 3 or attempts < 1 or shard < 1:
        raise ValueError("release dimensions must be positive and include three players")
    seed = int(dataset["seed"])
    split_seed = int(dataset["split_seed"])
    player_ids = list(range(n_players))
    partial.mkdir(parents=True)
    os.environ["MATCH3_CALIBRATION_PATH"] = str(calibration_file)
    spec_sha = sha256(spec_file)
    copied_spec = partial / "accepted_benchmark.json"
    copied_spec.write_bytes(spec_file.read_bytes())
    splits = create_splits(player_ids, split_seed)
    splits_path = _write_json(partial / "splits.json", splits)
    model = load_win_propensity_model(WIN_PROPENSITY_PATH)
    release_manifest = {
        "schema_version": 1,
        "status": "accepted_reference_dataset",
        "name": spec["name"],
        "code_sha": git_revision(),
        "accepted_benchmark": {
            "path": copied_spec.name,
            "sha256": spec_sha,
        },
        "quota_calibration_sha256": sha256(calibration_file),
        "win_propensity_sha256": sha256(WIN_PROPENSITY_PATH),
        "seed": seed,
        "split_seed": split_seed,
        "players": n_players,
        "max_attempts": attempts,
        "shard_size": shard,
        "include_transitions": include_transitions,
        "splits": {"path": splits_path.name, "sha256": sha256(splits_path)},
        "regimes": {},
    }
    for regime in ("natural", "randomized"):
        release_manifest["regimes"][regime] = _write_regime(
            partial,
            regime,
            player_ids,
            spec=spec,
            spec_sha256=spec_sha,
            model=model,
            seed=seed,
            max_attempts=attempts,
            shard_size=shard,
            workers=workers,
            include_transitions=include_transitions,
        )
    manifest_path = _write_json(partial / "release-manifest.json", release_manifest)
    qc = audit_release(partial, release_manifest, splits)
    qc_path = _write_json(partial / "qc.json", qc)
    release_manifest["qc"] = {"path": qc_path.name, "sha256": sha256(qc_path)}
    _write_json(manifest_path, release_manifest)
    partial.rename(output_path)
    print(f"release complete: {output_path}")
    return output_path / "release-manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spec", type=Path, default=ACCEPTED_SPEC_PATH)
    parser.add_argument(
        "--calibration", type=Path, default=ACCEPTED_CALIBRATION_PATH
    )
    parser.add_argument("--players", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-transitions", action="store_true")
    args = parser.parse_args()
    generate_release(
        args.out,
        spec_path=args.spec,
        calibration_path=args.calibration,
        players=args.players,
        max_attempts=args.max_attempts,
        shard_size=args.shard_size,
        workers=args.workers,
        include_transitions=not args.no_transitions,
    )


if __name__ == "__main__":
    main()