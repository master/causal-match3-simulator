"""Stage 03a: copy the verified release payload into the package, verify it against the registry, and run the structural release checks.

``copy``   — copy ``<source>`` (the verified payload) to ``data/release`` (``.cache`` excluded) and verify.
``verify`` — overlay ``wm1_verify.verify`` with the registry identity (manifest chain, counts, bytes, every file hash).
``check``  — original ``release.audit_release`` (shapes, special codes, action-index formula, step contiguity, legality, disjoint splits,
             assignment-matching QC) plus additional checks: unique keys, episode/transition joins, exact before/after continuity, terminal
             consistency, colour support per level, absorbing churn and the 20-attempt cap, oracle leakage, one split per player, matched ids
             across regimes. Structural validity only; physical replay is ``match3_simulator.evaluation replay``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from match3_simulator.evaluation.common import DATA, Logger, assert_package_import, read_json, relpath, stage_dir, write_json
from match3_simulator.evaluation.release_registry import identity, local_root

STAGE = "dataset_checks"
FORBIDDEN_LOGGED = {"mastery_before", "mastery_after", "oracle_win_probability", "churn_probability", "k_search", "k_pattern", "k_planning", "k_strategy"}


def copy_payload(source: Path, target: Path = DATA) -> Path:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {relpath(target)}")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".cache", "__pycache__"))
    return target


def verify_payload(root: Path = DATA) -> dict[str, object]:
    from match3_simulator.world_modeling.wm1_verify import verify

    return verify(root, identity=identity(), write=True)


def _episodes(root: Path, regime: str) -> tuple[dict[tuple[int, int], dict[str, str]], list[str]]:
    manifest = read_json(root / "release-manifest.json")
    rows: dict[tuple[int, int], dict[str, str]] = {}
    fields: list[str] = []
    for shard in manifest["regimes"][regime]["shards"]:
        shard_dir = root / Path(shard["manifest"]).parent
        with (shard_dir / "episodes.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            for row in reader:
                key = (int(row["player_id"]), int(row["attempt_id"]))
                if key in rows:
                    raise RuntimeError(f"duplicate episode key {key} in {regime}")
                rows[key] = row
    return rows, fields


def _transitions(root: Path, regime: str) -> dict[str, np.ndarray]:
    manifest = read_json(root / "release-manifest.json")
    parts: dict[str, list[np.ndarray]] = {}
    for shard in manifest["regimes"][regime]["shards"]:
        shard_dir = root / Path(shard["manifest"]).parent
        with np.load(shard_dir / "transitions.npz", allow_pickle=False) as npz:
            for key in npz.files:
                if key == "schema_version":
                    parts.setdefault(key, []).append(np.asarray(npz[key]))
                    continue
                parts.setdefault(key, []).append(np.asarray(npz[key]))
    out = {key: np.concatenate(values) for key, values in parts.items() if key != "schema_version"}
    out["schema_version"] = np.unique(np.concatenate(parts["schema_version"]))
    return out


def _json_differences(left, right, path: str = "", acc: dict | None = None) -> dict[str, object]:
    """Count discrete differences and the maximum absolute float difference between two JSON documents (floats drift at ~1e-15 across numpy builds)."""
    acc = acc if acc is not None else {"discrete_differences": 0, "max_abs_float_diff": 0.0, "float_fields_differing": []}
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            acc["discrete_differences"] += 1
        for key in set(left) & set(right):
            _json_differences(left[key], right[key], f"{path}/{key}", acc)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            acc["discrete_differences"] += 1
        for index, (a, b) in enumerate(zip(left, right)):
            _json_differences(a, b, f"{path}[{index}]", acc)
    elif isinstance(left, float) or isinstance(right, float):
        diff = abs(float(left) - float(right))
        if diff > 0:
            acc["max_abs_float_diff"] = max(acc["max_abs_float_diff"], diff)
            acc["float_fields_differing"].append(path)
    elif left != right:
        acc["discrete_differences"] += 1
    return acc


def check_release(root: Path = DATA, *, log=print) -> dict[str, object]:
    from match3_simulator.release import audit_release
    from match3_simulator.scm import LEVELS, TIER_NAMES

    started = time.perf_counter()
    manifest = read_json(root / "release-manifest.json")
    splits = read_json(root / "splits.json")
    checks: dict[str, dict[str, object]] = {}

    def record(name: str, passed: bool, **details) -> None:
        checks[name] = {"passed": bool(passed), **details}
        log(f"{'PASS' if passed else 'FAIL'} {name} {json.dumps(details, default=str)[:300]}")

    log("original release.audit_release ...")
    qc = audit_release(root, manifest, splits)
    record("original_audit_release", qc["status"] == "pass", split_counts=qc["split_counts"], players_active_at_attempt_20={r: qc["regimes"][r]["players_active_at_attempt_20"] for r in qc["regimes"]})
    shipped_qc = read_json(root / "qc.json")
    identical = json.dumps(qc, sort_keys=True) == json.dumps(shipped_qc, sort_keys=True)
    float_differences = _json_differences(qc, shipped_qc)
    record("original_audit_matches_shipped_qc", identical or (float_differences["discrete_differences"] == 0 and float_differences["max_abs_float_diff"] <= 1e-9),
           identical_strings=identical, **float_differences)

    split_sets = {name: set(int(v) for v in splits[name]) for name in ("train", "validation", "test")}
    all_players = set.union(*split_sets.values())
    record("one_split_per_player", sum(len(s) for s in split_sets.values()) == len(all_players) and all_players == set(range(manifest["players"])),
           n_players=len(all_players), sizes={k: len(v) for k, v in split_sets.items()})

    level_by_name = {level.name: level for level in LEVELS}
    level_index = {level.name: i for i, level in enumerate(LEVELS)}
    tier_index = {name: i for i, name in enumerate(TIER_NAMES)}
    ids_by_regime: dict[str, set[int]] = {}
    summary: dict[str, object] = {}
    for regime in ("natural", "randomized"):
        log(f"{regime}: loading episodes and transitions")
        episodes, fields = _episodes(root, regime)
        arrays = _transitions(root, regime)
        n_rows = len(arrays["action_index"])
        record(f"{regime}/transition_schema_version_3", list(arrays["schema_version"].tolist()) == [3])
        record(f"{regime}/no_oracle_fields_in_logged_table", not (FORBIDDEN_LOGGED & set(fields)), fields=len(fields))
        oracle_in_logged = [a["path"] for s in manifest["regimes"][regime]["shards"] for a in read_json(root / s["manifest"])["logged_artifacts"].values() if "oracle" in Path(a["path"]).parts]
        record(f"{regime}/no_oracle_path_among_logged_artifacts", not oracle_in_logged)
        ids_by_regime[regime] = {k[0] for k in episodes}
        # attempts: contiguous, <= 20, churn absorbing (churn_after == 1 only on the last attempt; shorter histories end with churn)
        by_player: dict[int, list[tuple[int, int]]] = {}
        for (pid, aid), row in episodes.items():
            by_player.setdefault(pid, []).append((aid, int(row["churn_after"])))
        contiguous = absorbing = capped = True
        for pid, items in by_player.items():
            items.sort()
            aids = [a for a, _ in items]
            churn = [c for _, c in items]
            contiguous &= aids == list(range(1, len(aids) + 1))
            capped &= len(aids) <= manifest["max_attempts"]
            absorbing &= all(c == 0 for c in churn[:-1]) and (churn[-1] == 1 or len(aids) == manifest["max_attempts"])
        record(f"{regime}/attempts_contiguous_from_1", contiguous)
        record(f"{regime}/attempts_capped_at_max", capped, max_attempts=manifest["max_attempts"])
        record(f"{regime}/churn_absorbing", absorbing)
        record(f"{regime}/active_before_always_1", all(row["active_before"] == "1" for row in episodes.values()))
        # joins: every transition episode exists; steps per episode == moves_used; episodes with moves_used > 0 have transitions
        key = arrays["player_id"].astype(np.int64) * 1000 + arrays["attempt_id"].astype(np.int64)
        uniq, first, counts = np.unique(key, return_index=True, return_counts=True)
        logged_keys = {pid * 1000 + aid for pid, aid in episodes}
        record(f"{regime}/transition_episodes_exist_in_table", set(uniq.tolist()) <= logged_keys, transition_episodes=len(uniq), table_episodes=len(episodes))
        moves_used = {pid * 1000 + aid: int(row["moves_used"]) for (pid, aid), row in episodes.items()}
        steps_match = all(moves_used[int(k)] == int(c) for k, c in zip(uniq, counts))
        missing = [k for k, m in moves_used.items() if m > 0 and k not in set(uniq.tolist())]
        record(f"{regime}/steps_per_episode_equal_moves_used", steps_match and not missing, episodes_without_transitions=len(missing))
        # sorted by (player, attempt, step) with contiguous steps -> continuity checks on consecutive rows
        order = np.lexsort((arrays["step_id"], arrays["attempt_id"], arrays["player_id"]))
        a = {k: v[order] for k, v in arrays.items() if k != "schema_version"}
        same = (a["player_id"][1:] == a["player_id"][:-1]) & (a["attempt_id"][1:] == a["attempt_id"][:-1])
        record(f"{regime}/step_ids_contiguous", bool(np.all(a["step_id"][1:][same] == a["step_id"][:-1][same] + 1)) and bool(np.all(a["step_id"][1:][~same] == 0)) and int(a["step_id"][0]) == 0)
        record(f"{regime}/board_continuity_exact", bool(np.array_equal(a["board_after"][:-1][same], a["board_before"][1:][same])))
        record(f"{regime}/specials_continuity_exact", bool(np.array_equal(a["specials_after"][:-1][same], a["specials_before"][1:][same])))
        record(f"{regime}/moves_continuity_exact", bool(np.array_equal(a["moves_left_next"][:-1][same], a["moves_left"][1:][same])))
        record(f"{regime}/goals_continuity_exact", bool(np.array_equal(a["goals_left_next"][:-1][same], a["goals_left"][1:][same])))
        record(f"{regime}/moves_decrement_by_one", bool(np.all(a["moves_left"] - a["moves_left_next"] == 1)))
        record(f"{regime}/goals_nonincreasing_nonnegative", bool(np.all(a["goals_left_next"] <= a["goals_left"])) and bool(np.all(a["goals_left_next"] >= 0)))
        # episode-level consistency with the first / last row
        starts = np.flatnonzero(np.concatenate(([True], ~same)))
        ends = np.concatenate((starts[1:] - 1, [n_rows - 1]))
        first_ok = last_ok = goal_ok = e_ok = level_ok = tier_ok = colour_ok = True
        no_terminal_mid = True
        for s, e in zip(starts, ends):
            pid, aid = int(a["player_id"][s]), int(a["attempt_id"][s])
            row = episodes[(pid, aid)]
            first_ok &= int(a["moves_left"][s]) == int(row["move_budget"]) and int(a["goals_left"][s]) == int(row["served_goal_count"])
            won = int(a["goals_left_next"][e]) <= 0
            last_ok &= (int(row["R"]) == int(won)) and (won or int(a["moves_left_next"][e]) == 0 or int(row["reshuffles"]) > 0 or int(a["moves_left_next"][e]) >= 0)
            goal_ok &= int(row["goals_cleared"]) == int(a["goals_left"][s]) - int(a["goals_left_next"][e])
            e_ok &= np.float32(float(row["E"])) == a["served_difficulty"][s] and bool(np.all(a["served_difficulty"][s : e + 1] == a["served_difficulty"][s]))
            level_ok &= int(a["level"][s]) == level_index[row["level"]] and bool(np.all(a["level"][s : e + 1] == a["level"][s]))
            tier_ok &= int(a["tier"][s]) == tier_index[row["tier"]]
            no_terminal_mid &= bool(np.all(a["goals_left_next"][s:e] > 0)) and bool(np.all(a["moves_left_next"][s:e] > 0))
        record(f"{regime}/opening_counters_match_table", first_ok)
        record(f"{regime}/terminal_outcome_consistent", last_ok)
        record(f"{regime}/no_terminal_state_before_last_step", no_terminal_mid)
        record(f"{regime}/goals_cleared_matches_counters", goal_ok)
        record(f"{regime}/served_difficulty_matches_E_float32", e_ok)
        record(f"{regime}/level_tier_codes_match_table", level_ok and tier_ok)
        for name, level in level_by_name.items():
            rows_of_level = a["level"] == level_index[name]
            boards = np.concatenate((a["board_before"][rows_of_level], a["board_after"][rows_of_level]))
            colour_ok &= bool(np.all((boards >= 0) & (boards < level.n_colours)))
        record(f"{regime}/colour_support_per_level_no_empty", colour_ok)
        record(f"{regime}/special_codes_in_0_2", bool(np.all((a["specials_before"] >= 0) & (a["specials_before"] <= 2))) and bool(np.all((a["specials_after"] >= 0) & (a["specials_after"] <= 2))))
        record(f"{regime}/goal_colour_constant_default", bool(np.all(a["goal_colour"] == a["goal_colour"][0])), goal_colour=int(a["goal_colour"][0]))
        lengths = np.asarray([len(v) for v in by_player.values()])
        summary[regime] = {"players": len(by_player), "attempts": len(episodes), "transitions": n_rows, "players_with_attempt_20": int((lengths >= 20).sum()),
                           "churned_players": int(sum(items[-1][1] for items in by_player.values())), "mean_attempts": float(lengths.mean())}
    record("matched_player_ids_across_regimes", ids_by_regime["natural"] == ids_by_regime["randomized"] == all_players)
    passed = all(c["passed"] for c in checks.values())
    report = {"schema_version": 1, "status": "pass" if passed else "fail", "root": relpath(root), "release_identity": identity().name, "checks": checks, "summary": summary,
              "original_qc": qc, "seconds": time.perf_counter() - started, "note": "structural validity only; physical replay is a separate stage (match3_simulator.evaluation replay)"}
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("copy", "verify", "check"))
    parser.add_argument("--source", default=None, help="verified payload directory to copy from (copy only)")
    args = parser.parse_args(argv)
    assert_package_import()
    directory = stage_dir(STAGE)
    log = Logger(STAGE, args.command)
    if args.command == "copy":
        if args.source is None:
            parser.error("--source is required for copy")
        if not DATA.exists():
            copy_payload(Path(args.source))
            log(f"copied payload -> {relpath(DATA)}")
        else:
            log(f"{relpath(DATA)} exists; not copied")
    if args.command in ("copy", "verify"):
        record = verify_payload(DATA)
        write_json(directory / "verify_release.json", record)
        log(f"verified {record['files']} files / {record['bytes']} bytes, cross-checked {record['cross_checked_files']} -> {relpath(directory / 'verify_release.json')}")
        return
    report = check_release(DATA, log=log)
    write_json(directory / "check_release.json", report)
    log(f"check_release: {report['status']} ({sum(c['passed'] for c in report['checks'].values())}/{len(report['checks'])} checks passed, {report['seconds']:.0f}s)")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
