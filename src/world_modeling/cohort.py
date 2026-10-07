"""Simulate a cohort of logged player trajectories once and cache it on disk.

The cohort specification carries the complete data-generating configuration — assignment gains and sigmas,
the churn schedule, the mastery rule and the SHA-256 of the accepted benchmark it was resolved from — and all of it
enters the cache file name, so a cohort simulated under the simulator's legacy defaults can never be mistaken for
an accepted-benchmark cohort. Per-player seeding is deterministic, so parallel simulation gives the same cohort
as serial simulation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Callable

from match3_simulator.calibrate import load_win_propensity_model
from match3_simulator.release import ACCEPTED_CALIBRATION_PATH, ACCEPTED_SPEC_PATH, load_accepted_spec, resolved_configs, sha256
from match3_simulator.retention import CHURN_SCHEDULE, ChurnSchedule, MasteryConfig, PlayerTrajectory, simulate_player_trajectory
from match3_simulator.scm import DDA_GAINS, E_SIGMAS
from match3_simulator.spec import BENCHMARK_CONFIG

REGIMES: tuple[str, ...] = ("natural", "randomized")


def use_accepted_calibration(path: str | Path = ACCEPTED_CALIBRATION_PATH) -> str:
    """Point every quota lookup (goal_count_for_E) at the accepted calibration table, as release.py does.

    Inputs: calibration file (default the accepted one shipped with the package).
    Outputs: its SHA-256. Raises when MATCH3_CALIBRATION_PATH already names a different file.
    """
    resolved = str(Path(path).resolve())
    current = os.environ.get("MATCH3_CALIBRATION_PATH")
    if current is not None and str(Path(current).resolve()) != resolved:
        raise RuntimeError(f"MATCH3_CALIBRATION_PATH is already set to {current}; refusing to switch calibration tables mid-process")
    os.environ["MATCH3_CALIBRATION_PATH"] = resolved
    return sha256(resolved)


@dataclass(frozen=True)
class CohortSpec:
    """Everything that determines a simulated cohort; every field enters the cache name."""

    n_players: int
    seed: int
    max_attempts: int
    dda_gains: tuple[float, ...] = DDA_GAINS
    e_sigmas: tuple[float, ...] = E_SIGMAS
    regime: str = "natural"
    churn_schedule: dict | None = None
    mastery: dict | None = None
    accepted_spec_sha256: str | None = None
    calibration_sha256: str | None = None
    player_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        if self.player_ids is not None and len(self.player_ids) != self.n_players:
            raise ValueError("player_ids must list exactly n_players ids")

    @classmethod
    def from_accepted(cls, *, n_players: int, seed: int, max_attempts: int = BENCHMARK_CONFIG.landmark_attempt, regime: str = "natural",
                      path: str | Path = ACCEPTED_SPEC_PATH, player_ids: tuple[int, ...] | None = None, dda_gains: tuple[float, ...] | None = None) -> "CohortSpec":
        """Resolve churn, mastery and assignment from the accepted benchmark for one regime and pin the accepted quota calibration.

        dda_gains, when given, replaces the accepted assignment gains (confounding-strength ablation); the cache name changes with it because it hashes the full spec."""
        spec = load_accepted_spec(path)
        _, churn, mastery, gains, sigmas = resolved_configs(spec, regime)
        if dda_gains is not None:
            gains = tuple(float(g) for g in dda_gains)
        calibration_sha = use_accepted_calibration()
        return cls(n_players=n_players, seed=seed, max_attempts=max_attempts, dda_gains=tuple(gains), e_sigmas=tuple(sigmas), regime=regime,
                   churn_schedule=asdict(churn), mastery=asdict(mastery), accepted_spec_sha256=sha256(path), calibration_sha256=calibration_sha, player_ids=player_ids)

    @property
    def legacy(self) -> bool:
        return self.accepted_spec_sha256 is None

    def churn(self) -> ChurnSchedule:
        return CHURN_SCHEDULE if self.churn_schedule is None else ChurnSchedule(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in self.churn_schedule.items()})

    def mastery_config(self) -> MasteryConfig:
        return MasteryConfig() if self.mastery is None else MasteryConfig(**self.mastery)

    def ids(self) -> list[int]:
        return list(self.player_ids) if self.player_ids is not None else list(range(self.n_players))

    def cache_name(self) -> str:
        """Derive the cache file name from the full specification (assignment, churn, mastery, spec and calibration hashes, player ids)."""
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]
        source = "legacy" if self.legacy else f"accepted-{self.accepted_spec_sha256[:8]}"
        return f"trajectories-{self.regime}-{source}-n{self.n_players}-seed{self.seed}-a{self.max_attempts}-{digest}.pkl"


def _simulate_one(payload: tuple[CohortSpec, int]) -> PlayerTrajectory:
    """Simulate one player of the cohort under the cohort's full configuration."""
    spec, player_id = payload
    if spec.calibration_sha256 is not None:
        use_accepted_calibration()
    return simulate_player_trajectory(
        load_win_propensity_model(), player_id=player_id, seed=spec.seed, max_attempts=spec.max_attempts, benchmark=BENCHMARK_CONFIG,
        churn_config=spec.churn(), mastery_config=spec.mastery_config(), dda_gains=spec.dda_gains, e_sigmas=spec.e_sigmas,
    )


def load_or_simulate_cohort(
    spec: CohortSpec,
    *,
    cache_dir: str | Path | None,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[PlayerTrajectory], dict[str, object]]:
    """Return the cohort from the cache or simulate and cache it.

    Inputs: cohort specification; cache directory or None; worker processes; optional progress callback.
    Outputs: the list of trajectories and a provenance record (including the full specification).
    """
    emit = progress or (lambda _: None)
    if spec.calibration_sha256 is not None:
        use_accepted_calibration()
    path = Path(cache_dir) / spec.cache_name() if cache_dir is not None else None
    if path is not None and path.exists():
        with path.open("rb") as handle:
            trajectories = pickle.load(handle)
        emit(f"loaded {len(trajectories)} cached trajectories from {path}")
        return trajectories, {"cache_path": str(path), "cache_hit": True, "spec": asdict(spec)}
    started = time.perf_counter()
    tasks = [(spec, player_id) for player_id in spec.ids()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trajectories = list(executor.map(_simulate_one, tasks, chunksize=max(1, len(tasks) // (4 * workers))))
    else:
        trajectories = [_simulate_one(task) for task in tasks]
    emit(f"simulated {spec.n_players} players ({'legacy defaults' if spec.legacy else 'accepted benchmark ' + spec.accepted_spec_sha256[:12]}, {spec.regime}) "
         f"in {time.perf_counter() - started:.1f}s with {workers} worker(s)")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # write to a sibling temp file and rename so concurrent readers never see a half-written cache
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        with temporary.open("wb") as handle:
            pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
        emit(f"cached trajectories at {path}")
    return trajectories, {"cache_path": None if path is None else str(path), "cache_hit": False, "simulation_seconds": time.perf_counter() - started, "workers": workers, "spec": asdict(spec)}


def release_dataset_players(root: str | Path, regime: str) -> dict[str, object]:
    """Read a release.py output (complete, or a <name>.partial with finished shards of the regime) and return its seed, max_attempts, player ids and splits.

    Inputs: dataset directory; regime natural | randomized.
    Outputs: dict with seed, max_attempts, player_ids (sorted tuple), splits {train, validation, test}, accepted_spec_sha256, partial flag.
    """
    root = Path(root)
    if (root / "release-manifest.json").exists():
        manifest = json.loads((root / "release-manifest.json").read_text())
        block = manifest["regimes"][regime]
        ids = tuple(sorted(pid for shard in block["shards"] for pid in range(int(shard["player_id_min"]), int(shard["player_id_max"]) + 1)))
        splits = json.loads((root / manifest["splits"]["path"]).read_text())
        return {"seed": int(manifest["seed"]), "max_attempts": int(manifest["max_attempts"]), "player_ids": ids, "splits": splits, "accepted_spec_sha256": manifest["accepted_benchmark"]["sha256"], "partial": False}
    shard_manifests = sorted((root / regime).glob("shard-*/manifest.json"))
    if not shard_manifests or not (root / "splits.json").exists():
        raise RuntimeError(f"{root} has neither release-manifest.json nor finished {regime} shards with splits.json")
    ids, seeds, attempts, shas = [], set(), set(), set()
    for path in shard_manifests:
        m = json.loads(path.read_text())
        ids.extend(int(p) for p in m["provenance"]["player_ids"]); seeds.add(int(m["configuration"]["seed"])); attempts.add(int(m["configuration"]["max_attempts"])); shas.add(m["provenance"]["accepted_benchmark_sha256"])
    if len(seeds) != 1 or len(attempts) != 1 or len(shas) != 1:
        raise RuntimeError("partial release shards disagree on seed, max_attempts or accepted benchmark")
    splits = json.loads((root / "splits.json").read_text())
    if len(ids) != len(splits["train"]) + len(splits["validation"]) + len(splits["test"]):
        raise RuntimeError(f"partial release: {regime} regime has {len(ids)} players in finished shards, splits expect more")
    return {"seed": seeds.pop(), "max_attempts": attempts.pop(), "player_ids": tuple(sorted(ids)), "splits": splits, "accepted_spec_sha256": shas.pop(), "partial": True}


__all__ = ["REGIMES", "CohortSpec", "load_or_simulate_cohort", "release_dataset_players", "use_accepted_calibration"]
