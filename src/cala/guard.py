from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np

PHASES: tuple[str, ...] = ("train", "select", "evaluate", "correct")
DEPLOYABLE_ARMS: frozenset[str] = frozenset({"naive", "handcrafted", "det_z", "var_z"})

_phase: contextvars.ContextVar[str] = contextvars.ContextVar("cala_phase", default="train")
_oracle_token: contextvars.ContextVar["EvaluationOnly | None"] = contextvars.ContextVar("cala_oracle_token", default=None)


class GuardViolation(RuntimeError):
    """A CaLA stop condition was hit (oracle access by a deployable arm, test outcome read before serialisation, wrong phase)."""


def current_phase() -> str:
    return _phase.get()


@contextmanager
def phase(name: str):
    """Run a block in a named phase. Inputs: phase name (one of PHASES)."""
    if name not in PHASES:
        raise ValueError(f"unknown phase {name!r}; expected one of {PHASES}")
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)


@dataclass
class EvaluationOnly:
    """Token authorising oracle re-derivation for a named consumer (an arm label or a diagnostic). Records every access."""

    consumer: str
    accesses: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.consumer in DEPLOYABLE_ARMS:
            raise GuardViolation(f"deployable arm {self.consumer!r} may not hold an oracle token")

    def __enter__(self) -> "EvaluationOnly":
        self._reset = _oracle_token.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _oracle_token.reset(self._reset)


def oracle_player_skill(player_id: int, *, field_name: str = "K") -> tuple[float, ...]:
    """Guarded re-derivation of a player's true skill from the simulator (never the dataset).

    Inputs: player id; oracle field name (for the access log). Outputs: 4-tuple. Raises GuardViolation without an active EvaluationOnly token.
    """
    token = _oracle_token.get()
    if token is None:
        raise GuardViolation(f"oracle field {field_name} requested for player {player_id} without an EvaluationOnly token")
    from match3_simulator.world_modeling.cala_support import player_skill

    token.accesses.append({"player_id": int(player_id), "field": field_name, "phase": current_phase()})
    return player_skill(int(player_id))


def oracle_skills(player_ids, *, field_name: str = "K") -> np.ndarray:
    """Inputs: player ids. Outputs: (N, 4) float64 array of true skills (each access logged on the active token)."""
    return np.asarray([oracle_player_skill(int(pid), field_name=field_name) for pid in player_ids], dtype=np.float64).reshape(-1, 4)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SerializedPredictions:
    """Token proving predictions were written before any test outcome is read. Inputs: path of the finalised predictions file."""

    path: str
    sha256: str

    @classmethod
    def of(cls, path: str | Path) -> "SerializedPredictions":
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            raise GuardViolation(f"predictions file {path} is missing or empty; serialise before unlocking test outcomes")
        return cls(path=str(path), sha256=sha256_file(path))


class TestOutcomeLock:
    """Attempt-20 outcome columns of the test players, readable only in the ``correct`` phase with a SerializedPredictions token."""

    def __init__(self, columns: dict[str, np.ndarray], *, player_ids: np.ndarray):
        self._columns = {k: np.asarray(v) for k, v in columns.items()}
        self.player_ids = np.asarray(player_ids, dtype=np.int64)
        self.unlocked_by: list[str] = []
        for value in self._columns.values():
            if value.shape[0] != len(self.player_ids):
                raise ValueError("locked columns must align with player_ids")

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._columns))

    def unlock(self, token: SerializedPredictions) -> "TestOutcomes":
        if current_phase() != "correct":
            raise GuardViolation(f"test outcomes requested in phase {current_phase()!r}; only the 'correct' phase may read them")
        if not isinstance(token, SerializedPredictions) or not token.sha256:
            raise GuardViolation("a SerializedPredictions token is required to read test outcomes")
        if sha256_file(token.path) != token.sha256:
            raise GuardViolation("predictions file changed after the token was issued")
        self.unlocked_by.append(token.sha256)
        return TestOutcomes({k: v.copy() for k, v in self._columns.items()}, player_ids=self.player_ids.copy(), token=token)

    def __getattr__(self, name: str):  # any other attribute access is a bug
        raise GuardViolation(f"test outcome column {name!r} is locked; call unlock(SerializedPredictions)")


@dataclass(frozen=True)
class TestOutcomes:
    """Unlocked attempt-20 outcomes of the test players (row order = player_ids)."""

    columns: dict[str, np.ndarray]
    player_ids: np.ndarray
    token: SerializedPredictions

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]


__all__ = ["DEPLOYABLE_ARMS", "EvaluationOnly", "GuardViolation", "PHASES", "SerializedPredictions", "TestOutcomeLock", "TestOutcomes", "current_phase", "oracle_player_skill",
           "oracle_skills", "phase", "sha256_file"]
