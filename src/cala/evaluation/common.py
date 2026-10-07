"""Shared helpers: package paths, hashing, atomic JSON writes, DONE markers, HISTORY.md appends, and anonymized environment records."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable

import match3_simulator
PACKAGE = Path(match3_simulator.__file__).resolve().parent
ROOT = Path(os.environ.get("CALA_WORKSPACE", str(Path.cwd()))).resolve()
SRC = PACKAGE
VENDOR = PACKAGE
OVERLAY = PACKAGE
from match3_simulator.experiments.paper import PACKAGE as CALA_PACKAGE
CONFIGS = CALA_PACKAGE / "configs"
RUNS = ROOT / "runs"
WORKSPACE = ROOT
DATA = Path(os.environ.get("CALA_DATA_ROOT", str(ROOT / "data" / "release"))).resolve()
HISTORY = ROOT / "HISTORY.md"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(json_ready(value), sort_keys=True, separators=(",", ":")).encode())


def json_ready(value: Any) -> Any:
    """Convert numpy scalars / arrays, paths, tuples and dataclasses into JSON-serialisable values (NaN -> None)."""
    import numpy as np

    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (frozenset, set)):
        return sorted(json_ready(item) for item in value)
    return value


def relpath(path: str | Path) -> str:
    """Path relative to the package root when possible (shipped text never carries absolute home paths)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name if path.is_absolute() else path.as_posix()


def write_json(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    """Atomic JSON write (partial file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial-{os.getpid()}")
    partial.write_text(json.dumps(json_ready(value), indent=indent, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial-{os.getpid()}")
    partial.write_text(text)
    os.replace(partial, path)
    return path


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def environment_record() -> dict[str, object]:
    """Python / package versions and hardware without hostname, user or home paths."""
    import numpy

    record: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": numpy.__version__,
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        record["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        record["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        record["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover - torch missing
        record["torch"] = None
    try:
        import pyro

        record["pyro"] = pyro.__version__
    except Exception:  # pragma: no cover
        record["pyro"] = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    record["memory_total_kib"] = int(line.split()[1])
                    break
    except OSError:
        pass
    return record


def assert_package_import() -> Path:
    """Check that ``match3_simulator`` imports from the installed package directory."""
    import match3_simulator

    location = Path(match3_simulator.__file__).resolve()
    if not str(location).startswith(str(PACKAGE.resolve())):
        raise RuntimeError(f"match3_simulator imports from {location}, expected a path under {PACKAGE}")
    return location


def set_accepted_calibration() -> Path:
    """Point the original quota-calibration loader at the accepted table (the release generator did the same) and return its path."""
    path = PACKAGE / "accepted_calibration.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    current = os.environ.get("MATCH3_CALIBRATION_PATH")
    if current is not None and Path(current).resolve() != path.resolve():
        raise RuntimeError(f"MATCH3_CALIBRATION_PATH already points elsewhere: {current}")
    os.environ["MATCH3_CALIBRATION_PATH"] = str(path.resolve())
    return path


def hash_tree(root: Path, *, exclude_dirs: Iterable[str] = (".git", "__pycache__", ".cache", "wandb"), exclude_suffixes: Iterable[str] = (".pyc",)) -> dict[str, str]:
    """Relative path -> sha256 of every file under root."""
    out: dict[str, str] = {}
    excluded = set(exclude_dirs)
    suffixes = tuple(exclude_suffixes)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if excluded & set(relative.parts[:-1]) or path.name.endswith(suffixes):
            continue
        out[relative.as_posix()] = sha256_file(path)
    return out


# --------------------------------------------------------------------------- stages ----


def stage_dir(stage: str) -> Path:
    path = RUNS / stage
    (path / "logs").mkdir(parents=True, exist_ok=True)
    return path


def write_done(stage: str, *, inputs: dict[str, str | Path], outputs: Iterable[str | Path], extra: dict[str, object] | None = None) -> Path:
    """Write runs/<stage>/DONE after re-hashing every output file; inputs are hashed too so drivers can detect stale results."""
    directory = stage_dir(stage)
    record = {
        "stage": stage,
        "finished_at": now(),
        "inputs": {name: (sha256_file(path) if Path(path).is_file() else str(path)) for name, path in inputs.items()},
        "outputs": {relpath(path): sha256_file(path) for path in outputs if Path(path).is_file()},
    }
    if extra:
        record["extra"] = json_ready(extra)
    return write_json(directory / "DONE", record)


def done_is_current(stage: str, inputs: dict[str, str | Path]) -> bool:
    """True when DONE exists, its recorded inputs hash to the same values and every recorded output still hashes identically."""
    path = RUNS / stage / "DONE"
    if not path.is_file():
        return False
    record = read_json(path)
    for name, value in inputs.items():
        expected = record["inputs"].get(name)
        actual = sha256_file(value) if Path(value).is_file() else str(value)
        if expected != actual:
            return False
    for relative, digest in record["outputs"].items():
        target = ROOT / relative
        if not target.is_file() or sha256_file(target) != digest:
            return False
    return True


def history(message: str) -> None:
    """Append one dated line to HISTORY.md."""
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a") as handle:
        handle.write(f"- {time.strftime('%Y-%m-%d %H:%M')} — {message}\n")


class Logger:
    """Print and tee into runs/<stage>/logs/<name>.log."""

    def __init__(self, stage: str, name: str | None = None):
        directory = stage_dir(stage)
        self.path = directory / "logs" / f"{name or stage}.log"

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a") as handle:
            handle.write(line + "\n")


__all__ = [name for name in dir() if not name.startswith("_")]
