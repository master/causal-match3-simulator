"""Registered releases (``configs/release_registry.json``) -> ``ReleaseIdentity`` objects consumed by the overlay loaders.

The overlay's ``wm1_verify.verify`` / ``wm1_data.load_splits`` / ``wm1_data.EpisodeStore`` take a ``ReleaseIdentity``; every check stays a hard
failure. Registering a new release means adding an entry here (with its hashes) — never bypassing verification.
"""

from __future__ import annotations

from match3_simulator.evaluation.common import CONFIGS, ROOT, read_json

REGISTRY_PATH = CONFIGS / "release_registry.json"


def registry() -> dict[str, object]:
    return read_json(REGISTRY_PATH)


def identity(name: str | None = None):
    """Build the overlay ``ReleaseIdentity`` for a registered release (default: the registry's default entry)."""
    from match3_simulator.world_modeling.wm1_verify import ReleaseIdentity

    table = registry()
    key = name or table["default"]
    entry = table["releases"][key]
    return ReleaseIdentity(name=entry["name"], payload_commit=entry["payload_commit"], files=int(entry["files"]), bytes=int(entry["bytes"]), hashes=dict(entry["hashes"]), source=entry["source"])


def entry(name: str | None = None) -> dict[str, object]:
    table = registry()
    return table["releases"][name or table["default"]]


def local_root(name: str | None = None):
    return ROOT / entry(name)["local_root"]


__all__ = ["REGISTRY_PATH", "entry", "identity", "local_root", "registry"]
