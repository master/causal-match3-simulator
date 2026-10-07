"""
This is the CaLA artifact: ``run_report.json`` per arm run and per evaluated cell — status ``cala_causal_run``, source / simulator / payload commits, the
four release hashes, split hash, full config, environment, parameter counts, compute, validation histories, checkpoint, per-player arrays
(by file + hash), support diagnostics, uncertainty summaries, oracle fields accessed, unlock token. Written to ``.partial`` -> validated -> atomic rename;
never overwrites.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from match3_simulator.world_modeling.wm1_verify import provenance, sha256_file

STATUS = "cala_causal_run"
STATUSES = (STATUS, "fire_test")
REQUIRED_RUN = ("status", "kind", "label", "arm", "regime", "seed", "provenance", "config", "arm_config", "parameters",
                "compute", "history", "checkpoints", "policy_metrics", "retention_calibration", "temperature",
                "propensity", "oracle_fields_accessed", "created_at")
REQUIRED_CELL = ("status", "kind", "label", "arm", "regime", "seed", "provenance", "spec", "arrays", "report", "unlock_token", "oracle_fields_accessed", "created_at")


class ArtifactError(RuntimeError):
    """Schema or hash validation failed."""


def _dataset_provenance(dataset_root: str, regime: str) -> dict[str, object]:
    prov = provenance(dataset_root)
    prov["dataset"]["regime"] = regime
    prov.setdefault("workspace", {})["cala_note"] = "CaLA stage; kernels from runs/wm1 (frozen)"
    return prov


def assemble_run(run_dir: Path) -> dict[str, object]:
    training = json.loads((run_dir / "training.json").read_text())
    best = run_dir / "best.pt"
    if not best.exists():
        raise ArtifactError("best.pt missing")
    import torch

    meta = torch.load(best, map_location="cpu", weights_only=False)
    return {"status": training.get("status", STATUS), "kind": "arm_run", "schema_version": 1, "label": training["label"], "arm": training["arm"], "regime": training["regime"], "seed": training["seed"], "kernel_regime": training["kernel_regime"],
            "provenance": _dataset_provenance(training["config"]["dataset_root"], training["regime"]), "config": training["config"], "arm_config": training["arm_config"], "context_width": training["context_width"],
            "parameters": training["parameters"], "data": training["data"], "compute": training["compute"], "selection": training["selection"], "history": training["history"],
            "checkpoints": {"best": {"path": "best.pt", "sha256": sha256_file(best), "update": int(meta["update"]), "bytes": best.stat().st_size}},
            "policy_metrics": training["policy_metrics"], "retention_calibration": training["retention_calibration"], "temperature": training["temperature"], "propensity": training["propensity"],
            "oracle_fields_accessed": training["oracle_fields_accessed"], "features": training.get("features"), "logs": sorted(p.name for p in run_dir.glob("*.log")), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def validate_run(report: dict[str, object], run_dir: Path) -> None:
    missing = [k for k in REQUIRED_RUN if k not in report]
    if missing:
        raise ArtifactError(f"run report lacks {missing}")
    if report["status"] not in STATUSES:
        raise ArtifactError(f"status must be one of {STATUSES}")
    if sha256_file(run_dir / "best.pt") != report["checkpoints"]["best"]["sha256"]:
        raise ArtifactError("checkpoint hash changed")
    dataset = report["provenance"]["dataset"]
    for key in ("payload_commit", "release_manifest_sha256", "splits_sha256", "accepted_benchmark_sha256", "qc_sha256"):
        if not dataset.get(key):
            raise ArtifactError(f"provenance lacks {key}")
    if report["arm"] in ("naive", "handcrafted", "det_z", "var_z") and report["oracle_fields_accessed"]["n_accesses"]:
        raise ArtifactError("a deployable arm accessed oracle fields")
    json.dumps(report, allow_nan=False)


def finalize_run(run_dir: str | Path, *, progress=None) -> Path:
    """Write run_report.json into the .partial arm run directory and rename it (never overwriting)."""
    emit = progress or (lambda _: None)
    run = Path(run_dir)
    if not run.name.endswith(".partial"):
        raise ArtifactError(f"{run} is not a .partial run directory")
    final = run.with_name(run.name[: -len(".partial")])
    if final.exists():
        raise ArtifactError(f"refusing to overwrite {final}")
    report = assemble_run(run)
    validate_run(report, run)
    tmp = run / "run_report.json.partial"
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False, default=float) + "\n")
    tmp.replace(run / "run_report.json")
    os.rename(run, final)
    emit(f"finalized {final}")
    return final


def write_cell_report(cell_dir: str | Path, *, dataset_root: str = "data/release", progress=None) -> Path:
    """run_report.json for an estimated cell (arrays + estimate report + unlock token). Skips when present."""
    emit = progress or (lambda _: None)
    cell = Path(cell_dir)
    path = cell / "run_report.json"
    if path.exists():
        return path
    meta = json.loads((cell / "cell.json").read_text())
    report = json.loads((cell / "report.json").read_text())
    arrays = {name: {"path": name, "sha256": sha256_file(cell / name), "bytes": (cell / name).stat().st_size} for name in ("arrays.npz", "predictions.npz", "estimate.npz") if (cell / name).exists()}
    training = json.loads((Path(meta["spec"]["arm_run"]) / "training.json").read_text())
    out = {"status": training.get("status", STATUS), "kind": "evaluated_cell", "schema_version": 1, "label": meta["label"], "arm": meta["arm"], "regime": meta["spec"]["regime"], "seed": meta["spec"]["seed"], "kernel": report["kernel"],
           "provenance": _dataset_provenance(dataset_root, meta["spec"]["regime"]), "spec": meta["spec"], "arm_run": {"path": meta["spec"]["arm_run"], "checkpoint_sha256": meta["arm_checkpoint_sha256"], "label": training["label"]},
           "compute": {"seconds": meta["seconds"], "seconds_per_candidate": meta["seconds_per_candidate"], "rows_per_candidate": meta["rows_per_candidate"], "n_actions": meta["n_actions"], "illegal": meta["illegal"]},
           "arrays": arrays, "report": {k: v for k, v in report.items() if k != "levels"}, "levels": {name: {k: v for k, v in lv.items() if k != "routes"} | {"routes": {r: {k: v for k, v in rv.items() if k not in ("individual", "bootstrap")} | {"bootstrap_summary": None if not rv.get("bootstrap") else {k: rv["bootstrap"][k] for k in ("regret_lower", "regret_upper", "p_engine_optimum")}} for r, rv in lv["routes"].items()}} for name, lv in report["levels"].items()},
           "support": {name: lv["routes"]["WM-DR"]["support"] for name, lv in report["levels"].items()}, "unlock_token": report["unlock_token"],
           "oracle_fields_accessed": {"cell_accesses": meta["oracle_accesses"], "arm_training": training["oracle_fields_accessed"]}, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    missing = [k for k in REQUIRED_CELL if k not in out]
    if missing:
        raise ArtifactError(f"cell report lacks {missing}")
    for block in arrays.values():
        if sha256_file(cell / block["path"]) != block["sha256"]:
            raise ArtifactError("array hash changed")
    tmp = cell / "run_report.json.partial"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False, default=float) + "\n")
    tmp.replace(path)
    emit(f"cell report {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    finalize_run(args.run, progress=print)


if __name__ == "__main__":
    main()


__all__ = ["STATUS", "ArtifactError", "assemble_run", "finalize_run", "validate_run", "write_cell_report"]
