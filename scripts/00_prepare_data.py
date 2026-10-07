"""Verify the accepted local payload and build the training caches once."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/release")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    from match3_simulator.evaluation.dataset_checks import verify_payload, check_release
    from match3_simulator.experiments.paper import write
    from match3_simulator.world_modeling.wm1_data import build_cache
    from match3_simulator.experiments.data_base import (
        build_cache as build_contexts,
    )
    from match3_simulator.experiments.data import build_cache4

    root = Path(a.root)
    verify_payload(root)
    report = check_release(root)
    write("runs/data/release_audit.json", report)
    if report["status"] != "pass":
        raise RuntimeError(
            "Release structural audit failed; inspect runs/data/release_audit.json"
        )
    if not a.audit_only:
        build_cache(root, "runs/wm1/data/natural", workers=a.workers)
        for regime in ("natural", "randomized"):
            build_contexts(
                root, regime, "runs/data/schema1", device=a.device, progress=print
            )
            build_cache4(
                root, regime, "runs/data/schema2", device=a.device, progress=print
            )


if __name__ == "__main__":
    main()
