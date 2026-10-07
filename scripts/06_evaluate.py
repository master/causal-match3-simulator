"""Evaluate saved WM arrays or fit Direct-DR; no WM rollout is repeated here."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", choices=["WM"], default="WM")
    p.add_argument("--group", choices=["main"], default="main")
    p.add_argument("--job", type=int)
    p.add_argument("--list", action="store_true")
    p.add_argument("--freeze", default="runs/freeze.json")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    from match3_simulator.experiments.paper import (
        training_jobs,
        rollout_jobs,
        arm_path,
        verify_freeze,
        job_key,
        read,
        write,
        sha,
    )
    from match3_simulator.evaluation.estimators import evaluate

    jobs = training_jobs("current") if a.route == "Direct-DR" else rollout_jobs(a.group)
    if a.list:
        print(json.dumps([dict(job=i, **j) for i, j in enumerate(jobs)], indent=2))
        return
    if a.job is None or not 0 <= a.job < len(jobs):
        p.error("choose a valid --job from --list")
    frozen = verify_freeze(a.freeze)
    job = jobs[a.job]
    run = arm_path(job["policy"], job["regime"], job["arm"], job["seed"])
    if str(run / "best.pt") not in frozen["assets"]:
        raise ValueError("Unfrozen arm")
    gold = Path("runs/reference/logged") / job["regime"] / "test/gold.json"
    if read(gold)["replicates"] != 128:
        raise ValueError("Paper evaluation requires the logged-state R128 reference")
    cell = None
    if a.route == "WM":
        base = (
            Path("runs/cells")
            / job["group"]
            / job["regime"]
            / f"{job['policy']}-{job['arm']}-{job['seed']}"
        )
        matches = [
            f.parent
            for f in base.rglob("paper_job.json")
            if job_key(read(f)["job"]) == job_key(job)
            and read(f)["freeze_sha256"] == sha(a.freeze)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Need exactly one frozen prediction cell, found {len(matches)}"
            )
        cell = matches[0]
        key = job_key(job)
    else:
        key = f"direct__{job['regime']}__{job['arm']}__{job['seed']}"
    out = Path("runs/evaluation") / key
    evaluate(
        route=a.route,
        out=out,
        gold_path=gold,
        arm_run=run,
        regime=job["regime"],
        seed=job["seed"],
        root=frozen["dataset_root"],
        cell=cell,
        device=a.device,
    )
    write(
        out / "paper_job.json",
        {"job": job, "route": a.route, "freeze_sha256": sha(a.freeze)},
    )
    print(out)


if __name__ == "__main__":
    main()
