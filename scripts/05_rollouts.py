"""Generate 128 imagined trajectories per player and candidate difficulty."""

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=["main"], default="main")
    p.add_argument("--job", type=int)
    p.add_argument("--list", action="store_true")
    p.add_argument("--freeze", default="runs/freeze.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    a = p.parse_args()
    from match3_simulator.experiments.paper import rollout_jobs, rollout

    jobs = rollout_jobs(a.group)
    if a.list:
        print(json.dumps([dict(job=i, **j) for i, j in enumerate(jobs)], indent=2))
    elif a.job is None or not 0 <= a.job < len(jobs):
        p.error("choose a valid --job from --list")
    else:
        print(
            rollout(jobs[a.job], freeze_path=a.freeze, device=a.device, chunk=a.chunk)
        )


if __name__ == "__main__":
    main()
