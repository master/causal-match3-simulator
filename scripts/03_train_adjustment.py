"""Train one frozen paper adjustment configuration."""

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=["main"], default="main")
    p.add_argument("--job", type=int)
    p.add_argument("--list", action="store_true")
    p.add_argument("--root", default="data/release")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from match3_simulator.experiments.paper import training_jobs, train_job

    jobs = training_jobs(a.group)
    if a.list:
        print(json.dumps([dict(job=i, **j) for i, j in enumerate(jobs)], indent=2))
    elif a.job is None or not 0 <= a.job < len(jobs):
        p.error("choose a valid --job from --list")
    else:
        print(train_job(jobs[a.job], root=a.root, device=a.device))


if __name__ == "__main__":
    main()
