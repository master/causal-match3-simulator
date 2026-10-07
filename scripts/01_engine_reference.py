"""Generate the cohort benchmark or the separate logged-player reference."""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["benchmark", "logged"])
    p.add_argument("--stage", choices=["equivalence", "calibrate", "validate", "audit"])
    p.add_argument("--regime", choices=["natural", "randomized"])
    p.add_argument("--split", choices=["validation", "test"])
    p.add_argument("--root", default="data/release")
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    from match3_simulator.evaluation.reference import benchmark, logged

    if a.kind == "benchmark":
        if a.stage is None:
            p.error("benchmark needs --stage")
        benchmark(a.stage, workers=a.workers)
    else:
        if not a.regime or not a.split:
            p.error("logged needs --regime and --split")
        logged(root=a.root, regime=a.regime, split=a.split, workers=a.workers)


if __name__ == "__main__":
    main()
