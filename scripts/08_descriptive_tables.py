"""Produce Table 1, Tables 4--7, or Table 8 and its scaling plot from new runs."""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["benchmark", "data", "world-models"])
    p.add_argument("--root", default="data/release")
    p.add_argument("--out", default="runs/paper")
    a = p.parse_args()
    from match3_simulator.evaluation.descriptive import (
        benchmark_report,
        dataset_report,
        world_model_report,
    )

    if a.kind == "benchmark":
        benchmark_report("runs/reference/benchmark", a.out)
    elif a.kind == "data":
        dataset_report(a.root, a.out)
    else:
        world_model_report(a.out)


if __name__ == "__main__":
    main()
