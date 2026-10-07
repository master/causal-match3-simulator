"""Record checkpoint, split, nuisance and configuration hashes before test use."""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/release")
    p.add_argument(
        "--groups", nargs="+", choices=["main"], default=["main"]
    )
    p.add_argument("--out", default="runs/freeze.json")
    a = p.parse_args()
    from match3_simulator.experiments.paper import freeze

    freeze(a.out, root=a.root, groups=a.groups)
    print(a.out)


if __name__ == "__main__":
    main()
