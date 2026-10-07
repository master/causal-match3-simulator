"""Generate decision tables and figures after all three seeds have completed."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=["main"], required=True)
    p.add_argument("--root", default="runs/evaluation")
    p.add_argument("--out", default="runs/paper")
    a = p.parse_args()
    from match3_simulator.evaluation.reporting import (
        load_groups,
        decision_tables,
        curves,
        individual_plot,
    )

    groups = load_groups(a.root, a.group)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    decision_tables(groups, out, a.group)
    if a.group == "main":
        curves(groups, out / "figure02_wm_mc.pdf", route="WM-MC")
        curves(groups, out / "figure10_wm_dr.pdf", route="WM-DR")
        individual_plot(groups, out / "figure09_individual_wm.pdf", route="WM-MC")
    elif a.group == "direct":
        curves(groups, out / "figure07_direct_dr.pdf", route="Direct-DR")
        individual_plot(
            groups, out / "figure08_individual_direct.pdf", route="Direct-DR"
        )
    else:
        curves(
            groups, out / "figure11_capacity_curves.pdf", route="WM-MC", capacity=True
        )
        individual_plot(
            groups, out / "figure12_capacity_regret.pdf", route="WM-MC", capacity=True
        )
    print(out)


if __name__ == "__main__":
    main()
