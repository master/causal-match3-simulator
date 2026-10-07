"""Tables and one-row paper figures from new, identity-checked evaluation reports."""

from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from match3_simulator.experiments.paper import read, write, protocol
from match3_simulator.experiments.estimators import curve_metrics
from match3_simulator.experiments.bootstrap import whole_player_bootstrap

LEVELS = ("orchard", "harbour", "foundry")
ARMS = ("handcrafted", "det_z", "var_z")
NAMES = dict(
    naive="No history",
    handcrafted="Handcrafted history",
    det_z="Deterministic $Z$",
    var_z="Variational CaLA",
    oracle_k="Oracle $K$",
)
COLORS = dict(
    naive="#D55E00",
    handcrafted="#009E73",
    det_z="#0072B2",
    var_z="#CC79A7",
    oracle_k="#E69F00",
)


def moments(values):
    a = np.asarray(values, float)
    if len(a) != 3 or not np.isfinite(a).all():
        raise ValueError("Exactly three finite seed values are required")
    return float(a.mean()), float(a.std(ddof=1))


def pm(values):
    mean, sd = moments(values)
    return f"${mean:.3f} \\pm {sd:.3f}$"


def table(out, name, headers, rows, caption, label):
    """Write full text cells as CSV and a ready-to-include booktabs table."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / (name + ".csv")).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]
    (out / (name + ".tex")).write_text("\n".join(lines) + "\n")


def load_groups(root, group):
    groups = defaultdict(list)
    for path in sorted(Path(root).glob("*/report.json")):
        meta_path = path.parent / "paper_job.json"
        if not meta_path.exists():
            continue
        meta = read(meta_path)
        job = meta["job"]
        if job.get("policy") != "special" or job.get("arm") not in ARMS:
            continue
        which = "direct" if meta["route"] == "Direct-DR" else job["group"]
        if which != group:
            continue
        report = read(path)
        report["_dir"] = path.parent
        report["_job"] = job
        report["_freeze"] = meta["freeze_sha256"]
        key = (
            report["regime"],
            report["arm"],
            job.get("kernel", "direct"),
            job["policy"],
        )
        groups[key].append(report)
    if not groups:
        raise FileNotFoundError(f"No completed {group} evaluations in {root}")
    for key, reports in groups.items():
        reports.sort(key=lambda r: r["seed"])
        if [r["seed"] for r in reports] != protocol()["seeds"]:
            raise ValueError(f"Missing or duplicated seeds: {key}")
        if (
            len({r["gold_sha256"] for r in reports}) != 1
            or len({r["_freeze"] for r in reports}) != 1
        ):
            raise ValueError(f"Mixed reference or freeze identities: {key}")
        first = reports[0]
        for r in reports:
            if r["gold_replicates"] != 128 or r["grid"] != first["grid"]:
                raise ValueError("Reference budget/grid mismatch")
            for level in LEVELS:
                if (
                    r["levels"][level]["gold_m_do"]
                    != first["levels"][level]["gold_m_do"]
                ):
                    raise ValueError("References differ across seeds")
    expected = 24 if group == "capacity" else 10
    if len(groups) != expected:
        raise ValueError(
            f"{group} expects {expected} configurations with 3 seeds each; found {len(groups)}"
        )
    for regime in ("natural", "randomized"):
        rows = [r for k, v in groups.items() if k[0] == regime for r in v]
        if len({r["gold_sha256"] for r in rows}) != 1:
            raise ValueError("Arms must use the same reference within a regime")
    return groups


def joint_direct(reports, level):
    """Average aligned seed contributions, then resample the same players jointly.

    The engine population curve stays fixed, as in the archived estimator.
    This is not an average of three confidence intervals.
    """
    values, supports, reference_ids = [], [], None
    for report in reports:
        with np.load(report["_dir"] / "analysis_arrays.npz") as a:
            rows = a[f"{level}__rows"]
            ids = a["player_ids"][rows]
            order = np.argsort(ids)
            if reference_ids is None:
                reference_ids = ids[order]
            if not np.array_equal(reference_ids, ids[order]):
                raise ValueError("Seed populations do not match")
            values.append(a[f"{level}__Direct-DR__terms"][order])
            supports.append(float((a["g_grid"][rows] > 0.02).mean()))
    terms = np.mean(values, axis=0)
    gold = np.asarray(reports[0]["levels"][level]["gold_m_do"])
    grid = np.asarray(reports[0]["grid"])
    metrics = curve_metrics(terms.mean(0), gold, grid)
    boot = whole_player_bootstrap(terms, gold, grid, n_bootstrap=2000, seed=77)
    return metrics, boot, float(np.mean(supports))


def decision_tables(groups, out, group):
    aggregate, per_level, direct_rows = [], [], []
    keys = sorted(groups, key=lambda k: (k[0] != "natural", ARMS.index(k[1]), k[2]))
    for key in keys:
        regime, arm, kernel, policy = key
        reps = groups[key]
        routes = ["Direct-DR"] if group == "direct" else ["WM-MC", "WM-DR"]
        row = [regime.title(), NAMES[arm]]
        if group == "capacity":
            row += [kernel]
        for route in routes:
            row += [
                pm([r["aggregate"][route][m] for r in reps])
                for m in ("weighted_curve_mae", "mean_population_regret")
            ]
        row += [
            pm([r["aggregate"][routes[0]]["pooled_individual_regret"] for r in reps])
        ]
        aggregate.append(row)
        for level in LEVELS:
            if group == "direct" and regime == "natural":
                met, boot, support = joint_direct(reps, level)
                lo, hi = boot["regret_lower"], boot["regret_upper"]
                direct_rows.append(
                    [
                        level.title(),
                        NAMES[arm],
                        f"{met['integrated_abs_error']:.3f}",
                        f"${met['selected_e']:+.2f}$",
                        f"${met['engine_optimum_e']:+.2f}$",
                        f"{met['regret']:.3f}",
                        f"${(lo+hi)/2:.3f} \\pm {(hi-lo)/2:.3f}$",
                        f"{100*boot['p_engine_optimum']:.1f}\\%",
                        f"{100*support:.1f}\\%",
                    ]
                )
            elif group != "direct":
                for route in routes:
                    vals = [r["levels"][level]["routes"][route] for r in reps]
                    per_level.append(
                        [
                            regime.title(),
                            level.title(),
                            NAMES[arm],
                            kernel,
                            route,
                            pm([v["integrated_abs_error"] for v in vals]),
                            "$("
                            + ",".join(f"{v['selected_e']:+.2f}" for v in vals)
                            + ")$",
                            f"${vals[0]['engine_optimum_e']:+.2f}$",
                            pm([v["regret"] for v in vals]),
                        ]
                    )
    headers = ["Regime", "Adjustment arm"] + (
        ["World model"] if group == "capacity" else []
    )
    if group == "direct":
        headers += [
            "Weighted curve MAE",
            "Mean population regret",
            "Pooled individual regret",
        ]
        table(
            out,
            "table09_direct_aggregate",
            headers,
            aggregate,
            "Aggregate Direct-DR performance (mean $\\pm$ sample SD across three seeds).",
            "tab:direct-dr-aggregate",
        )
        table(
            out,
            "table10_direct_natural_level",
            [
                "Level",
                "Adjustment arm",
                "Curve MAE",
                "Selected difficulty",
                "Engine optimum",
                "Point regret",
                "Bootstrap regret (95\\% range)",
                "Selection probability",
                "Support",
            ],
            sorted(direct_rows, key=lambda r: LEVELS.index(r[0].lower())),
            "Natural-assignment Direct-DR performance. Point estimates use the mean curve; ranges use 2,000 joint player resamples.",
            "tab:direct-dr-natural-level",
        )
    else:
        headers += [
            "WM-MC curve MAE",
            "WM-MC population regret",
            "WM-DR curve MAE",
            "WM-DR population regret",
            "Individual regret",
        ]
        table(
            out,
            "table03_wm_aggregate" if group == "main" else "capacity_all_routes",
            headers,
            aggregate,
            "World-model performance (mean $\\pm$ sample SD across three seeds). Individual regret is shared by WM-MC and WM-DR.",
            "tab:wm-aggregate" if group == "main" else "tab:capacity-all-routes",
        )
        h = [
            "Regime",
            "Level",
            "Adjustment arm",
            "World model",
            "Route",
            "Curve MAE",
            "Selected difficulty",
            "Engine optimum",
            "Regret",
        ]
        if group == "main":
            for route, num in [("WM-MC", 11), ("WM-DR", 12)]:
                rows = [
                    r
                    for r in per_level
                    if r[4] == route and (route == "WM-DR" or r[0] == "Natural")
                ]
                table(
                    out,
                    f"table{num:02d}_{route.lower()}_level",
                    h,
                    rows,
                    f"{route} population performance (mean $\\pm$ sample SD across three seeds).",
                    f"tab:{route.lower()}-level",
                )
        else:
            rows = [
                [r[0], r[2], r[3], r[4], r[-1]]
                for r in aggregate
                if r[1] == NAMES["var_z"]
            ]
            table(
                out,
                "table13_capacity",
                [
                    "Regime",
                    "World model",
                    "Curve MAE",
                    "Population regret",
                    "Individual regret",
                ],
                rows,
                "WM-MC performance with variational CaLA across world models (mean $\\pm$ sample SD across three seeds).",
                "tab:cala-wm-capacity",
            )
    write(
        Path(out) / f"{group}_provenance.json",
        {
            str(k): [
                {
                    "seed": r["seed"],
                    "gold_sha256": r["gold_sha256"],
                    "freeze_sha256": r["_freeze"],
                    "policy": k[-1],
                    "source": str(r["_dir"]),
                }
                for r in v
            ]
            for k, v in groups.items()
        },
    )


def curves(groups, out, *, route, capacity=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), sharey=True)
    kernels = sorted({k[2] for k in groups})
    palette = dict(zip(kernels, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
    for ax, level in zip(axes, LEVELS):
        reference_seen = set()
        for (regime, arm, kernel, _), reps in groups.items():
            if capacity and arm != "var_z":
                continue
            grid = np.array(reps[0]["grid"])
            values = np.array(
                [r["levels"][level]["routes"][route]["curve"] for r in reps]
            )
            mean, sd = values.mean(0), values.std(0, ddof=1)
            style = "-" if regime == "randomized" else "--"
            color = palette[kernel] if capacity else COLORS[arm]
            ax.plot(grid, mean, style, color=color, lw=1.5)
            ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.08)
            if regime not in reference_seen:
                ax.plot(
                    grid,
                    reps[0]["levels"][level]["gold_m_do"],
                    style,
                    color="black",
                    lw=1.8,
                )
                reference_seen.add(regime)
        ax.set_title(level.title())
        ax.set_xlabel("Difficulty")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Churn probability")
    items = [("Engine reference", "black")]
    items += (
        list(palette.items()) if capacity else [(NAMES[a], COLORS[a]) for a in ARMS]
    )
    handles = [Line2D([0], [0], color=color, label=name) for name, color in items]
    handles += [
        Line2D([0], [0], color="gray", ls=ls, label=name)
        for name, ls in [("Randomized", "-"), ("Natural", "--")]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def individual_plot(groups, out, *, route, capacity=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    arms = ("naive", "var_z", "oracle_k") if capacity else ARMS
    kernels = sorted({k[2] for k in groups})
    for ax, regime in zip(axes, ("natural", "randomized")):
        for j, kernel in enumerate(kernels):
            means, sds = [], []
            for arm in arms:
                matches = [
                    v for k, v in groups.items() if k[:3] == (regime, arm, kernel)
                ]
                if len(matches) != 1:
                    raise ValueError("Ambiguous model/arm group")
                m, s = moments(
                    [
                        r["aggregate"][route]["pooled_individual_regret"]
                        for r in matches[0]
                    ]
                )
                means.append(m)
                sds.append(s)
            offset = (j - (len(kernels) - 1) / 2) * 0.16
            ax.errorbar(
                np.arange(len(arms)) + offset,
                means,
                yerr=sds,
                fmt="o",
                capsize=3,
                label=kernel,
            )
        ax.set_title(regime.title())
        ax.set_xticks(
            range(len(arms)), [NAMES[a] for a in arms], rotation=20, ha="right"
        )
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pooled individual regret")
    if capacity:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9
        )
    fig.tight_layout(rect=(0, 0, 1, 0.9 if capacity else 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
