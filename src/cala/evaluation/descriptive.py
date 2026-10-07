"""Benchmark, dataset and world-model summaries; never contains paper values."""

from pathlib import Path
import numpy as np
from match3_simulator.experiments.paper import read, write, protocol
from match3_simulator.evaluation.reporting import table, pm, LEVELS


def benchmark_report(root, out):
    from match3_simulator.evaluation.reference_common import (
        accepted_context,
        load_surface_and_risk_set,
        curves_from_surface,
    )
    from match3_simulator.experiments.paper import PACKAGE

    cfg = read(PACKAGE / "configs/engine_reference.json")
    ctx = accepted_context("natural")
    full = {}
    for level in LEVELS:
        full[level] = []
        for seed in cfg["validation"]["seeds"]:
            path = Path(root) / f"seed-{seed}"
            risk, surface = load_surface_and_risk_set(path, level)
            full[level].append(
                curves_from_surface(
                    risk,
                    surface,
                    churn_level=ctx["churn"].for_level(level),
                    mastery=ctx["mastery"],
                    n_bootstrap=0,
                    bootstrap_seed=seed,
                    benchmark=ctx["benchmark"],
                )
            )

    def extent(xs, digits):
        lo, hi = min(xs), max(xs)
        return f"{lo:.{digits}f}" if lo == hi else f"{lo:.{digits}f}--{hi:.{digits}f}"

    rows = [
        [level.title()]
        + [
            extent([r["summary"][key] for r in full[level]], digits)
            for key, digits in [
                ("observational_recommendation", 2),
                ("interventional_optimum", 2),
                ("absolute_gap", 2),
                ("added_churn_regret", 3),
            ]
        ]
        for level in LEVELS
    ]
    table(
        out,
        "table01_benchmark",
        [
            "Level",
            "Observational recommendation",
            "Interventional optimum",
            "Recommendation gap",
            "Added churn",
        ],
        rows,
        "Observational and interventional difficulty recommendations. Ranges span five benchmark seeds.",
        "tab:heldout-summary",
    )
    write(Path(out) / "benchmark_curves.json", full)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharey=True)
    for ax, level in zip(axes, LEVELS):
        for key, color, label in [
            ("causal", "#0072B2", "Interventional"),
            ("observational", "#D55E00", "Observational"),
        ]:
            y = np.array([r[key] for r in full[level]])
            grid = full[level][0]["grid"]
            ax.plot(grid, y.mean(0), label=label, color=color)
            ax.fill_between(grid, y.min(0), y.max(0), alpha=0.12, color=color)
        ax.set_title(level.title())
        ax.set_xlabel("Difficulty")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Churn probability")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(out) / "figure01_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


def world_model_report(out):
    p = protocol()
    rows = []
    data = []
    for band in p["world_model"]["bands"]:
        for family, name in [
            ("transformer", "Transformer"),
            ("jepa", "LeJEPA"),
            ("lewm", "LeWM"),
        ]:
            metrics = []
            for seed in p["seeds"]:
                path = Path("runs/wm1") / f"{family}-{band}-{seed}"
                if family == "lewm":
                    path = path.parent / "variants" / (path.name + "-reencode")
                ev = read(path / "evaluation.json")["checkpoints"]["best"]
                tr = read(path / "training.json")
                if family == "lewm" and tr["model_config"]["rollout"] != "reencode":
                    raise ValueError("Wrong LeWM variant")
                values = [
                    ev["replay"]["decodings"]["argmax"]["horizons"][f"k{k}"][
                        "skill_score"
                    ]
                    for k in (1, 3, 7)
                ]
                values += [
                    ev["imagination"]["horizons"][f"k{k}"]["board_energy_score"]
                    for k in (3, 7)
                ]
                values += [
                    ev["imagination"]["horizons"][f"k{k}"]["goals_left_energy_score"]
                    for k in (1, 3, 7)
                ]
                metrics.append(values)
                data.append(dict(family=family, band=band, seed=seed, metrics=values))
            rows.append([name, band] + [pm(v) for v in np.asarray(metrics).T])
    headers = [
        "Model",
        "Parameters",
        "Skill $k=1$",
        "Skill $k=3$",
        "Skill $k=7$",
        "Board ES $k=3$",
        "Board ES $k=7$",
        "Goal ES $k=1$",
        "Goal ES $k=3$",
        "Goal ES $k=7$",
    ]
    table(
        out,
        "table08_world_models",
        headers,
        rows,
        "World-model validation performance (mean $\\pm$ sample SD across three seeds).",
        "tab:results-WM",
    )
    write(Path(out) / "wm_scaling.json", data)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    bands = p["world_model"]["bands"]
    palette = ["#0072B2", "#D55E00", "#009E73"]
    for family, color in zip(p["world_model"]["families"], palette):
        vals = np.array(
            [
                [d["metrics"] for d in data if d["family"] == family and d["band"] == b]
                for b in bands
            ]
        )
        for ax, indices, label in zip(
            axes,
            [(0, 1, 2), (3, 4), (5, 6, 7)],
            ["Persistence-normalized skill", "Board energy score", "Goal energy score"],
        ):
            for idx in indices:
                k = [1, 3, 7, 3, 7, 1, 3, 7][idx]
                y = vals[:, :, idx]
                ax.errorbar(
                    np.arange(3),
                    y.mean(1),
                    yerr=y.std(1, ddof=1),
                    label=f"{family}, k={k}",
                    color=color,
                    linestyle={1: ":", 3: "--", 7: "-"}[k],
                    capsize=2,
                )
            ax.set_ylabel(label)
            ax.set_xticks(range(3), bands)
            ax.set_xlabel("Parameters")
            ax.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.savefig(Path(out) / "world_model_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def dataset_report(root, out):
    from match3_simulator.evaluation.dataset_checks import _episodes, _transitions
    from match3_simulator.world_modeling.wm1_data import (
        load_splits,
        EpisodeStore,
        verify_store_invariants,
    )
    from match3_simulator.experiments.data import Logged4
    from match3_simulator.experiments.training_utils import accepted
    from match3_simulator.experiments.guard import (
        SerializedPredictions,
        phase,
    )

    root = Path(root)
    splits = read(root / "splits.json")
    release_rows = []
    split_rows = []
    cohort_rows = []
    for regime in ("natural", "randomized"):
        # Descriptive test-outcome summaries are permitted only after final predictions exist.
        predictions = sorted(
            Path("runs/evaluation").glob(f"*{regime}*/predictions.npz")
        )
        if not predictions:
            raise FileNotFoundError(
                f"Evaluate {regime} before summarizing test outcomes"
            )
        _, mastery, _, _ = accepted(regime)
        logged = Logged4(
            "runs/data/schema2",
            regime,
            splits=load_splits(root),
            mastery_config=mastery,
        )
        with phase("correct"):
            y = logged.lock.unlock(SerializedPredictions.of(predictions[0]))
        churn = {int(pid): int(v) for pid, v in zip(y.player_ids, y["churn_after"])}
        episodes, _ = _episodes(root, regime)
        a = _transitions(root, regime)
        opening = set(
            map(int, a["player_id"][(a["attempt_id"] == 20) & (a["step_id"] == 0)])
        )
        release_rows.append(
            [
                regime.title(),
                len({p for p, _ in episodes}),
                len(episodes),
                len(a["player_id"]),
                len(opening),
            ]
        )
        for split in ("train", "validation", "test"):
            split_rows.append(
                [
                    regime.title(),
                    split.title(),
                    len(splits[split]),
                    len(opening & set(splits[split])),
                ]
            )
        for level in LEVELS:
            ids = {
                p
                for (p, attempt), r in episodes.items()
                if attempt == 20 and r["level"] == level and p in opening
            }
            val = ids & set(splits["validation"])
            test = ids & set(splits["test"])
            count = sum(churn[i] for i in test)
            cohort_rows.append(
                [
                    regime.title(),
                    level.title(),
                    len(val),
                    len(test),
                    count,
                    f"{100*count/len(test):.2f}",
                ]
            )
    table(
        out,
        "table04_release",
        ["Regime", "Players", "Attempts", "Transitions", "Active at attempt 20"],
        release_rows,
        "Complete-release counts.",
        "tab:data-release",
    )
    table(
        out,
        "table05_splits",
        ["Regime", "Split", "Nominal players", "Active at attempt 20"],
        split_rows,
        "Player counts by split and landmark eligibility.",
        "tab:data-splits",
    )
    table(
        out,
        "table06_cohorts",
        ["Regime", "Level", "Validation n", "Test n", "Churn count", "Churn (\\%)"],
        cohort_rows,
        "Landmark cohort composition and observed test churn.",
        "tab:data-cohorts",
    )
    stats = []
    for split in ("train", "validation"):
        store = EpisodeStore(
            "runs/wm1/data/natural", split, splits=load_splits(root), target="settled"
        )
        stats.append(store.summary())
        verify_store_invariants(store)
    write(Path(out) / "wm_dataset_statistics.json", stats)
    keys = [k for k, v in stats[0].items() if isinstance(v, (int, float))]
    table(
        out,
        "table07_wm_data",
        ["Quantity", "Training", "Validation"],
        [[k.replace("_", " "), stats[0][k], stats[1][k]] for k in keys],
        "Natural-assignment data used for world modeling.",
        "tab:wm-data",
    )
