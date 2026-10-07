"""Plot observational and interventional churn curves from a benchmark report."""

from __future__ import annotations

import json
from pathlib import Path


def _select_landmark_report(
    report: dict[str, object],
) -> tuple[dict[str, object], str | None]:
    seed_reports = report.get("seed_reports")
    if seed_reports is None:
        return report, None
    if not isinstance(seed_reports, list) or not seed_reports:
        raise ValueError("validation suite contains no seed reports")
    selected = seed_reports[0]
    passed_count = sum(bool(item["passed"]) for item in seed_reports)
    subtitle = (
        f"validation seed {selected['seed']} shown; "
        f"{passed_count}/{len(seed_reports)} seeds passed"
    )
    return selected, subtitle


def _curve_values(level_report: dict[str, object]) -> dict[str, object]:
    """Return the authoritative curve payload across report schema versions."""
    engine = level_report.get("engine")
    if engine is None:
        return level_report
    values = dict(engine)
    bootstrap = level_report.get("bootstrap")
    if isinstance(bootstrap, dict):
        values["contrast_intervals"] = {
            "causal": bootstrap["causal_recommendation_contrast"],
            "observational": bootstrap[
                "observational_recommendation_contrast"
            ],
        }
    return values


def plot_landmark_report(report_path: str | Path, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    loaded_report = json.loads(Path(report_path).read_text())
    report, subtitle = _select_landmark_report(loaded_report)
    levels = report["levels"]
    figure, axes = plt.subplots(
        1, len(levels), figsize=(4.3 * len(levels), 3.7), sharey=True
    )
    if len(levels) == 1:
        axes = [axes]
    for axis, (level_name, level_report) in zip(axes, levels.items()):
        values = _curve_values(level_report)
        grid = values["grid"]
        causal = values["causal"]
        observational = values["observational"]
        axis.plot(
            grid,
            observational,
            color="#c2452d",
            marker="o",
            markersize=3.5,
            label="observational",
        )
        axis.plot(
            grid,
            causal,
            color="#1f4e79",
            marker="s",
            markersize=3.5,
            label="interventional",
        )
        axis.axvline(
            values["observational_optimum"],
            color="#c2452d",
            linestyle=":",
            linewidth=1.2,
        )
        axis.axvline(
            values["causal_optimum"],
            color="#1f4e79",
            linestyle=":",
            linewidth=1.2,
        )
        axis.set_title(level_name)
        axis.set_xlabel("served difficulty E")
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        annotation = f"optimum gap = {values['recommendation_gap']:.2f}"
        intervals = values.get("contrast_intervals")
        if intervals is not None:
            causal_interval = intervals["causal"]
            observational_interval = intervals["observational"]
            annotation += (
                "\n"
                f"causal contrast [{causal_interval['lower']:.2f}, "
                f"{causal_interval['upper']:.2f}]"
                "\n"
                f"naive contrast [{observational_interval['lower']:.2f}, "
                f"{observational_interval['upper']:.2f}]"
            )
        use_lower_corner = min(causal) > 0.65 and min(observational) > 0.65
        axis.text(
            0.03,
            0.04 if use_lower_corner else 0.96,
            annotation,
            transform=axis.transAxes,
            va="bottom" if use_lower_corner else "top",
            fontsize=7.5,
            linespacing=1.35,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    axes[0].set_ylabel("next-attempt churn probability")
    axes[-1].legend(frameon=False, fontsize=9)
    if subtitle is not None:
        figure.suptitle(subtitle, fontsize=9)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(plot_landmark_report(args.report, args.out))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["_curve_values", "plot_landmark_report"]