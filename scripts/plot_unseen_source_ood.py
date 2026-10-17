"""Create publication-style figures for the unseen-source OOD benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "chirp": "Chirp",
    "exponential_layer": "Exponential layer",
    "linear_ramp": "Linear ramp",
    "lorentzian_peak": "Lorentzian peak",
    "sawtooth": "Sawtooth",
    "triangular_tent": "Triangular tent",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_error_summary(
    families: list[str], statistics: dict[str, dict[str, float]], output: Path
) -> None:
    positions = np.arange(len(families))
    mean = np.array([statistics[name]["mean"] for name in families]) * 100.0
    median = np.array([statistics[name]["median"] for name in families]) * 100.0
    p95 = np.array([statistics[name]["p95"] for name in families]) * 100.0

    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    bars = axis.bar(
        positions,
        mean,
        width=0.64,
        color="#4472C4",
        edgecolor="#1F1F1F",
        linewidth=0.6,
        label="Mean",
        zorder=2,
    )
    axis.scatter(
        positions,
        median,
        marker="o",
        s=28,
        color="#FFFFFF",
        edgecolor="#1F1F1F",
        linewidth=0.9,
        label="Median",
        zorder=4,
    )
    axis.scatter(
        positions,
        p95,
        marker="D",
        s=28,
        color="#ED7D31",
        edgecolor="#1F1F1F",
        linewidth=0.6,
        label="95th percentile",
        zorder=4,
    )
    for bar, value in zip(bars, mean, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(positions, [LABELS[name] for name in families], rotation=22, ha="right")
    axis.set_ylabel(r"Relative $L_2$ error (%)")
    axis.set_title("Prediction error by unseen source family")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0.0, max(p95) * 1.18)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    save_figure(figure, output)


def plot_cases(
    *,
    families: list[str],
    indices: dict[str, int],
    x: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    errors: np.ndarray,
    heading: str,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.2, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    for panel, (axis, family) in enumerate(zip(axes.flat, families, strict=True)):
        index = indices[family]
        axis.plot(x, truth[index], color="#1F1F1F", linewidth=2.1, label="Reference")
        axis.plot(
            x,
            prediction[index],
            color="#D55E00",
            linewidth=1.8,
            linestyle="--",
            label="DeepONet",
        )
        axis.set_title(
            f"({chr(97 + panel)}) {LABELS[family]}  |  error = {100.0 * errors[index]:.2f}%",
            loc="left",
        )
        axis.grid(color="#D9D9D9", linewidth=0.65, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        if panel % 3 == 0:
            axis.set_ylabel(r"Scalar flux, $\phi(x)$")
        if panel >= 3:
            axis.set_xlabel(r"Position, $x$ (cm)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.suptitle(heading, fontsize=12, fontweight="bold")
    save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    with np.load(args.dataset) as data:
        x = data["x"].astype(np.float64)
        source_family = data["source_family"].astype(str)
    with np.load(args.predictions) as results:
        truth = results["truth"].astype(np.float64)
        prediction = results["prediction"].astype(np.float64)
        errors = results["relative_l2"].astype(np.float64)

    families = sorted(np.unique(source_family).tolist())
    statistics: dict[str, dict[str, float]] = {}
    representative: dict[str, int] = {}
    worst: dict[str, int] = {}
    for family in families:
        indices = np.flatnonzero(source_family == family)
        family_errors = errors[indices]
        statistics[family] = {
            "samples": int(indices.size),
            "mean": float(family_errors.mean()),
            "median": float(np.median(family_errors)),
            "p95": float(np.percentile(family_errors, 95)),
            "max": float(family_errors.max()),
        }
        representative[family] = int(
            indices[np.argmin(np.abs(family_errors - np.median(family_errors)))]
        )
        worst[family] = int(indices[np.argmax(family_errors)])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "unseen_source_ood_family_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("family", "samples", "mean_relative_l2", "median_relative_l2", "p95_relative_l2", "max_relative_l2", "representative_index", "worst_index")
        )
        for family in families:
            values = statistics[family]
            writer.writerow(
                (
                    family,
                    values["samples"],
                    values["mean"],
                    values["median"],
                    values["p95"],
                    values["max"],
                    representative[family],
                    worst[family],
                )
            )

    plot_error_summary(
        families,
        statistics,
        args.output_dir / "unseen_source_ood_error_summary",
    )
    plot_cases(
        families=families,
        indices=representative,
        x=x,
        truth=truth,
        prediction=prediction,
        errors=errors,
        heading="Representative unseen-source predictions",
        output=args.output_dir / "unseen_source_ood_representative_predictions",
    )
    plot_cases(
        families=families,
        indices=worst,
        x=x,
        truth=truth,
        prediction=prediction,
        errors=errors,
        heading="Worst unseen-source prediction in each family",
        output=args.output_dir / "unseen_source_ood_worst_predictions",
    )
    for path in sorted(args.output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
