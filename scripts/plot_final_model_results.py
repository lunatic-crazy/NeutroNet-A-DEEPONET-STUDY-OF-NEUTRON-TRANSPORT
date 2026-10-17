"""Create publication figures for the retained final DeepONet model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "combination": "Combination",
    "gaussian_narrow": "Narrow Gaussian",
    "gaussian_wide": "Wide Gaussian",
    "grf": "GRF",
    "multiple_peak": "Multiple peaks",
    "sinusoid": "Sinusoid",
    "step": "Step",
}

COLORS = {
    "In distribution": "#0072B2",
    "Three-layer OOD": "#D55E00",
    "Unseen sources": "#009E73",
}


def save(figure: plt.Figure, destination: Path) -> None:
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    unseen = json.loads((args.run_dir / "unseen_source_ood.json").read_text())
    with np.load(args.run_dir / "predictions.npz") as archive:
        predictions = {key: archive[key] for key in archive.files}

    split_names = ["test_in_distribution", "test_three_layer_ood"]
    split_labels = ["In distribution", "Three-layer OOD", "Unseen sources"]
    summaries = [metrics["splits"][name] for name in split_names] + [unseen]

    metric_columns = (
        "mse", "rmse", "mae", "median_absolute_error", "max_absolute_error",
        "mean_error_bias", "normalized_rmse", "r2_score", "mean_relative_l1",
        "mean_relative_l2", "interface_mae", "interface_rmse",
    )
    with (output / "error_metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("distribution", *metric_columns))
        writer.writeheader()
        for label, summary in zip(split_labels, summaries, strict=True):
            writer.writerow(
                {"distribution": label, **{name: summary[name] for name in metric_columns}}
            )

    colors = [COLORS[label] for label in split_labels]
    positions = np.arange(len(split_labels))
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    axes[0, 0].bar(positions, [item["mse"] for item in summaries], color=colors)
    axes[0, 0].set_title("Mean squared error")
    axes[0, 0].set_ylabel(r"MSE (flux$^2$)")

    width = 0.34
    axes[0, 1].bar(
        positions - width / 2,
        [item["rmse"] for item in summaries],
        width=width,
        label="RMSE",
        color="#5B9BD5",
    )
    axes[0, 1].bar(
        positions + width / 2,
        [item["mae"] for item in summaries],
        width=width,
        label="MAE",
        color="#A5A5A5",
    )
    axes[0, 1].set_title("Absolute-error scale")
    axes[0, 1].set_ylabel("Flux")
    axes[0, 1].legend(frameon=False)

    width = 0.24
    for offset, key, label, color in (
        (-width, "normalized_rmse", "NRMSE", "#4472C4"),
        (0.0, "mean_relative_l1", r"Relative $L_1$", "#70AD47"),
        (width, "mean_relative_l2", r"Relative $L_2$", "#ED7D31"),
    ):
        axes[1, 0].bar(
            positions + offset,
            [100.0 * item[key] for item in summaries],
            width=width,
            label=label,
            color=color,
        )
    axes[1, 0].set_title("Scale-independent errors")
    axes[1, 0].set_ylabel("Error (%)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    r2_values = [item["r2_score"] for item in summaries]
    axes[1, 1].bar(positions, r2_values, color=colors)
    axes[1, 1].set_title(r"Coefficient of determination ($R^2$)")
    axes[1, 1].set_ylabel(r"$R^2$")
    axes[1, 1].set_ylim(min(0.8, min(r2_values) - 0.02), 1.005)

    for axis in axes.flat:
        axis.set_xticks(positions, split_labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Prediction error across evaluation sets")
    save(figure, output / "error_metrics_dashboard")
    means = np.array([item["mean_relative_l2"] for item in summaries]) * 100
    medians = np.array([item["median_relative_l2"] for item in summaries]) * 100
    p95 = np.array([item["p95_relative_l2"] for item in summaries]) * 100
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    bars = axis.bar(x, means, color=colors, width=0.62)
    axis.scatter(x, medians, color="white", edgecolor="#222222", label="Median", zorder=3)
    axis.scatter(x, p95, marker="D", color="#7030A0", label="95th percentile", zorder=3)
    for bar, value in zip(bars, means, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.2f}%", ha="center")
    axis.set_xticks(x, split_labels)
    axis.set_ylabel(r"Relative $L_2$ error (%)")
    axis.set_title("Relative flux error by evaluation set")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)
    save(figure, output / "performance_summary")

    families = list(LABELS)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    for offset, name, label, color in (
        (-width / 2, split_names[0], "In distribution", COLORS["In distribution"]),
        (width / 2, split_names[1], "Three-layer OOD", COLORS["Three-layer OOD"]),
    ):
        values = [metrics["splits"][name]["by_source_family"][family]["mean_relative_l2"] * 100 for family in families]
        axis.bar(np.arange(len(families)) + offset, values, width=width, label=label, color=color)
    axis.set_xticks(np.arange(len(families)), [LABELS[name] for name in families], rotation=24, ha="right")
    axis.set_ylabel(r"Mean relative $L_2$ error (%)")
    axis.set_title("Relative flux error by source family")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)
    save(figure, output / "source_family_comparison")

    history = np.genfromtxt(args.run_dir / "training_history.csv", delimiter=",", names=True)
    figure, axis = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    axis.semilogy(history["epoch"], history["training_loss"], label="Training")
    axis.semilogy(history["epoch"], history["validation_loss"], label="Validation")
    axis.axvline(metrics["best_epoch"], color="#C00000", linestyle="--", label=f"Best epoch {metrics['best_epoch']}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(r"Mean squared relative $L_2$ loss")
    axis.set_title("Training history")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    save(figure, output / "training_history")

    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), sharex=True, constrained_layout=True)
    quantiles = (0.5, 0.9, 1.0)
    for row, (name, row_label) in enumerate(zip(split_names, split_labels[:2], strict=True)):
        with np.load(args.data_dir / f"{name}.npz") as data:
            coordinates = data["x"]
        errors = predictions[f"{name}_relative_l2"]
        truth = predictions[f"{name}_truth"]
        prediction = predictions[f"{name}_prediction"]
        order = np.argsort(errors)
        for column, quantile in enumerate(quantiles):
            index = order[-1] if quantile == 1.0 else order[min(int(quantile * len(order)), len(order) - 1)]
            axis = axes[row, column]
            axis.plot(coordinates, truth[index], color="#1F1F1F", label="Reference")
            axis.plot(coordinates, prediction[index], color="#D55E00", linestyle="--", label="DeepONet")
            descriptor = "worst" if quantile == 1.0 else f"q={quantile:.1f}"
            axis.set_title(f"{row_label}, {descriptor}\nerror={100 * errors[index]:.2f}%")
            axis.grid(alpha=0.3)
            if column == 0:
                axis.set_ylabel(r"Scalar flux, $\phi(x)$")
            if row == 1:
                axis.set_xlabel(r"Position, $x$ (cm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    save(figure, output / "prediction_examples")

    # Show the full case-to-case error distribution, rather than only aggregate
    # means. The unseen-source predictions are stored in a separate archive.
    error_sets = {
        split_labels[0]: predictions["test_in_distribution_relative_l2"],
        split_labels[1]: predictions["test_three_layer_ood_relative_l2"],
    }
    with np.load(args.run_dir / "unseen_source_ood_predictions.npz") as archive:
        error_sets[split_labels[2]] = archive["relative_l2"]
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for label, values in error_sets.items():
        ordered = np.sort(np.asarray(values)) * 100.0
        probability = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.step(ordered, probability, where="post", label=label, color=COLORS[label])
    axis.set_xlabel(r"Per-case relative $L_2$ error (%)")
    axis.set_ylabel("Cumulative proportion of cases")
    axis.set_title("Distribution of flux prediction errors")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    save(figure, output / "relative_error_cdf")

    # Spatial absolute-error profiles summarize the location and spread of
    # pointwise errors across cases. Bands are interquartile ranges, not CIs.
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for split, label in zip(split_names, split_labels[:2], strict=True):
        truth = predictions[f"{split}_truth"]
        prediction = predictions[f"{split}_prediction"]
        with np.load(args.data_dir / f"{split}.npz") as data:
            coordinates = np.asarray(data["x"])
        coordinates = coordinates[0] if coordinates.ndim > 1 else coordinates
        pointwise_error = np.abs(prediction - truth)
        median = np.median(pointwise_error, axis=0)
        lower, upper = np.percentile(pointwise_error, [25, 75], axis=0)
        axis.plot(coordinates, median, label=label, color=COLORS[label])
        axis.fill_between(coordinates, lower, upper, color=COLORS[label], alpha=0.16)
    with np.load(args.run_dir / "unseen_source_ood_predictions.npz") as archive:
        truth = archive["truth"]
        prediction = archive["prediction"]
    with np.load(args.data_dir / "test_unseen_sources.npz") as data:
        coordinates = np.asarray(data["x"])
    coordinates = coordinates[0] if coordinates.ndim > 1 else coordinates
    pointwise_error = np.abs(prediction - truth)
    median = np.median(pointwise_error, axis=0)
    lower, upper = np.percentile(pointwise_error, [25, 75], axis=0)
    axis.plot(coordinates, median, label=split_labels[2], color=COLORS[split_labels[2]])
    axis.fill_between(coordinates, lower, upper, color=COLORS[split_labels[2]], alpha=0.16)
    axis.set_xlabel(r"Position, $x$ (cm)")
    axis.set_ylabel(r"Absolute flux error, $|\hat{\phi}-\phi|$")
    axis.set_title("Median pointwise error (shading: interquartile range)")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    save(figure, output / "spatial_absolute_error")

    for path in sorted(output.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
