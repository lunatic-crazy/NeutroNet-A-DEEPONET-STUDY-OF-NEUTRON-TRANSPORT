"""Recompute metrics from saved final-model predictions without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deeponet_1d.learning import load_array_dataset, regression_metrics
from deeponet_1d.publication import (
    bootstrap_mean_interval,
    grouped_regression_metrics,
    per_sample_relative_l2,
    regime_labels,
    stratified_error_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics.json"
    results = json.loads(metrics_path.read_text())
    configuration = json.loads((args.run_dir / "config.json").read_text())
    seed = int(configuration["seed"])
    with np.load(args.run_dir / "predictions.npz") as archive:
        predictions = {name: archive[name] for name in archive.files}

    split_files = {
        "validation": "validation.npz",
        "test_in_distribution": "test_in_distribution.npz",
        "test_three_layer_ood": "test_three_layer_ood.npz",
    }
    for split_index, (name, filename) in enumerate(split_files.items()):
        dataset = load_array_dataset(args.data_dir / filename)
        prediction = predictions[f"{name}_prediction"].astype(np.float64)
        truth = predictions[f"{name}_truth"].astype(np.float64)
        np.testing.assert_allclose(truth, dataset.scalar_flux, rtol=0.0, atol=0.0)
        errors = per_sample_relative_l2(prediction, truth)
        np.testing.assert_allclose(
            errors, predictions[f"{name}_relative_l2"], rtol=1.0e-12, atol=1.0e-14
        )
        updated = regression_metrics(prediction, truth, dataset.interfaces)
        if dataset.source_family is not None:
            updated["by_source_family"] = grouped_regression_metrics(
                prediction,
                truth,
                dataset.interfaces,
                dataset.source_family,
                seed=seed + split_index + 100,
            )
        low, high = bootstrap_mean_interval(errors, seed=seed + split_index)
        previous = results["splits"][name]
        updated.update(
            {
                "mean_relative_l2_ci95_low": low,
                "mean_relative_l2_ci95_high": high,
                "prediction_seconds": previous["prediction_seconds"],
                "microseconds_per_sample": previous["microseconds_per_sample"],
                "by_scattering_regime": stratified_error_summary(
                    errors,
                    regime_labels(dataset.sigma_t, dataset.sigma_s),
                    seed=seed + split_index,
                ),
            }
        )
        results["splits"][name] = updated

    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
