"""Evaluate a saved material-aware DeepONet checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeponet_1d.experiments import ExperimentConfig, build_model  # noqa: E402
from deeponet_1d.learning import (  # noqa: E402
    FieldStatistics,
    NormalizationStatistics,
    denormalize_flux,
    load_array_dataset,
    normalize_inputs,
    regression_metrics,
)
from deeponet_1d.publication import (  # noqa: E402
    bootstrap_mean_interval,
    grouped_regression_metrics,
    per_sample_relative_l2,
    regime_labels,
    stratified_error_summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path)
    args = parser.parse_args()
    configuration = json.loads((args.run_dir / "config.json").read_text())
    data = load_array_dataset(args.dataset)
    with np.load(args.run_dir / "normalization.npz") as values:
        statistics = NormalizationStatistics(
            source=FieldStatistics(float(values["Q_mean"]), float(values["Q_std"])),
            sigma_t=FieldStatistics(
                float(values["Sigma_t_mean"]), float(values["Sigma_t_std"])
            ),
            sigma_s=FieldStatistics(
                float(values["Sigma_s_mean"]), float(values["Sigma_s_std"])
            ),
            scalar_flux=FieldStatistics(
                float(values["phi_mean"]), float(values["phi_std"])
            ),
            x_min=float(values["x_min"]),
            x_max=float(values["x_max"]),
        )
    config = ExperimentConfig(
        architecture=configuration["architecture"],
        latent_dim=configuration["latent_dim"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, data.x.size).to(device)
    model.load_state_dict(torch.load(args.run_dir / "model.pt", map_location=device))
    model.eval()
    branch, trunk, _ = normalize_inputs(data, statistics)
    coordinates = torch.from_numpy(trunk).to(device)
    predictions = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(branch), 64):
            inputs = torch.from_numpy(branch[start : start + 64]).to(device)
            predictions.append(model(inputs, coordinates).cpu().numpy())
    elapsed = time.perf_counter() - started
    prediction = denormalize_flux(np.concatenate(predictions), statistics).astype(
        np.float64
    )
    errors = per_sample_relative_l2(prediction, data.scalar_flux)
    low, high = bootstrap_mean_interval(errors, resamples=20_000)
    metrics = regression_metrics(prediction, data.scalar_flux, data.interfaces)
    if data.source_family is not None:
        metrics["by_source_family"] = grouped_regression_metrics(
            prediction,
            data.scalar_flux,
            data.interfaces,
            data.source_family,
            seed=142,
        )
    metrics.update(
        {
            "mean_relative_l2_ci95_low": low,
            "mean_relative_l2_ci95_high": high,
            "prediction_seconds": elapsed,
            "by_scattering_regime": stratified_error_summary(
                errors, regime_labels(data.sigma_t, data.sigma_s)
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.prediction_output:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.prediction_output,
            prediction=prediction,
            truth=data.scalar_flux,
            relative_l2=errors,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
