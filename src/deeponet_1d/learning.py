"""Dataset, normalization, and evaluation utilities for the retained DeepONet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class ArrayDataset:
    x: NDArray[np.float64]
    source: NDArray[np.float64]
    sigma_t: NDArray[np.float64]
    sigma_s: NDArray[np.float64]
    scalar_flux: NDArray[np.float64]
    interfaces: NDArray[np.int64]
    source_family: NDArray[np.str_] | None = None


@dataclass(frozen=True)
class FieldStatistics:
    mean: float
    std: float


@dataclass(frozen=True)
class NormalizationStatistics:
    source: FieldStatistics
    sigma_t: FieldStatistics
    sigma_s: FieldStatistics
    scalar_flux: FieldStatistics
    x_min: float
    x_max: float


def load_array_dataset(path: str | Path) -> ArrayDataset:
    """Load and validate an NPZ dataset produced by ``save_dataset``."""
    with np.load(path) as archive:
        required = {"x", "Q", "Sigma_t", "Sigma_s", "phi", "interfaces"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Dataset {path} is missing fields: {sorted(missing)}")
        result = ArrayDataset(
            x=archive["x"].astype(np.float64),
            source=archive["Q"].astype(np.float64),
            sigma_t=archive["Sigma_t"].astype(np.float64),
            sigma_s=archive["Sigma_s"].astype(np.float64),
            scalar_flux=archive["phi"].astype(np.float64),
            interfaces=archive["interfaces"].astype(np.int64),
            source_family=(
                archive["source_family"].astype(str)
                if "source_family" in archive.files
                else None
            ),
        )
    samples, cells = result.source.shape
    for name, values in (
        ("Sigma_t", result.sigma_t),
        ("Sigma_s", result.sigma_s),
        ("phi", result.scalar_flux),
    ):
        if values.shape != (samples, cells):
            raise ValueError(f"{name} has shape {values.shape}; expected {(samples, cells)}.")
    if result.x.shape != (cells,):
        raise ValueError(f"x has shape {result.x.shape}; expected {(cells,)}.")
    return result


def _field_statistics(values: NDArray[np.float64]) -> FieldStatistics:
    standard_deviation = float(values.std())
    if standard_deviation <= 0.0 or not np.isfinite(standard_deviation):
        raise ValueError("Cannot normalize a constant or non-finite field.")
    return FieldStatistics(float(values.mean()), standard_deviation)


def fit_normalization(training: ArrayDataset) -> NormalizationStatistics:
    """Fit all learned statistics exclusively from the training split."""
    return NormalizationStatistics(
        source=_field_statistics(training.source),
        sigma_t=_field_statistics(training.sigma_t),
        sigma_s=_field_statistics(training.sigma_s),
        scalar_flux=_field_statistics(training.scalar_flux),
        x_min=float(training.x.min()),
        x_max=float(training.x.max()),
    )


def normalize_inputs(
    dataset: ArrayDataset, statistics: NormalizationStatistics
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    fields = []
    for values, stats in (
        (dataset.source, statistics.source),
        (dataset.sigma_t, statistics.sigma_t),
        (dataset.sigma_s, statistics.sigma_s),
    ):
        fields.append((values - stats.mean) / stats.std)
    branch = np.concatenate(fields, axis=1).astype(np.float32)
    span = statistics.x_max - statistics.x_min
    if span <= 0.0:
        raise ValueError("Coordinate range must be positive.")
    trunk = (2.0 * (dataset.x - statistics.x_min) / span - 1.0).reshape(-1, 1)
    target = (
        (dataset.scalar_flux - statistics.scalar_flux.mean)
        / statistics.scalar_flux.std
    )
    return branch, trunk.astype(np.float32), target.astype(np.float32)


def denormalize_flux(values: NDArray[np.float32], statistics: NormalizationStatistics):
    return values * statistics.scalar_flux.std + statistics.scalar_flux.mean


def make_mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(input_size, output_size))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def regression_metrics(
    prediction: NDArray[np.float64],
    truth: NDArray[np.float64],
    interfaces: NDArray[np.int64] | None = None,
) -> dict[str, float]:
    """Return complementary dimensional and scale-independent error metrics."""
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and truth must be matching two-dimensional arrays.")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise ValueError("prediction and truth must contain only finite values.")

    residual = prediction - truth
    absolute_error = np.abs(residual)
    squared_error = residual**2
    epsilon = 1.0e-12
    relative_l2 = np.linalg.norm(residual, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1), epsilon
    )
    relative_l1 = np.sum(absolute_error, axis=1) / np.maximum(
        np.sum(np.abs(truth), axis=1), epsilon
    )
    mse = float(np.mean(squared_error))
    rmse = float(np.sqrt(mse))
    target_rms = float(np.sqrt(np.mean(truth**2)))
    total_sum_squares = float(np.sum((truth - truth.mean()) ** 2))
    residual_sum_squares = float(np.sum(squared_error))

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(absolute_error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_error_bias": float(np.mean(residual)),
        "normalized_rmse": rmse / max(target_rms, epsilon),
        "r2_score": 1.0 - residual_sum_squares / max(total_sum_squares, epsilon),
        "mean_relative_l1": float(relative_l1.mean()),
        "median_relative_l1": float(np.median(relative_l1)),
        "p95_relative_l1": float(np.percentile(relative_l1, 95)),
        "mean_relative_l2": float(relative_l2.mean()),
        "median_relative_l2": float(np.median(relative_l2)),
        "p95_relative_l2": float(np.percentile(relative_l2, 95)),
        "max_relative_l2": float(relative_l2.max()),
    }
    if interfaces is not None:
        interface_errors = []
        for sample_index, sample_interfaces in enumerate(interfaces):
            for interface in sample_interfaces[sample_interfaces >= 0]:
                start = max(0, int(interface) - 1)
                end = min(truth.shape[1], int(interface) + 2)
                interface_errors.extend(absolute_error[sample_index, start:end])
        if interface_errors:
            interface_array = np.asarray(interface_errors, dtype=np.float64)
            metrics["interface_mae"] = float(np.mean(interface_array))
            metrics["interface_rmse"] = float(
                np.sqrt(np.mean(interface_array**2))
            )
        else:
            metrics["interface_mae"] = 0.0
            metrics["interface_rmse"] = 0.0
    return metrics

