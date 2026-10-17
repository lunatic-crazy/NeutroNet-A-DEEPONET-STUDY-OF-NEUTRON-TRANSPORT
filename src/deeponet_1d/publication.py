"""Retained DeepONet architecture and statistical evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .learning import make_mlp, regression_metrics


class SeparateBranchDeepONet(nn.Module):
    """Parameter-controlled DeepONet with one encoder per physical field.

    The 3M-vector is split into Q, Sigma_t, and Sigma_s, and each physical
    field is encoded independently before feature fusion.
    """

    def __init__(
        self,
        *,
        cells: int,
        latent_dim: int = 128,
        encoder_width: int = 128,
        fusion_width: int = 192,
        trunk_hidden: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.cells = cells
        self.source_branch = make_mlp([cells, encoder_width, latent_dim])
        self.total_branch = make_mlp([cells, encoder_width, latent_dim])
        self.scatter_branch = make_mlp([cells, encoder_width, latent_dim])
        self.fusion = make_mlp([3 * latent_dim, fusion_width, latent_dim])
        self.trunk = make_mlp([1, *trunk_hidden, latent_dim])
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, branch_input: Tensor, coordinates: Tensor) -> Tensor:
        if branch_input.ndim != 2 or branch_input.shape[1] != 3 * self.cells:
            raise ValueError(f"branch_input must have shape (batch, {3 * self.cells}).")
        source, sigma_t, sigma_s = torch.split(branch_input, self.cells, dim=1)
        encoded = torch.cat(
            (
                self.source_branch(source),
                self.total_branch(sigma_t),
                self.scatter_branch(sigma_s),
            ),
            dim=1,
        )
        branch_features = self.fusion(encoded)
        trunk_features = self.trunk(coordinates)
        return torch.einsum("bp,mp->bm", branch_features, trunk_features) + self.bias


def grouped_regression_metrics(
    prediction: NDArray[np.float64],
    truth: NDArray[np.float64],
    interfaces: NDArray[np.int64],
    labels: NDArray[np.str_],
    *,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """Compute the full metric suite independently for each string label."""
    summary: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(np.unique(labels)):
        mask = labels == label
        errors = per_sample_relative_l2(prediction[mask], truth[mask])
        low, high = bootstrap_mean_interval(errors, seed=seed + index)
        metrics: dict[str, float | int] = regression_metrics(
            prediction[mask], truth[mask], interfaces[mask]
        )
        metrics.update(
            {
                "samples": int(mask.sum()),
                "mean_relative_l2_ci95_low": low,
                "mean_relative_l2_ci95_high": high,
            }
        )
        summary[str(label)] = metrics
    return summary


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def per_sample_relative_l2(
    prediction: NDArray[np.float64], truth: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.linalg.norm(prediction - truth, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1), 1.0e-12
    )


def bootstrap_mean_interval(
    values: NDArray[np.float64],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a sample mean."""
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be a one-dimensional sample of size >= 2.")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 1_000
    for start in range(0, resamples, chunk):
        count = min(chunk, resamples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def regime_labels(
    sigma_t: NDArray[np.float64], sigma_s: NDArray[np.float64]
) -> NDArray[np.str_]:
    """Classify each slab using its cell-averaged scattering ratio."""
    ratio = np.mean(sigma_s / sigma_t, axis=1)
    return np.where(
        ratio < 0.3,
        "absorption_dominated",
        np.where(ratio < 0.7, "moderate", "scattering_dominated"),
    )


def stratified_error_summary(
    errors: NDArray[np.float64],
    labels: NDArray[np.str_],
    *,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    summary = {}
    for index, label in enumerate(np.unique(labels)):
        selected = errors[labels == label]
        if selected.size < 2:
            low = high = float(selected.mean()) if selected.size else float("nan")
        else:
            low, high = bootstrap_mean_interval(selected, seed=seed + index)
        summary[str(label)] = {
            "samples": int(selected.size),
            "mean_relative_l2": float(selected.mean()) if selected.size else float("nan"),
            "median_relative_l2": float(np.median(selected)) if selected.size else float("nan"),
            "ci95_low": low,
            "ci95_high": high,
        }
    return summary
