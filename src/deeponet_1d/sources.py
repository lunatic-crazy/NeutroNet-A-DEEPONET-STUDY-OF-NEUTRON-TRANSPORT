"""Source-function generation for one-dimensional slab problems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gaussian_random_field_sources(
    *,
    cells: int,
    cell_width: float,
    samples: int,
    mean: float = 5.0,
    length_scale: float = 0.1,
    variance: float = 1.0,
    seed: int | None = None,
    jitter: float = 1.0e-10,
    minimum: float | None = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample squared-exponential Gaussian random fields at cell centers."""
    if cells <= 0 or samples <= 0 or cell_width <= 0.0:
        raise ValueError("cells, samples, and cell_width must be positive.")
    if length_scale <= 0.0 or variance < 0.0 or jitter < 0.0:
        raise ValueError("Invalid covariance parameters.")

    rng = np.random.default_rng(seed)
    x = (np.arange(cells, dtype=np.float64) + 0.5) * cell_width
    distances = x[:, None] - x[None, :]
    covariance = variance * np.exp(
        -(distances**2) / (2.0 * length_scale**2)
    )
    factor = np.linalg.cholesky(covariance + jitter * np.eye(cells))
    sources = mean + rng.standard_normal((samples, cells)) @ factor.T
    if minimum is not None:
        sources = np.maximum(sources, minimum)
    return x, sources


def mixed_source_samples(
    *,
    cells: int,
    cell_width: float,
    samples: int,
    seed: int | None = None,
    minimum: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.str_]]:
    """Generate a balanced, shuffled mix of diverse nonnegative sources.

    Narrow and wide peaks are separate families so validation cannot hide poor
    localized-source performance behind easier broad peaks. Combination
    sources superpose a discontinuity, a sinusoid, and a localized peak.
    """
    if cells <= 0 or samples <= 0 or cell_width <= 0.0:
        raise ValueError("cells, samples, and cell_width must be positive.")
    if minimum < 0.0:
        raise ValueError("minimum must be non-negative.")

    rng = np.random.default_rng(seed)
    x = (np.arange(cells, dtype=np.float64) + 0.5) * cell_width
    length = cells * cell_width
    family_names = np.array(
        [
            "grf",
            "gaussian_narrow",
            "gaussian_wide",
            "step",
            "sinusoid",
            "multiple_peak",
            "combination",
        ],
        dtype="U20",
    )
    labels = np.resize(family_names, samples).copy()
    rng.shuffle(labels)
    sources = np.empty((samples, cells), dtype=np.float64)

    def gaussian(center: float, width: float, amplitude: float) -> NDArray[np.float64]:
        return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

    for index, family in enumerate(labels):
        if family == "grf":
            _, source = gaussian_random_field_sources(
                cells=cells,
                cell_width=cell_width,
                samples=1,
                seed=int(rng.integers(0, 2**32 - 1)),
                minimum=minimum,
            )
            sources[index] = source[0]
        elif family == "gaussian_narrow":
            center = rng.uniform(0.05 * length, 0.95 * length)
            width = rng.uniform(0.01 * length, 0.04 * length)
            background = rng.uniform(1.0, 3.0)
            amplitude = rng.uniform(5.0, 10.0)
            sources[index] = background + gaussian(center, width, amplitude)
        elif family == "gaussian_wide":
            center = rng.uniform(0.10 * length, 0.90 * length)
            width = rng.uniform(0.08 * length, 0.22 * length)
            background = rng.uniform(1.0, 3.0)
            amplitude = rng.uniform(5.0, 10.0)
            sources[index] = background + gaussian(center, width, amplitude)
        elif family == "step":
            interface = rng.uniform(0.15 * length, 0.85 * length)
            left = rng.uniform(1.0, 4.0)
            right = rng.uniform(6.0, 12.0)
            if rng.random() < 0.5:
                left, right = right, left
            sources[index] = np.where(x < interface, left, right)
        elif family == "sinusoid":
            mean = rng.uniform(4.0, 8.0)
            amplitude = rng.uniform(1.0, 3.0)
            cycles = rng.uniform(0.5, 3.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            sources[index] = mean + amplitude * np.sin(
                2.0 * np.pi * cycles * x / length + phase
            )
        elif family == "multiple_peak":
            background = rng.uniform(1.0, 3.0)
            source = np.full(cells, background, dtype=np.float64)
            for _ in range(int(rng.integers(2, 5))):
                source += gaussian(
                    center=rng.uniform(0.05 * length, 0.95 * length),
                    width=rng.uniform(0.015 * length, 0.06 * length),
                    amplitude=rng.uniform(2.0, 6.0),
                )
            sources[index] = source
        elif family == "combination":
            interface = rng.uniform(0.20 * length, 0.80 * length)
            step_height = rng.uniform(1.0, 4.0)
            if rng.random() < 0.5:
                step_height = -step_height
            mean = rng.uniform(4.0, 7.0)
            cycles = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            source = mean + np.where(x < interface, 0.0, step_height)
            source += rng.uniform(0.5, 2.0) * np.sin(
                2.0 * np.pi * cycles * x / length + phase
            )
            source += gaussian(
                center=rng.uniform(0.10 * length, 0.90 * length),
                width=rng.uniform(0.02 * length, 0.08 * length),
                amplitude=rng.uniform(3.0, 7.0),
            )
            sources[index] = source
        else:  # pragma: no cover - labels are defined immediately above.
            raise RuntimeError(f"Unsupported source family: {family}")

    return x, np.maximum(sources, minimum), labels

