"""Physically constrained material-profile generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class MaterialProfiles:
    """Cell-wise material fields and their layer metadata."""

    sigma_t: FloatArray
    sigma_s: FloatArray
    layer_count: IntArray
    interfaces: IntArray

    @property
    def sigma_a(self) -> FloatArray:
        return self.sigma_t - self.sigma_s

    @property
    def scattering_ratio(self) -> FloatArray:
        return self.sigma_s / self.sigma_t


def generate_layered_materials(
    *,
    samples: int,
    cells: int,
    min_layers: int = 1,
    max_layers: int = 2,
    sigma_t_range: tuple[float, float] = (0.2, 2.0),
    scattering_ratio_range: tuple[float, float] = (0.05, 0.95),
    min_layer_cells: int = 5,
    seed: int | None = None,
) -> MaterialProfiles:
    """Generate piecewise-constant, physically valid material profiles.

    Interfaces are stored as cell indices. Unused interface entries are -1,
    which makes the metadata suitable for fixed-shape NPZ files.
    """
    if samples <= 0 or cells <= 0:
        raise ValueError("samples and cells must be positive.")
    if not 1 <= min_layers <= max_layers:
        raise ValueError("Require 1 <= min_layers <= max_layers.")
    if min_layer_cells <= 0 or max_layers * min_layer_cells > cells:
        raise ValueError("min_layer_cells is incompatible with cells and max_layers.")

    t_low, t_high = sigma_t_range
    c_low, c_high = scattering_ratio_range
    if not 0.0 < t_low <= t_high:
        raise ValueError("sigma_t_range must be positive and ordered.")
    if not 0.0 <= c_low <= c_high <= 1.0:
        raise ValueError("scattering_ratio_range must lie within [0, 1].")

    rng = np.random.default_rng(seed)
    sigma_t = np.empty((samples, cells), dtype=np.float64)
    sigma_s = np.empty_like(sigma_t)
    layer_counts = rng.integers(min_layers, max_layers + 1, size=samples)
    interfaces = np.full((samples, max_layers - 1), -1, dtype=np.int64)

    for sample_index, layers in enumerate(layer_counts):
        remaining = cells - int(layers) * min_layer_cells
        if layers == 1:
            widths = np.array([cells], dtype=np.int64)
        else:
            extra = rng.multinomial(remaining, np.full(int(layers), 1.0 / layers))
            widths = extra + min_layer_cells
        boundaries = np.cumsum(widths)[:-1]
        interfaces[sample_index, : len(boundaries)] = boundaries

        layer_total = rng.uniform(t_low, t_high, size=int(layers))
        layer_ratio = rng.uniform(c_low, c_high, size=int(layers))
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [cells]))
        for start, end, total, ratio in zip(
            starts, ends, layer_total, layer_ratio, strict=True
        ):
            sigma_t[sample_index, start:end] = total
            sigma_s[sample_index, start:end] = ratio * total

    return MaterialProfiles(
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        layer_count=layer_counts.astype(np.int64),
        interfaces=interfaces,
    )
