"""Discrete-ordinates solver for a heterogeneous one-dimensional slab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TransportResult:
    """Converged solution and diagnostic information."""

    scalar_flux: FloatArray
    angular_flux: FloatArray
    ordinates: FloatArray
    weights: FloatArray
    iterations: int
    relative_change: float
    converged: bool


def _cell_array(value: ArrayLike, cells: int, name: str) -> FloatArray:
    """Return a writable one-dimensional cell array from a scalar or array."""
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        return np.full(cells, float(raw), dtype=np.float64)
    if raw.shape != (cells,):
        raise ValueError(f"{name} must be a scalar or have shape ({cells},), got {raw.shape}.")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite values.")
    return raw.copy()


def solve_sn_slab(
    *,
    length: float = 10.0,
    cell_width: float = 0.1,
    sigma_t: ArrayLike = 1.0,
    sigma_s: ArrayLike = 0.5,
    source: ArrayLike | None = None,
    directions: int = 32,
    tolerance: float = 1.0e-8,
    max_iterations: int = 10_000,
) -> TransportResult:
    """Solve one-group steady 1D slab transport with vacuum boundaries.

    ``sigma_t`` and ``sigma_s`` may be scalars for a homogeneous slab or
    arrays with one value per spatial cell for a heterogeneous slab. The
    diamond-difference discretization and source iteration match the original
    project notebooks.
    """
    if length <= 0.0 or cell_width <= 0.0:
        raise ValueError("length and cell_width must be positive.")
    cells_float = length / cell_width
    cells = int(round(cells_float))
    if not np.isclose(cells_float, cells, rtol=0.0, atol=1.0e-12):
        raise ValueError("length must be an integer multiple of cell_width.")
    if directions <= 0 or directions % 2 != 0:
        raise ValueError("directions must be a positive even integer.")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive.")

    total = _cell_array(sigma_t, cells, "sigma_t")
    scatter = _cell_array(sigma_s, cells, "sigma_s")
    if np.any(total <= 0.0):
        raise ValueError("sigma_t must be positive in every cell.")
    if np.any(scatter < 0.0):
        raise ValueError("sigma_s cannot be negative.")
    if np.any(scatter > total):
        raise ValueError("sigma_s cannot exceed sigma_t in any cell.")

    fixed_source = (
        np.ones(cells, dtype=np.float64)
        if source is None
        else _cell_array(source, cells, "source")
    )
    if np.any(fixed_source < 0.0):
        raise ValueError("source cannot be negative.")

    mu, weights = np.polynomial.legendre.leggauss(directions)
    abs_mu = np.abs(mu)
    scalar_flux = np.zeros(cells, dtype=np.float64)
    angular_flux = np.zeros((directions, cells), dtype=np.float64)
    left_edges = np.zeros((directions, cells + 1), dtype=np.float64)
    right_edges = np.zeros((directions, cells + 1), dtype=np.float64)

    relative_change = np.inf
    converged = False
    completed_iterations = 0

    for iteration in range(1, max_iterations + 1):
        previous_flux = scalar_flux.copy()
        isotropic_source = 0.5 * (scatter * scalar_flux + fixed_source)
        angular_flux.fill(0.0)

        for n in np.flatnonzero(mu > 0.0):
            coupling = 2.0 * abs_mu[n] / cell_width
            left_edges[n, 0] = 0.0
            for j in range(cells):
                right_edges[n, j] = (
                    2.0 * isotropic_source[j]
                    + (coupling - total[j]) * left_edges[n, j]
                ) / (coupling + total[j])
                angular_flux[n, j] = 0.5 * (
                    right_edges[n, j] + left_edges[n, j]
                )
                left_edges[n, j + 1] = right_edges[n, j]

        for n in np.flatnonzero(mu < 0.0):
            coupling = 2.0 * abs_mu[n] / cell_width
            right_edges[n, cells] = 0.0
            for j in range(cells - 1, -1, -1):
                left_edges[n, j] = (
                    2.0 * isotropic_source[j]
                    + (coupling - total[j]) * right_edges[n, j + 1]
                ) / (coupling + total[j])
                angular_flux[n, j] = 0.5 * (
                    right_edges[n, j + 1] + left_edges[n, j]
                )
                right_edges[n, j] = left_edges[n, j]

        scalar_flux = (angular_flux * weights[:, None]).sum(axis=0)
        relative_change = float(
            np.max(
                np.abs(scalar_flux - previous_flux)
                / np.maximum(1.0, np.abs(scalar_flux))
            )
        )
        completed_iterations = iteration
        if relative_change < tolerance:
            converged = True
            break

    return TransportResult(
        scalar_flux=scalar_flux,
        angular_flux=angular_flux,
        ordinates=mu,
        weights=weights,
        iterations=completed_iterations,
        relative_change=relative_change,
        converged=converged,
    )
