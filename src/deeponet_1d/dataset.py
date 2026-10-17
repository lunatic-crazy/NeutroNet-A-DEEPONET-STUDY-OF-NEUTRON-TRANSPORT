"""Reproducible heterogeneous transport dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .materials import generate_layered_materials
from .sources import gaussian_random_field_sources, mixed_source_samples
from .transport import solve_sn_slab


@dataclass(frozen=True)
class DatasetSplit:
    x: NDArray[np.float64]
    source: NDArray[np.float64]
    sigma_t: NDArray[np.float64]
    sigma_s: NDArray[np.float64]
    scalar_flux: NDArray[np.float64]
    layer_count: NDArray[np.int64]
    interfaces: NDArray[np.int64]
    solver_iterations: NDArray[np.int64]
    source_family: NDArray[np.str_] | None = None


def generate_dataset(
    *,
    samples: int,
    length: float = 10.0,
    cell_width: float = 0.1,
    directions: int = 32,
    min_layers: int = 1,
    max_layers: int = 2,
    sigma_t_range: tuple[float, float] = (0.2, 2.0),
    scattering_ratio_range: tuple[float, float] = (0.05, 0.95),
    source_mean: float = 5.0,
    source_length_scale: float = 0.1,
    source_variance: float = 1.0,
    source_distribution: str = "grf",
    seed: int = 42,
) -> DatasetSplit:
    """Generate sources, materials, and reference scalar-flux solutions."""
    cells = int(round(length / cell_width))
    seed_sequence = np.random.SeedSequence(seed)
    source_seed, material_seed = [
        int(child.generate_state(1)[0]) for child in seed_sequence.spawn(2)
    ]

    if source_distribution == "grf":
        x, sources = gaussian_random_field_sources(
            cells=cells, cell_width=cell_width, samples=samples,
            mean=source_mean, length_scale=source_length_scale,
            variance=source_variance, seed=source_seed,
        )
        source_families = None
    elif source_distribution == "mixed":
        x, sources, source_families = mixed_source_samples(
            cells=cells, cell_width=cell_width, samples=samples, seed=source_seed)
    else:
        raise ValueError("source_distribution must be 'grf' or 'mixed'.")
    materials = generate_layered_materials(
        samples=samples,
        cells=cells,
        min_layers=min_layers,
        max_layers=max_layers,
        sigma_t_range=sigma_t_range,
        scattering_ratio_range=scattering_ratio_range,
        seed=material_seed,
    )

    flux = np.empty((samples, cells), dtype=np.float64)
    iterations = np.empty(samples, dtype=np.int64)
    for index in range(samples):
        result = solve_sn_slab(
            length=length,
            cell_width=cell_width,
            sigma_t=materials.sigma_t[index],
            sigma_s=materials.sigma_s[index],
            source=sources[index],
            directions=directions,
        )
        if not result.converged:
            raise RuntimeError(f"Transport solve {index} did not converge.")
        flux[index] = result.scalar_flux
        iterations[index] = result.iterations

    return DatasetSplit(
        x=x,
        source=sources,
        sigma_t=materials.sigma_t,
        sigma_s=materials.sigma_s,
        scalar_flux=flux,
        layer_count=materials.layer_count,
        interfaces=materials.interfaces,
        solver_iterations=iterations,
        source_family=source_families,
    )


def save_dataset(dataset: DatasetSplit, path: str | Path) -> Path:
    """Save a dataset as a compressed, self-describing NPZ archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        x=dataset.x,
        Q=dataset.source,
        Sigma_t=dataset.sigma_t,
        Sigma_s=dataset.sigma_s,
        Sigma_a=dataset.sigma_t - dataset.sigma_s,
        phi=dataset.scalar_flux,
        layer_count=dataset.layer_count,
        interfaces=dataset.interfaces,
        solver_iterations=dataset.solver_iterations,
    )
    if dataset.source_family is not None:
        arrays["source_family"] = dataset.source_family
    np.savez_compressed(destination, **arrays)
    return destination
