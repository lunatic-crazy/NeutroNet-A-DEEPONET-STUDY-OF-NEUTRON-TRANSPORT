"""Generate a source-shape OOD benchmark with families absent from training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeponet_1d.dataset import DatasetSplit, save_dataset  # noqa: E402
from deeponet_1d.materials import generate_layered_materials  # noqa: E402
from deeponet_1d.transport import solve_sn_slab  # noqa: E402


FAMILIES = np.array(
    (
        "linear_ramp",
        "exponential_layer",
        "triangular_tent",
        "sawtooth",
        "chirp",
        "lorentzian_peak",
    ),
    dtype="U20",
)


def unseen_source_samples(
    *,
    cells: int,
    cell_width: float,
    samples: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.str_]]:
    """Return balanced sources with shapes not used by mixed-source training."""
    if cells <= 0 or cell_width <= 0.0 or samples <= 0:
        raise ValueError("cells, cell_width, and samples must be positive.")
    rng = np.random.default_rng(seed)
    x = (np.arange(cells, dtype=np.float64) + 0.5) * cell_width
    length = cells * cell_width
    t = x / length
    labels = np.resize(FAMILIES, samples).copy()
    rng.shuffle(labels)
    sources = np.empty((samples, cells), dtype=np.float64)

    for index, family in enumerate(labels):
        if family == "linear_ramp":
            left, right = rng.uniform(1.0, 10.0, size=2)
            sources[index] = left + (right - left) * t
        elif family == "exponential_layer":
            background = rng.uniform(1.0, 3.0)
            amplitude = rng.uniform(4.0, 9.0)
            scale = rng.uniform(0.05, 0.30)
            coordinate = t if rng.random() < 0.5 else 1.0 - t
            sources[index] = background + amplitude * np.exp(-coordinate / scale)
        elif family == "triangular_tent":
            center = rng.uniform(0.15, 0.85)
            half_width = rng.uniform(0.08, 0.35)
            background = rng.uniform(1.0, 3.0)
            amplitude = rng.uniform(4.0, 9.0)
            sources[index] = background + amplitude * np.maximum(
                1.0 - np.abs(t - center) / half_width, 0.0
            )
        elif family == "sawtooth":
            mean = rng.uniform(4.0, 8.0)
            amplitude = rng.uniform(1.0, 3.0)
            cycles = rng.uniform(0.75, 2.5)
            phase = rng.uniform(0.0, 1.0)
            wave = 2.0 * np.mod(cycles * t + phase, 1.0) - 1.0
            sources[index] = mean + amplitude * wave
        elif family == "chirp":
            mean = rng.uniform(4.0, 8.0)
            amplitude = rng.uniform(1.0, 3.0)
            start_frequency = rng.uniform(0.25, 1.0)
            end_frequency = rng.uniform(2.0, 5.0)
            phase = 2.0 * np.pi * (
                start_frequency * t
                + 0.5 * (end_frequency - start_frequency) * t**2
            ) + rng.uniform(0.0, 2.0 * np.pi)
            sources[index] = mean + amplitude * np.sin(phase)
        else:
            center = rng.uniform(0.10, 0.90)
            width = rng.uniform(0.02, 0.12)
            background = rng.uniform(1.0, 3.0)
            amplitude = rng.uniform(4.0, 9.0)
            sources[index] = background + amplitude / (
                1.0 + ((t - center) / width) ** 2
            )

    return x, sources, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-family", type=int, default=100)
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--cell-width", type=float, default=0.1)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    samples = args.samples_per_family * len(FAMILIES)
    cells = int(round(args.length / args.cell_width))
    seed_sequence = np.random.SeedSequence(args.seed)
    source_seed, material_seed = (
        int(child.generate_state(1)[0]) for child in seed_sequence.spawn(2)
    )
    x, sources, labels = unseen_source_samples(
        cells=cells,
        cell_width=args.cell_width,
        samples=samples,
        seed=source_seed,
    )
    materials = generate_layered_materials(
        samples=samples,
        cells=cells,
        min_layers=1,
        max_layers=2,
        seed=material_seed,
    )
    flux = np.empty_like(sources)
    iterations = np.empty(samples, dtype=np.int64)
    for index in range(samples):
        result = solve_sn_slab(
            length=args.length,
            cell_width=args.cell_width,
            sigma_t=materials.sigma_t[index],
            sigma_s=materials.sigma_s[index],
            source=sources[index],
            directions=args.directions,
        )
        if not result.converged:
            raise RuntimeError(f"Transport solve {index} did not converge.")
        flux[index] = result.scalar_flux
        iterations[index] = result.iterations

    destination = save_dataset(
        DatasetSplit(
            x=x,
            source=sources,
            sigma_t=materials.sigma_t,
            sigma_s=materials.sigma_s,
            scalar_flux=flux,
            layer_count=materials.layer_count,
            interfaces=materials.interfaces,
            solver_iterations=iterations,
            source_family=labels,
        ),
        args.output,
    )
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Saved {samples} cases to {destination}")
    print(dict(zip(unique.tolist(), counts.tolist(), strict=True)))
    print(f"Mean reference iterations: {iterations.mean():.1f}")


if __name__ == "__main__":
    main()
