"""Generate the corrected final-model training and evaluation datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeponet_1d import generate_dataset, save_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/final_model"))
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--ood-samples", type=int, default=100)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20260825)
    parser.add_argument(
        "--source-distribution",
        choices=("grf", "mixed"),
        default="grf",
    )
    args = parser.parse_args()
    specifications = (
        ("train", args.train_samples, 1, 2, args.base_seed),
        ("validation", args.validation_samples, 1, 2, args.base_seed + 1),
        ("test_in_distribution", args.test_samples, 1, 2, args.base_seed + 2),
        ("test_three_layer_ood", args.ood_samples, 3, 3, args.base_seed + 3),
    )
    for name, samples, min_layers, max_layers, seed in specifications:
        dataset = generate_dataset(
            samples=samples,
            min_layers=min_layers,
            max_layers=max_layers,
            directions=args.directions,
            source_distribution=args.source_distribution,
            seed=seed,
        )
        destination = save_dataset(dataset, args.output_dir / f"{name}.npz")
        print(
            f"{name}: {samples} samples -> {destination} "
            f"(mean iterations {dataset.solver_iterations.mean():.1f})"
        )


if __name__ == "__main__":
    main()
