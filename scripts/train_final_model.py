"""Train and evaluate the retained material-aware DeepONet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeponet_1d.experiments import (  # noqa: E402
    ExperimentConfig,
    run_experiment,
    subset_dataset,
)
from deeponet_1d.learning import load_array_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-samples", type=int)
    parser.add_argument("--loss", choices=("mse", "relative_l2"), default="relative_l2")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    training = load_array_dataset(args.data_dir / "train.npz")
    if args.training_samples is not None:
        training = subset_dataset(training, args.training_samples)
    validation = load_array_dataset(args.data_dir / "validation.npz")
    tests = {}
    if not args.validation_only:
        tests = {
            "test_in_distribution": load_array_dataset(
                args.data_dir / "test_in_distribution.npz"
            ),
            "test_three_layer_ood": load_array_dataset(
                args.data_dir / "test_three_layer_ood.npz"
            ),
        }
    result = run_experiment(
        training=training,
        validation=validation,
        tests=tests,
        output_directory=args.output_dir,
        config=ExperimentConfig(
            architecture="separate_branch",
            loss=args.loss,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
