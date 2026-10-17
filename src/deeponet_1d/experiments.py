"""Reproducible training and evaluation for material-aware DeepONets."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .learning import (
    ArrayDataset,
    denormalize_flux,
    fit_normalization,
    normalize_inputs,
    regression_metrics,
)
from .publication import (
    SeparateBranchDeepONet,
    bootstrap_mean_interval,
    grouped_regression_metrics,
    model_parameter_count,
    per_sample_relative_l2,
    regime_labels,
    stratified_error_summary,
)


@dataclass(frozen=True)
class ExperimentConfig:
    architecture: str = "separate_branch"
    loss: str = "relative_l2"
    latent_dim: int = 128
    batch_size: int = 32
    epochs: int = 1_000
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-6
    patience: int = 150
    seed: int = 42
    device: str = "auto"


def subset_dataset(dataset: ArrayDataset, samples: int) -> ArrayDataset:
    if not 1 <= samples <= len(dataset.source):
        raise ValueError("Requested subset size is outside the dataset.")
    return ArrayDataset(
        x=dataset.x,
        source=dataset.source[:samples],
        sigma_t=dataset.sigma_t[:samples],
        sigma_s=dataset.sigma_s[:samples],
        scalar_flux=dataset.scalar_flux[:samples],
        interfaces=dataset.interfaces[:samples],
        source_family=(
            dataset.source_family[:samples]
            if dataset.source_family is not None
            else None
        ),
    )


def build_model(config: ExperimentConfig, cells: int) -> nn.Module:
    if config.architecture != "separate_branch":
        raise ValueError("Only the retained separate-branch architecture is supported.")
    return SeparateBranchDeepONet(cells=cells, latent_dim=config.latent_dim)


def _loss(
    prediction: Tensor,
    target: Tensor,
    *,
    loss_name: str,
    flux_mean: float,
    flux_std: float,
) -> Tensor:
    if loss_name == "mse":
        return torch.mean((prediction - target) ** 2)
    if loss_name == "relative_l2":
        prediction_physical = prediction * flux_std + flux_mean
        target_physical = target * flux_std + flux_mean
        numerator = torch.sum((prediction_physical - target_physical) ** 2, dim=1)
        denominator = torch.clamp(torch.sum(target_physical**2, dim=1), min=1.0e-12)
        return torch.mean(numerator / denominator)
    raise ValueError(f"Unknown loss: {loss_name}")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return selected


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(model, branch, trunk, device, batch_size):
    model.eval()
    outputs = []
    coordinates = torch.from_numpy(trunk).to(device)
    with torch.no_grad():
        for start in range(0, len(branch), batch_size):
            inputs = torch.from_numpy(branch[start : start + batch_size]).to(device)
            outputs.append(model(inputs, coordinates).cpu().numpy())
    return np.concatenate(outputs)


def run_experiment(
    *,
    training: ArrayDataset,
    validation: ArrayDataset,
    tests: dict[str, ArrayDataset],
    output_directory: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Train the retained model and save publication-oriented statistics."""
    _seed_everything(config.seed)
    selected_device = _device(config.device)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    statistics = fit_normalization(training)
    train_branch, trunk, train_target = normalize_inputs(training, statistics)
    validation_branch, validation_trunk, validation_target = normalize_inputs(
        validation, statistics
    )
    if not np.array_equal(trunk, validation_trunk):
        raise ValueError("All experiments require an identical coordinate grid.")

    model = build_model(config, training.x.size).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(10, config.patience // 5)
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_branch), torch.from_numpy(train_target)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=selected_device.type == "cuda",
    )
    coordinates = torch.from_numpy(trunk).to(selected_device)
    validation_inputs = torch.from_numpy(validation_branch).to(selected_device)
    validation_targets = torch.from_numpy(validation_target).to(selected_device)

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        accumulated = 0.0
        examples = 0
        for inputs, targets in loader:
            inputs = inputs.to(selected_device, non_blocking=True)
            targets = targets.to(selected_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(
                model(inputs, coordinates),
                targets,
                loss_name=config.loss,
                flux_mean=statistics.scalar_flux.mean,
                flux_std=statistics.scalar_flux.std,
            )
            loss.backward()
            optimizer.step()
            accumulated += float(loss.detach()) * len(inputs)
            examples += len(inputs)
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _loss(
                    model(validation_inputs, coordinates),
                    validation_targets,
                    loss_name=config.loss,
                    flux_mean=statistics.scalar_flux.mean,
                    flux_std=statistics.scalar_flux.std,
                )
            )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "training_loss": accumulated / examples,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), destination / "model.pt")
        else:
            stale += 1
            if stale >= config.patience:
                break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(destination / "model.pt", map_location=selected_device))

    with (destination / "training_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    np.savez(
        destination / "normalization.npz",
        Q_mean=statistics.source.mean,
        Q_std=statistics.source.std,
        Sigma_t_mean=statistics.sigma_t.mean,
        Sigma_t_std=statistics.sigma_t.std,
        Sigma_s_mean=statistics.sigma_s.mean,
        Sigma_s_std=statistics.sigma_s.std,
        phi_mean=statistics.scalar_flux.mean,
        phi_std=statistics.scalar_flux.std,
        x_min=statistics.x_min,
        x_max=statistics.x_max,
    )
    configuration = asdict(config)
    configuration.update(
        {
            "device_used": str(selected_device),
            "gpu_name": (
                torch.cuda.get_device_name(selected_device)
                if selected_device.type == "cuda"
                else None
            ),
            "training_samples": len(training.source),
            "parameter_count": model_parameter_count(model),
        }
    )
    (destination / "config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    results: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "splits": {},
    }
    predictions = {}
    for split_index, (name, split) in enumerate(
        {"validation": validation, **tests}.items()
    ):
        branch, split_trunk, _ = normalize_inputs(split, statistics)
        prediction_started = time.perf_counter()
        normalized_prediction = _predict(
            model, branch, split_trunk, selected_device, config.batch_size
        )
        prediction_seconds = time.perf_counter() - prediction_started
        physical_prediction = denormalize_flux(
            normalized_prediction, statistics
        ).astype(np.float64)
        errors = per_sample_relative_l2(physical_prediction, split.scalar_flux)
        ci_low, ci_high = bootstrap_mean_interval(
            errors, seed=config.seed + split_index
        )
        metrics = regression_metrics(
            physical_prediction, split.scalar_flux, split.interfaces
        )
        if split.source_family is not None:
            metrics["by_source_family"] = grouped_regression_metrics(
                physical_prediction,
                split.scalar_flux,
                split.interfaces,
                split.source_family,
                seed=config.seed + split_index + 100,
            )
        metrics.update(
            {
                "mean_relative_l2_ci95_low": ci_low,
                "mean_relative_l2_ci95_high": ci_high,
                "prediction_seconds": prediction_seconds,
                "microseconds_per_sample": 1.0e6
                * prediction_seconds
                / len(split.source),
                "by_scattering_regime": stratified_error_summary(
                    errors,
                    regime_labels(split.sigma_t, split.sigma_s),
                    seed=config.seed + split_index,
                ),
            }
        )
        results["splits"][name] = metrics
        predictions[f"{name}_prediction"] = physical_prediction
        predictions[f"{name}_truth"] = split.scalar_flux
        predictions[f"{name}_relative_l2"] = errors
    (destination / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    np.savez_compressed(destination / "predictions.npz", **predictions)
    return {"config": configuration, "metrics": results}
