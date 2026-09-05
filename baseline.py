"""Backward-compatible Baseline entry point.

The implementation lives in the modular ``data``, ``models``, ``utils`` and
``train`` modules. Running this file trains only the Baseline experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import (
    DataBundle,
    IndexedTransformDataset,
    build_dataloaders as _build_dataloaders,
    build_transforms as _build_transforms,
    seed_everything,
    stratified_split_indices as _stratified_split_indices,
)
from models.model import build_model
from train import parse_args as _parse_train_args
from train import run_training, train_one_epoch
from utils.metrics import evaluate_model


def build_transforms():
    """Return the original Baseline train/evaluation transforms."""

    return _build_transforms("baseline")


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Return the legacy tuple form while delegating the split calculation."""

    splits = _stratified_split_indices(labels, seed=seed)
    return splits.train.tolist(), splits.validation.tolist(), splits.test.tolist()


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Return the four-item Day 1 loader tuple using the shared data module."""

    bundle: DataBundle = _build_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        experiment="baseline",
        pin_memory=pin_memory,
    )
    return bundle.train_loader, bundle.val_loader, bundle.test_loader, bundle.class_names


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return the Day 1 ``(loss, Top-1)`` evaluation tuple."""

    num_classes = getattr(getattr(model, "fc", None), "out_features", None)
    if num_classes is None:
        sample = next(iter(loader))[0]
        with torch.inference_mode():
            num_classes = int(model(sample[:1].to(device)).shape[1])
    result = evaluate_model(model, loader, criterion, device, num_classes=num_classes)
    return result.loss, result.top1_accuracy


def parse_args(argv: Sequence[str] | None = None):
    """Parse the current shared CLI with Baseline selected by default."""

    arguments = list(argv) if argv is not None else sys.argv[1:]
    return _parse_train_args(["--experiments", "baseline", *arguments])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    run_training(args, device)


if __name__ == "__main__":
    main()
