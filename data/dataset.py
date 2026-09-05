"""Oxford-IIIT Pet loading, deterministic splitting, and transforms."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPERIMENTS = ("baseline", "randaugment", "label_smoothing")


@dataclass(frozen=True)
class SplitIndices:
    """Indices for the reproducible 70/15/15 split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class DataBundle:
    """Data loaders, class names, and their source split indices."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    splits: SplitIndices


class IndexedTransformDataset(Dataset[tuple[Any, int]]):
    """Expose indexed samples with a split-specific transform."""

    def __init__(
        self,
        dataset: Sequence[tuple[Any, int]] | Dataset[tuple[Any, int]],
        indices: Sequence[int],
        transform: Callable[[Any], Any],
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), int(label)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and cuDNN."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(experiment: str) -> tuple[transforms.Compose, transforms.Compose]:
    """Build one experiment's training transform and deterministic eval transform."""

    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose from {EXPERIMENTS}")

    train_steps: list[Callable[[Any], Any]] = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    if experiment == "randaugment":
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray,
    seed: int = 42,
) -> SplitIndices:
    """Split sample indices into reproducible stratified 70/15/15 partitions."""

    labels_array = np.asarray(labels, dtype=np.int64)
    indices = np.arange(len(labels_array), dtype=np.int64)
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=labels_array,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=labels_array[temporary_indices],
    )
    return SplitIndices(
        train=np.asarray(train_indices, dtype=np.int64),
        validation=np.asarray(validation_indices, dtype=np.int64),
        test=np.asarray(test_indices, dtype=np.int64),
    )


def _validate_splits(labels: np.ndarray, splits: SplitIndices, sample_count: int) -> None:
    split_sets = [
        set(splits.train.tolist()),
        set(splits.validation.tolist()),
        set(splits.test.tolist()),
    ]
    if any(
        split_sets[left] & split_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("Train, validation, and test indices overlap")
    if set().union(*split_sets) != set(range(sample_count)):
        raise RuntimeError("Dataset split does not cover every sample exactly once")

    expected_labels = set(range(37))
    for split_name, indices in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        if set(labels[indices].tolist()) != expected_labels:
            raise RuntimeError(f"{split_name} split does not contain all 37 classes")


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    experiment: str = "baseline",
    download: bool = True,
    pin_memory: bool | None = None,
) -> DataBundle:
    """Load all images and return stratified experiment-specific data loaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    trainval_dataset = OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        download=download,
    )
    official_test_dataset = OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="category",
        download=download,
    )
    full_dataset = ConcatDataset([trainval_dataset, official_test_dataset])
    labels = np.asarray(trainval_dataset._labels + official_test_dataset._labels, dtype=np.int64)
    class_names = list(trainval_dataset.classes)

    if len(labels) != len(full_dataset):
        raise RuntimeError("Dataset labels and samples have different lengths")
    if len(class_names) != 37 or set(labels.tolist()) != set(range(37)):
        raise RuntimeError("Expected 37 Oxford-IIIT Pet labels numbered 0 through 36")

    splits = stratified_split_indices(labels, seed=seed)
    _validate_splits(labels, splits, len(full_dataset))
    train_transform, eval_transform = build_transforms(experiment)
    train_dataset = IndexedTransformDataset(full_dataset, splits.train, train_transform)
    val_dataset = IndexedTransformDataset(full_dataset, splits.validation, eval_transform)
    test_dataset = IndexedTransformDataset(full_dataset, splits.test, eval_transform)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def make_loader(
        dataset: Dataset,
        *,
        shuffle: bool,
        generator_seed: int,
        persistent: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent and num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(generator_seed),
        )

    return DataBundle(
        train_loader=make_loader(
            train_dataset,
            shuffle=True,
            generator_seed=seed,
            persistent=True,
        ),
        val_loader=make_loader(
            val_dataset,
            shuffle=False,
            generator_seed=seed + 1,
            persistent=False,
        ),
        test_loader=make_loader(
            test_dataset,
            shuffle=False,
            generator_seed=seed + 2,
            persistent=False,
        ),
        class_names=class_names,
        splits=splits,
    )
