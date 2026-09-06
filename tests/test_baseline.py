"""Compatibility checks for the Day 1 entry point after modularization."""

import random
import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import baseline


class BaselineCompatibilityTests(unittest.TestCase):
    def test_split_keeps_the_legacy_tuple_shape(self):
        labels = np.repeat(np.arange(4), 20)
        first = baseline.stratified_split_indices(labels, seed=42)
        second = baseline.stratified_split_indices(labels, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(tuple(map(len, first)), (56, 12, 12))
        train, validation, test = map(set, first)
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(train | validation | test, set(range(80)))

    def test_transforms_and_indexed_dataset_still_delegate_to_shared_code(self):
        pixels = np.arange(300 * 280 * 3, dtype=np.uint8).reshape(300, 280, 3)
        image = Image.fromarray(pixels, mode="RGB")
        train_transform, eval_transform = baseline.build_transforms()

        self.assertEqual(tuple(train_transform(image).shape), (3, 224, 224))
        self.assertEqual(tuple(eval_transform(image).shape), (3, 224, 224))
        self.assertFalse(
            any(
                isinstance(step, (transforms.RandomCrop, transforms.RandomHorizontalFlip))
                for step in eval_transform.transforms
            )
        )
        source = [(2, 0), (3, 1), (5, 2)]
        dataset = baseline.IndexedTransformDataset(source, [2, 0], lambda value: value * 10)
        self.assertEqual(dataset[0], (50, 2))

    def test_train_and_evaluate_compatibility_functions_are_sample_weighted(self):
        features = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
        labels = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        train_loss, train_accuracy = baseline.train_one_epoch(
            model, loader, criterion, optimizer, torch.device("cpu")
        )
        eval_loss, eval_accuracy = baseline.evaluate(
            model, loader, criterion, torch.device("cpu")
        )

        self.assertAlmostEqual(train_loss, eval_loss, places=6)
        self.assertAlmostEqual(train_accuracy, 2 / 3, places=6)
        self.assertAlmostEqual(eval_accuracy, 2 / 3, places=6)

    def test_cli_selects_only_baseline_and_seed_remains_reproducible(self):
        args = baseline.parse_args([])
        self.assertEqual(args.experiments, ["baseline"])
        self.assertEqual(args.artifact_dir.as_posix(), "artifacts")
        self.assertEqual(args.epochs, 15)

        baseline.seed_everything(42)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        baseline.seed_everything(42)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
