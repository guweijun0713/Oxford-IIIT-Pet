"""Compatibility checks for the Day 2 entry points after modularization."""

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import day2_experiments
import day2_visualize


class Day2CompatibilityTests(unittest.TestCase):
    def test_cli_selects_the_two_single_variable_experiments(self):
        args = day2_experiments.parse_args([])
        self.assertEqual(args.experiments, ["randaugment", "label_smoothing"])
        self.assertEqual(args.artifact_dir.as_posix(), "artifacts")
        self.assertEqual(args.epochs, 15)

    def test_randaugment_and_label_smoothing_delegate_to_shared_configuration(self):
        randaugment = day2_experiments.build_train_transform("randaugment")
        baseline = day2_experiments.build_train_transform("label_smoothing")
        self.assertEqual(
            sum(isinstance(step, transforms.RandAugment) for step in randaugment.transforms),
            1,
        )
        self.assertFalse(
            any(isinstance(step, transforms.RandAugment) for step in baseline.transforms)
        )
        train_loss, eval_loss = day2_experiments.build_criteria("label_smoothing")
        self.assertEqual(train_loss.label_smoothing, 0.1)
        self.assertEqual(eval_loss.label_smoothing, 0.0)

    def test_prediction_evaluator_exposes_top1_top5_and_macro_f1(self):
        logits = torch.tensor([[3.0, 2.0, 1.0], [0.0, 2.0, 3.0]])
        labels = torch.tensor([0, 1])
        result = day2_experiments.evaluate_with_predictions(
            nn.Identity(),
            DataLoader(TensorDataset(logits, labels), batch_size=2),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            num_classes=3,
        )
        self.assertEqual(result.top1_accuracy, 0.5)
        self.assertEqual(result.top5_accuracy, 1.0)
        self.assertTrue(np.isfinite(result.macro_f1))

    def test_visualization_module_reexports_analysis_functions(self):
        counts, normalized = day2_visualize.compute_confusion_tables(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]),
            num_classes=2,
        )
        pairs = day2_visualize.top_confusion_pairs(counts, ["A", "B"], top_k=1)
        self.assertEqual(counts.shape, (2, 2))
        np.testing.assert_allclose(normalized.sum(axis=1), np.ones(2))
        self.assertEqual(pairs[0]["total_confusions"], 2)


if __name__ == "__main__":
    unittest.main()
