from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _FakeOxfordPet:
    classes = [f"class_{index}" for index in range(37)]

    def __init__(self, root, split, target_types, download):
        del root, target_types, download
        per_class = 8 if split == "trainval" else 4
        self._labels = [label for label in range(37) for _ in range(per_class)]
        self._image = Image.new("RGB", (300, 280), color=(100, 120, 140))

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return self._image, self._labels[index]


class DatasetModuleTests(unittest.TestCase):
    def test_data_bundle_exposes_reproducible_splits_and_limits_persistent_workers(self):
        from data import dataset as dataset_module

        with mock.patch.object(dataset_module, "OxfordIIITPet", _FakeOxfordPet):
            first = dataset_module.build_dataloaders(
                "unused",
                batch_size=8,
                num_workers=2,
                seed=42,
                experiment="baseline",
                download=False,
                pin_memory=False,
            )
            second = dataset_module.build_dataloaders(
                "unused",
                batch_size=8,
                num_workers=2,
                seed=42,
                experiment="randaugment",
                download=False,
                pin_memory=False,
            )

        self.assertEqual(first.class_names, _FakeOxfordPet.classes)
        np.testing.assert_array_equal(first.splits.train, second.splits.train)
        np.testing.assert_array_equal(first.splits.validation, second.splits.validation)
        np.testing.assert_array_equal(first.splits.test, second.splits.test)
        self.assertTrue(first.train_loader.persistent_workers)
        self.assertFalse(first.val_loader.persistent_workers)
        self.assertFalse(first.test_loader.persistent_workers)

        split_sets = [
            set(first.splits.train.tolist()),
            set(first.splits.validation.tolist()),
            set(first.splits.test.tolist()),
        ]
        self.assertFalse(split_sets[0] & split_sets[1])
        self.assertFalse(split_sets[0] & split_sets[2])
        self.assertFalse(split_sets[1] & split_sets[2])

    def test_transforms_change_only_randaugment_training_pipeline(self):
        from data.dataset import build_transforms
        from torchvision import transforms

        baseline_train, baseline_eval = build_transforms("baseline")
        rand_train, rand_eval = build_transforms("randaugment")
        smooth_train, smooth_eval = build_transforms("label_smoothing")

        self.assertFalse(any(isinstance(item, transforms.RandAugment) for item in baseline_train.transforms))
        self.assertEqual(sum(isinstance(item, transforms.RandAugment) for item in rand_train.transforms), 1)
        self.assertEqual(str(baseline_train), str(smooth_train))
        self.assertEqual(str(baseline_eval), str(rand_eval))
        self.assertEqual(str(baseline_eval), str(smooth_eval))
        for item in baseline_eval.transforms:
            self.assertNotIsInstance(item, (transforms.RandomCrop, transforms.RandomHorizontalFlip))


class MetricModuleTests(unittest.TestCase):
    def test_evaluate_model_returns_top1_top5_macro_f1_and_prediction_arrays(self):
        from utils.metrics import evaluate_model

        logits = torch.tensor(
            [
                [9.0, 8.0, 7.0, 6.0, 5.0, 0.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [8.0, 7.0, 0.0, 6.0, 5.0, 9.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 2], dtype=torch.int64)
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)

        result = evaluate_model(
            nn.Identity(),
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            num_classes=6,
        )

        self.assertAlmostEqual(result.top1_accuracy, 1 / 3)
        self.assertAlmostEqual(result.top5_accuracy, 2 / 3)
        self.assertAlmostEqual(result.macro_f1, 1 / 6)
        self.assertEqual(result.targets.shape, (3,))
        self.assertEqual(result.predictions.shape, (3,))
        self.assertEqual(result.top5_predictions.shape, (3, 5))
        self.assertEqual(result.confidences.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result.confidences)))

    def test_confusion_pair_ranking_merges_both_error_directions(self):
        from utils.metrics import compute_confusion_tables, top_confusion_pairs

        targets = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        predictions = np.array([0, 1, 1, 0, 1, 1, 2, 2])
        counts, normalized = compute_confusion_tables(targets, predictions, num_classes=3)
        pairs = top_confusion_pairs(counts, ["A", "B", "C"], top_k=2)

        self.assertEqual(counts.shape, (3, 3))
        np.testing.assert_allclose(normalized.sum(axis=1), np.ones(3))
        self.assertEqual(pairs[0]["class_a"], "A")
        self.assertEqual(pairs[0]["class_b"], "B")
        self.assertEqual(pairs[0]["a_as_b"], 2)
        self.assertEqual(pairs[0]["b_as_a"], 1)
        self.assertEqual(pairs[0]["total_confusions"], 3)

    def test_metric_plots_are_valid_png_files(self):
        from utils.metrics import (
            save_ablation_plot,
            save_confusion_matrix,
            save_local_confusion_matrix,
            save_training_curves,
        )

        results = [
            {
                "experiment": "baseline",
                "test_top1_accuracy": 0.90,
                "test_top5_accuracy": 0.99,
                "test_macro_f1": 0.89,
            },
            {
                "experiment": "label_smoothing",
                "test_top1_accuracy": 0.92,
                "test_top5_accuracy": 0.995,
                "test_macro_f1": 0.91,
            },
        ]
        histories = {
            "baseline": [
                {"epoch": 1, "train_loss": 1.0, "val_loss": 0.8, "val_accuracy": 0.7},
                {"epoch": 2, "train_loss": 0.7, "val_loss": 0.6, "val_accuracy": 0.8},
            ],
            "label_smoothing": [
                {"epoch": 1, "train_loss": 1.1, "val_loss": 0.7, "val_accuracy": 0.75},
                {"epoch": 2, "train_loss": 0.8, "val_loss": 0.5, "val_accuracy": 0.85},
            ],
        }
        normalized = np.eye(3)
        pairs = [
            {"class_a_index": 0, "class_b_index": 1},
            {"class_a_index": 1, "class_b_index": 2},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = [
                root / "ablation.png",
                root / "curves.png",
                root / "confusion.png",
                root / "local.png",
            ]
            save_ablation_plot(results, paths[0])
            save_training_curves(histories, paths[1])
            save_confusion_matrix(normalized, ["A", "B", "C"], paths[2])
            save_local_confusion_matrix(normalized, ["A", "B", "C"], pairs, paths[3])
            for path in paths:
                self.assertGreater(path.stat().st_size, 0)
                with Image.open(path) as image:
                    image.verify()


class GradCAMModuleTests(unittest.TestCase):
    def test_import_evaluate_uses_writable_matplotlib_cache(self):
        project_root = Path(__file__).resolve().parents[1]
        environment = dict(__import__("os").environ)
        environment.pop("MPLCONFIGDIR", None)
        completed = subprocess.run(
            [sys.executable, "-c", "import evaluate"],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertNotIn("Could not save font_manager cache", completed.stdout + completed.stderr)

    def test_gradcam_is_normalized_and_removes_forward_hook(self):
        from utils.gradcam import compute_gradcam

        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 3),
        )
        target_layer = model[0]
        hook_count = len(target_layer._forward_hooks)
        cam = compute_gradcam(
            model,
            target_layer,
            torch.rand(1, 3, 16, 16),
            target_class=1,
        )

        self.assertEqual(cam.shape, (16, 16))
        self.assertTrue(torch.isfinite(cam).all())
        self.assertGreaterEqual(cam.min().item(), 0.0)
        self.assertLessEqual(cam.max().item(), 1.0)
        self.assertEqual(len(target_layer._forward_hooks), hook_count)

    def test_gradcam_example_selection_prefers_top_confusion_pair(self):
        from utils.gradcam import select_gradcam_examples

        correct, incorrect = select_gradcam_examples(
            np.array([0, 0, 1, 1, 2]),
            np.array([0, 1, 1, 0, 1]),
            np.array([0.70, 0.80, 0.60, 0.95, 0.99]),
            top_pair=(0, 1),
        )
        self.assertEqual(correct, 0)
        self.assertEqual(incorrect, 3)


class ModelModuleTests(unittest.TestCase):
    def test_load_checkpoint_validates_fields_and_restores_model(self):
        from models.model import build_model, load_checkpoint

        model = build_model(num_classes=3, pretrained=False)
        checkpoint = {
            "epoch": 2,
            "experiment": "baseline",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_accuracy": 0.75,
            "class_names": ["a", "b", "c"],
            "config": {"seed": 42},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.pth"
            torch.save(checkpoint, path)
            loaded_model, loaded_checkpoint = load_checkpoint(
                path,
                torch.device("cpu"),
                num_classes=3,
            )

        self.assertEqual(loaded_model.fc.out_features, 3)
        self.assertEqual(loaded_checkpoint["experiment"], "baseline")
        for expected, actual in zip(model.parameters(), loaded_model.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_load_checkpoint_rejects_missing_required_fields(self):
        from models.model import load_checkpoint

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid.pth"
            torch.save({"model_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_checkpoint(path, torch.device("cpu"), num_classes=37)


class CommandLineContractTests(unittest.TestCase):
    def test_train_defaults_reproduce_all_three_experiments(self):
        import train

        args = train.parse_args([])
        self.assertEqual(args.experiments, ["baseline", "randaugment", "label_smoothing"])
        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.artifact_dir, Path("artifacts"))
        self.assertEqual(args.log_dir, Path("runs"))
        self.assertEqual(args.epochs, 15)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.num_workers, 4)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.device, "cuda")


class TrainingRunnerTests(unittest.TestCase):
    def test_label_smoothing_changes_only_training_criterion(self):
        import train

        train_criterion, val_criterion = train.build_criteria("label_smoothing")
        self.assertAlmostEqual(train_criterion.label_smoothing, 0.1)
        self.assertEqual(val_criterion.label_smoothing, 0.0)

        baseline_train, baseline_val = train.build_criteria("baseline")
        self.assertEqual(baseline_train.label_smoothing, 0.0)
        self.assertEqual(baseline_val.label_smoothing, 0.0)

    def test_training_runner_writes_checkpoint_history_summary_and_events_without_test_access(self):
        import train
        from data.dataset import SplitIndices

        features = torch.eye(4, dtype=torch.float32).repeat(2, 1)
        targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64)
        loader = DataLoader(TensorDataset(features, targets), batch_size=4, shuffle=False)

        class ForbiddenTestLoader:
            def __iter__(self):
                raise AssertionError("train.py must not access the test split")

        bundle = SimpleNamespace(
            train_loader=loader,
            val_loader=loader,
            test_loader=ForbiddenTestLoader(),
            class_names=[f"class_{index}" for index in range(37)],
            splits=SplitIndices(
                train=np.arange(8),
                validation=np.arange(8, 16),
                test=np.arange(16, 24),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args = argparse.Namespace(
                experiments=["baseline"],
                data_dir=root / "data",
                artifact_dir=root / "artifacts",
                log_dir=root / "runs",
                epochs=1,
                batch_size=4,
                lr=1e-3,
                weight_decay=1e-4,
                num_workers=0,
                seed=42,
                device="cpu",
            )
            with (
                mock.patch.object(train, "build_dataloaders", return_value=bundle),
                mock.patch.object(train, "build_model", return_value=nn.Linear(4, 37)),
            ):
                rows = train.run_training(args, torch.device("cpu"))

            checkpoint_path = root / "artifacts" / "checkpoints" / "baseline.pth"
            history_path = root / "artifacts" / "histories" / "baseline.csv"
            summary_path = root / "artifacts" / "training_summary.csv"
            event_files = list((root / "runs" / "baseline").glob("events.out.tfevents.*"))

            self.assertEqual(len(rows), 1)
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(
                set(checkpoint),
                {
                    "epoch",
                    "experiment",
                    "model_state_dict",
                    "optimizer_state_dict",
                    "val_accuracy",
                    "class_names",
                    "config",
                },
            )
            with history_path.open(encoding="utf-8", newline="") as stream:
                history_rows = list(csv.DictReader(stream))
            self.assertEqual(len(history_rows), 1)
            with summary_path.open(encoding="utf-8", newline="") as stream:
                summary_rows = list(csv.DictReader(stream))
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["experiment"], "baseline")
            self.assertEqual(len(event_files), 1)

    def test_subset_training_preserves_other_experiments_in_summary(self):
        import train

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = Path(tmp_dir) / "artifacts"
            artifact_dir.mkdir()
            baseline_row = {
                "experiment": "baseline",
                "changed_variable": "none",
                "seed": 42,
                "best_epoch": 9,
                "best_val_accuracy": 0.92,
                "train_seconds": 214.0,
                "checkpoint": "artifacts/checkpoints/baseline.pth",
            }
            with (artifact_dir / "training_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=train.TRAINING_SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerow(baseline_row)
            args = argparse.Namespace(
                experiments=["label_smoothing"],
                data_dir=Path("data"),
                artifact_dir=artifact_dir,
                log_dir=Path(tmp_dir) / "runs",
                epochs=1,
                batch_size=2,
                lr=1e-4,
                weight_decay=1e-4,
                num_workers=0,
                seed=42,
                device="cpu",
            )
            label_row = {
                **baseline_row,
                "experiment": "label_smoothing",
                "changed_variable": "training_loss",
                "best_epoch": 1,
                "best_val_accuracy": 0.93,
                "checkpoint": "artifacts/checkpoints/label_smoothing.pth",
            }
            with mock.patch.object(train, "_run_one_experiment", return_value=label_row):
                requested = train.run_training(args, torch.device("cpu"))

            with (artifact_dir / "training_summary.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual([row["experiment"] for row in requested], ["label_smoothing"])
            self.assertEqual(
                [row["experiment"] for row in saved],
                ["baseline", "label_smoothing"],
            )

    def test_evaluate_defaults_use_the_shared_artifact_layout(self):
        import evaluate

        args = evaluate.parse_args([])
        self.assertEqual(args.experiments, ["baseline", "randaugment", "label_smoothing"])
        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.artifact_dir, Path("artifacts"))
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.num_workers, 4)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.device, "cuda")


class EvaluationRunnerTests(unittest.TestCase):
    def test_evaluation_rejects_checkpoint_metadata_mismatch(self):
        import evaluate

        training_row = {
            "experiment": "baseline",
            "best_epoch": "9",
            "best_val_accuracy": "0.9274",
        }
        checkpoint = {
            "experiment": "randaugment",
            "epoch": 9,
            "val_accuracy": 0.9274,
        }
        with self.assertRaisesRegex(ValueError, "experiment metadata"):
            evaluate.validate_checkpoint_metadata(
                "baseline", training_row, checkpoint
            )

    def test_evaluation_runner_writes_complete_outputs_and_selects_by_validation(self):
        import evaluate
        from data.dataset import SplitIndices

        class TinyBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2 = nn.Conv2d(3, 4, kernel_size=3, padding=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.conv2(inputs))

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer4 = nn.Sequential(TinyBlock())
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(4, 3)
                with torch.no_grad():
                    self.fc.weight.fill_(0.1)
                    self.fc.bias.copy_(torch.tensor([1.0, 0.0, -1.0]))

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.fc(self.pool(self.layer4(inputs)).flatten(1))

        generator = torch.Generator().manual_seed(11)
        images = torch.rand(6, 3, 16, 16, generator=generator)
        targets = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.int64)
        loader = DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=False)
        bundle = SimpleNamespace(
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            class_names=["A", "B", "C"],
            splits=SplitIndices(
                train=np.arange(6),
                validation=np.arange(6, 12),
                test=np.arange(12, 18),
            ),
        )
        validation_scores = {
            "baseline": 0.80,
            "randaugment": 0.95,
            "label_smoothing": 0.99,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_dir = root / "artifacts"
            histories_dir = artifact_dir / "histories"
            checkpoints_dir = artifact_dir / "checkpoints"
            histories_dir.mkdir(parents=True)
            checkpoints_dir.mkdir(parents=True)
            summary_rows = []
            for index, experiment in enumerate(
                ("baseline", "randaugment", "label_smoothing"), start=1
            ):
                (checkpoints_dir / f"{experiment}.pth").touch()
                with (histories_dir / f"{experiment}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=(
                            "epoch",
                            "train_loss",
                            "train_accuracy",
                            "val_loss",
                            "val_accuracy",
                        ),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "epoch": 1,
                            "train_loss": 1.0,
                            "train_accuracy": 0.5,
                            "val_loss": 0.9,
                            "val_accuracy": validation_scores[experiment],
                        }
                    )
                summary_rows.append(
                    {
                        "experiment": experiment,
                        "changed_variable": "none" if index == 1 else experiment,
                        "seed": 42,
                        "best_epoch": index,
                        "best_val_accuracy": validation_scores[experiment],
                        "train_seconds": 10.0 * index,
                        "checkpoint": str(checkpoints_dir / f"{experiment}.pth"),
                    }
                )
            with (artifact_dir / "training_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)

            args = argparse.Namespace(
                experiments=["baseline", "randaugment", "label_smoothing"],
                data_dir=root / "data",
                artifact_dir=artifact_dir,
                batch_size=2,
                num_workers=0,
                seed=42,
                device="cpu",
            )

            def fake_load(path, device, num_classes):
                experiment = Path(path).stem
                return TinyModel().to(device), {
                    "epoch": ("baseline", "randaugment", "label_smoothing").index(experiment) + 1,
                    "experiment": experiment,
                    "val_accuracy": validation_scores[experiment],
                    "class_names": bundle.class_names,
                }

            with (
                mock.patch.object(evaluate, "build_dataloaders", return_value=bundle),
                mock.patch.object(evaluate, "load_checkpoint", side_effect=fake_load),
            ):
                rows = evaluate.run_evaluation(args, torch.device("cpu"))

            self.assertEqual(len(rows), 3)
            self.assertTrue(all("test_top5_accuracy" in row for row in rows))
            with (artifact_dir / "results.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                result_rows = list(csv.DictReader(stream))
            self.assertEqual(len(result_rows), 3)
            for experiment in ("baseline", "randaugment", "label_smoothing"):
                prediction_path = artifact_dir / "predictions" / f"{experiment}.npz"
                with np.load(prediction_path) as predictions:
                    self.assertEqual(predictions["targets"].shape, (6,))
                    self.assertEqual(predictions["top5_predictions"].shape, (6, 3))

            expected_figures = {
                "training_curves.png",
                "ablation_metrics.png",
                "confusion_matrix.png",
                "confusion_matrix_top_pairs.png",
                "gradcam_correct.png",
                "gradcam_incorrect.png",
            }
            self.assertEqual(
                {path.name for path in (artifact_dir / "figures").glob("*.png")},
                expected_figures,
            )
            for path in (artifact_dir / "figures").glob("*.png"):
                with Image.open(path) as image:
                    image.verify()
            summary = json.loads(
                (artifact_dir / "analysis_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["selected_experiment"], "label_smoothing")
            self.assertEqual(summary["selection_rule"], "maximum best_val_accuracy")


if __name__ == "__main__":
    unittest.main()
