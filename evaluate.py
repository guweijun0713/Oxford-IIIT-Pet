"""Independent evaluation and visualization entry point."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from data.dataset import DataBundle, build_dataloaders
from models.model import load_checkpoint
from utils.gradcam import compute_gradcam, save_gradcam_figure, select_gradcam_examples
from utils.metrics import (
    EvaluationResult,
    compute_confusion_tables,
    evaluate_model,
    save_ablation_plot,
    save_confusion_matrix,
    save_local_confusion_matrix,
    save_training_curves,
    top_confusion_pairs,
)

EXPERIMENTS = ("baseline", "randaugment", "label_smoothing")
RESULT_FIELDS = (
    "experiment",
    "changed_variable",
    "seed",
    "best_epoch",
    "best_val_accuracy",
    "test_loss",
    "test_top1_accuracy",
    "test_top5_accuracy",
    "test_macro_f1",
    "train_seconds",
    "checkpoint",
    "reused_checkpoint",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def validate_checkpoint_metadata(
    experiment: str,
    training_row: dict[str, str],
    checkpoint: dict[str, Any],
) -> None:
    """Reject swapped or stale checkpoints before reporting test metrics."""

    if checkpoint.get("experiment") != experiment:
        raise ValueError(
            f"{experiment} checkpoint experiment metadata is "
            f"{checkpoint.get('experiment')!r}"
        )
    expected_epoch = int(float(training_row["best_epoch"]))
    if int(checkpoint["epoch"]) != expected_epoch:
        raise ValueError(
            f"{experiment} checkpoint epoch metadata does not match training_summary.csv"
        )
    expected_accuracy = float(training_row["best_val_accuracy"])
    if not np.isclose(
        float(checkpoint["val_accuracy"]),
        expected_accuracy,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{experiment} checkpoint validation metadata does not match "
            "training_summary.csv"
        )


def _write_dict_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(matrix: np.ndarray, class_names: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_class", *class_names])
        for class_name, row in zip(class_names, matrix, strict=True):
            writer.writerow([class_name, *row.tolist()])


def _save_predictions(path: Path, result: EvaluationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        targets=result.targets,
        predictions=result.predictions,
        top5_predictions=result.top5_predictions,
        confidences=result.confidences,
    )


def _load_histories(
    artifact_dir: Path,
    experiments: Sequence[str],
) -> dict[str, list[dict[str, str]]]:
    return {
        experiment: _read_csv(artifact_dir / "histories" / f"{experiment}.csv")
        for experiment in experiments
    }


def _sample_for_gradcam(dataset: object, position: int, expected_target: int) -> torch.Tensor:
    image, target = dataset[position]  # type: ignore[index]
    if int(target) != expected_target:
        raise RuntimeError("Saved predictions do not match deterministic test order")
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise TypeError("Grad-CAM expects a transformed CHW image tensor")
    return image


def _sample_summary(
    position: int,
    result: EvaluationResult,
    class_names: Sequence[str],
    split_indices: np.ndarray,
) -> dict[str, Any]:
    target = int(result.targets[position])
    prediction = int(result.predictions[position])
    return {
        "dataset_position": position,
        "source_index": int(split_indices[position]),
        "true_index": target,
        "true_class": class_names[target],
        "predicted_index": prediction,
        "predicted_class": class_names[prediction],
        "confidence": float(result.confidences[position]),
    }


def run_evaluation(
    args: argparse.Namespace,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Evaluate requested checkpoints once, then generate all analysis artifacts."""

    if len(set(args.experiments)) != len(args.experiments):
        raise ValueError("experiments cannot contain duplicates")
    if device is None:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device(args.device)

    training_rows = _read_csv(args.artifact_dir / "training_summary.csv")
    training_by_experiment = {row["experiment"]: row for row in training_rows}
    missing_summaries = set(args.experiments) - set(training_by_experiment)
    if missing_summaries:
        raise ValueError(f"Training summary missing experiments: {sorted(missing_summaries)}")

    bundle: DataBundle = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        experiment="baseline",
    )
    class_names = bundle.class_names
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, Any]] = []
    evaluated: dict[str, EvaluationResult] = {}

    for experiment in args.experiments:
        training_row = training_by_experiment[experiment]
        checkpoint_path = args.artifact_dir / "checkpoints" / f"{experiment}.pth"
        model, checkpoint = load_checkpoint(
            checkpoint_path,
            device,
            num_classes=len(class_names),
        )
        validate_checkpoint_metadata(experiment, training_row, checkpoint)
        if list(checkpoint["class_names"]) != list(class_names):
            raise ValueError(f"{experiment} checkpoint class names do not match the dataset")
        result = evaluate_model(
            model,
            bundle.test_loader,
            criterion,
            device,
            num_classes=len(class_names),
        )
        evaluated[experiment] = result
        _save_predictions(args.artifact_dir / "predictions" / f"{experiment}.npz", result)
        row = {
            "experiment": experiment,
            "changed_variable": training_row.get("changed_variable", "none"),
            "seed": int(float(training_row.get("seed", args.seed))),
            "best_epoch": int(float(training_row.get("best_epoch", checkpoint["epoch"]))),
            "best_val_accuracy": float(
                training_row.get("best_val_accuracy", checkpoint["val_accuracy"])
            ),
            "test_loss": result.loss,
            "test_top1_accuracy": result.top1_accuracy,
            "test_top5_accuracy": result.top5_accuracy,
            "test_macro_f1": result.macro_f1,
            "train_seconds": training_row.get("train_seconds", ""),
            "checkpoint": str(checkpoint_path),
            "reused_checkpoint": training_row.get("reused_checkpoint", "False"),
        }
        rows.append(row)
        print(
            f"{experiment} | Val {row['best_val_accuracy']:.2%} | "
            f"Test Top-1 {result.top1_accuracy:.2%} | "
            f"Top-5 {result.top5_accuracy:.2%} | Macro-F1 {result.macro_f1:.2%}",
            flush=True,
        )
        del model

    _write_dict_csv(args.artifact_dir / "results.csv", rows, RESULT_FIELDS)
    histories = _load_histories(args.artifact_dir, args.experiments)
    figures_dir = args.artifact_dir / "figures"
    save_training_curves(histories, figures_dir / "training_curves.png")
    save_ablation_plot(rows, figures_dir / "ablation_metrics.png")

    selected_row = max(rows, key=lambda row: float(row["best_val_accuracy"]))
    selected_experiment = str(selected_row["experiment"])
    selected_result = evaluated[selected_experiment]
    counts, normalized = compute_confusion_tables(
        selected_result.targets,
        selected_result.predictions,
        num_classes=len(class_names),
    )
    pairs = top_confusion_pairs(counts, class_names, top_k=3)
    tables_dir = args.artifact_dir / "tables"
    _write_matrix_csv(counts, class_names, tables_dir / "confusion_matrix_counts.csv")
    _write_matrix_csv(
        normalized,
        class_names,
        tables_dir / "confusion_matrix_normalized.csv",
    )
    _write_dict_csv(
        tables_dir / "top_confusions.csv",
        pairs,
        (
            "class_a_index",
            "class_b_index",
            "class_a",
            "class_b",
            "a_as_b",
            "b_as_a",
            "total_confusions",
        ),
    )
    save_confusion_matrix(normalized, class_names, figures_dir / "confusion_matrix.png")
    save_local_confusion_matrix(
        normalized,
        class_names,
        pairs,
        figures_dir / "confusion_matrix_top_pairs.png",
    )

    top_pair = (
        int(pairs[0]["class_a_index"]),
        int(pairs[0]["class_b_index"]),
    )
    correct_position, incorrect_position = select_gradcam_examples(
        selected_result.targets,
        selected_result.predictions,
        selected_result.confidences,
        top_pair=top_pair,
    )
    selected_model, selected_checkpoint = load_checkpoint(
        args.artifact_dir / "checkpoints" / f"{selected_experiment}.pth",
        device,
        num_classes=len(class_names),
    )
    target_layer = selected_model.layer4[-1].conv2
    sample_records: dict[str, dict[str, Any]] = {}
    for label, position in (
        ("correct", correct_position),
        ("incorrect", incorrect_position),
    ):
        target = int(selected_result.targets[position])
        prediction = int(selected_result.predictions[position])
        image = _sample_for_gradcam(bundle.test_loader.dataset, position, target)
        cam = compute_gradcam(
            selected_model,
            target_layer,
            image.unsqueeze(0).to(device),
            target_class=prediction,
        )
        save_gradcam_figure(
            image,
            cam,
            true_name=class_names[target],
            predicted_name=class_names[prediction],
            confidence=float(selected_result.confidences[position]),
            output_path=figures_dir / f"gradcam_{label}.png",
        )
        sample_records[label] = _sample_summary(
            position,
            selected_result,
            class_names,
            bundle.splits.test,
        )

    analysis_summary = {
        "selected_experiment": selected_experiment,
        "selection_rule": "maximum best_val_accuracy",
        "best_val_accuracy": float(selected_row["best_val_accuracy"]),
        "checkpoint": str(args.artifact_dir / "checkpoints" / f"{selected_experiment}.pth"),
        "checkpoint_epoch": int(selected_checkpoint["epoch"]),
        "gradcam_target_layer": "model.layer4[-1].conv2",
        "top_confusions": pairs,
        "samples": sample_records,
    }
    (args.artifact_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Selected for analysis: {selected_experiment} "
        f"(Val {float(selected_row['best_val_accuracy']):.2%})",
        flush=True,
    )
    return rows


def main() -> None:
    args = parse_args()
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    run_evaluation(args, device)


if __name__ == "__main__":
    main()
