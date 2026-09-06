"""Evaluation metrics for fine-grained pet classification."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pet37-matplotlib"))

import matplotlib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    top1_accuracy: float
    top5_accuracy: float
    macro_f1: float
    targets: np.ndarray
    predictions: np.ndarray
    top5_predictions: np.ndarray
    confidences: np.ndarray

    @property
    def accuracy(self) -> float:
        """Backward-compatible alias for Top-1 accuracy."""

        return self.top1_accuracy


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 37,
) -> EvaluationResult:
    """Evaluate once and return sample-weighted loss plus prediction metrics."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_top5: list[np.ndarray] = []
    all_confidences: list[np.ndarray] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError(
                    f"Expected logits shaped [N, {num_classes}], got {tuple(logits.shape)}"
                )
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Encountered non-finite evaluation loss: {loss.item()}")

            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            top_k = min(5, logits.shape[1])
            top5_predictions = logits.topk(k=top_k, dim=1).indices
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_targets.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_top5.append(top5_predictions.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())

    if total_samples == 0:
        raise ValueError("Evaluation DataLoader produced no samples")

    targets = np.concatenate(all_targets).astype(np.int64, copy=False)
    predictions = np.concatenate(all_predictions).astype(np.int64, copy=False)
    top5 = np.concatenate(all_top5).astype(np.int64, copy=False)
    confidences = np.concatenate(all_confidences).astype(np.float64, copy=False)
    top1_accuracy = float(np.mean(predictions == targets))
    top5_accuracy = float(np.mean(np.any(top5 == targets[:, None], axis=1)))
    macro_f1 = float(
        f1_score(
            targets,
            predictions,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
    )
    return EvaluationResult(
        loss=total_loss / total_samples,
        top1_accuracy=top1_accuracy,
        top5_accuracy=top5_accuracy,
        macro_f1=macro_f1,
        targets=targets,
        predictions=predictions,
        top5_predictions=top5,
        confidences=confidences,
    )


def compute_confusion_tables(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    num_classes: int = 37,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and row-normalized confusion matrices."""

    labels = list(range(num_classes))
    counts = confusion_matrix(targets, predictions, labels=labels)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )
    return counts, normalized


def top_confusion_pairs(
    counts: np.ndarray,
    class_names: Sequence[str],
    *,
    top_k: int = 3,
) -> list[dict[str, int | str]]:
    """Rank unordered class pairs by off-diagonal errors in both directions."""

    if counts.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix size does not match class names")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    pairs: list[dict[str, int | str]] = []
    for class_a in range(len(class_names)):
        for class_b in range(class_a + 1, len(class_names)):
            a_as_b = int(counts[class_a, class_b])
            b_as_a = int(counts[class_b, class_a])
            pairs.append(
                {
                    "class_a_index": class_a,
                    "class_b_index": class_b,
                    "class_a": class_names[class_a],
                    "class_b": class_names[class_b],
                    "a_as_b": a_as_b,
                    "b_as_a": b_as_a,
                    "total_confusions": a_as_b + b_as_a,
                }
            )
    pairs.sort(
        key=lambda pair: (
            -int(pair["total_confusions"]),
            int(pair["class_a_index"]),
            int(pair["class_b_index"]),
        )
    )
    return pairs[:top_k]


def save_ablation_plot(results: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Plot Top-1, Top-5, and Macro-F1 for all experiments."""

    if not results:
        raise ValueError("At least one experiment result is required")
    names = [str(row["experiment"]) for row in results]
    series = {
        "Top-1 Accuracy": [float(row["test_top1_accuracy"]) for row in results],
        "Top-5 Accuracy": [float(row["test_top5_accuracy"]) for row in results],
        "Macro-F1": [float(row["test_macro_f1"]) for row in results],
    }
    positions = np.arange(len(names))
    width = 0.24
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5.8))
    for offset, (label, values) in zip((-width, 0.0, width), series.items(), strict=True):
        bars = axis.bar(positions + offset, values, width, label=label)
        axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=3, fontsize=8)
    minimum = min(value for values in series.values() for value in values)
    axis.set_ylim(max(0.0, minimum - 0.08), 1.02)
    axis.set_xticks(positions, names)
    axis.set_ylabel("Score")
    axis.set_title("Oxford-IIIT Pet Ablation Results")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_training_curves(
    histories: dict[str, Sequence[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Plot comparable train/validation loss and validation accuracy curves."""

    if not histories:
        raise ValueError("At least one experiment history is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for experiment, rows in histories.items():
        if not rows:
            continue
        epochs = [int(float(row["epoch"])) for row in rows]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label=f"{experiment} train")
        axes[0].plot(
            epochs,
            [float(row["val_loss"]) for row in rows],
            linestyle="--",
            label=f"{experiment} val",
        )
        axes[1].plot(
            epochs,
            [float(row["val_accuracy"]) for row in rows],
            marker="o",
            markersize=3,
            label=experiment,
        )
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Validation Top-1 accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_confusion_matrix(
    normalized: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """Save the complete row-normalized confusion matrix."""

    if normalized.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix size does not match class names")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(class_names)), class_names, fontsize=7)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Row-normalized Confusion Matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Recall fraction")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_local_confusion_matrix(
    normalized: np.ndarray,
    class_names: Sequence[str],
    pairs: Sequence[dict[str, int | str]],
    output_path: Path,
) -> None:
    """Save a readable submatrix containing the classes in the top confusion pairs."""

    if normalized.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix size does not match class names")
    indices = sorted(
        {
            int(pair[key])
            for pair in pairs
            for key in ("class_a_index", "class_b_index")
        }
    )
    if not indices:
        raise ValueError("At least one confusion pair is required")
    local = normalized[np.ix_(indices, indices)]
    labels = [class_names[index] for index in indices]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 7.2))
    image = axis.imshow(local, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Top-confusion Class Submatrix")
    for row in range(local.shape[0]):
        for column in range(local.shape[1]):
            axis.text(
                column,
                row,
                f"{local[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if local[row, column] > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
