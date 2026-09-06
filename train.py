"""Command-line training entry point for the three controlled experiments."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import DataBundle, build_dataloaders, seed_everything
from models.model import build_model
from utils.metrics import evaluate_model

EXPERIMENTS = ("baseline", "randaugment", "label_smoothing")
CHANGED_VARIABLE = {
    "baseline": "none",
    "randaugment": "training_augmentation",
    "label_smoothing": "training_loss",
}
TRAINING_SUMMARY_FIELDS = (
    "experiment",
    "changed_variable",
    "seed",
    "best_epoch",
    "best_val_accuracy",
    "train_seconds",
    "checkpoint",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def build_criteria(experiment: str) -> tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss]:
    """Return experiment-specific training loss and comparable validation loss."""

    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}")
    smoothing = 0.1 if experiment == "label_smoothing" else 0.0
    return nn.CrossEntropyLoss(label_smoothing=smoothing), nn.CrossEntropyLoss()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train one epoch and return sample-weighted loss and Top-1 accuracy."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Encountered non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader produced no samples")
    return total_loss / total_samples, total_correct / total_samples


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_config(args: argparse.Namespace, experiment: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "data_dir": str(args.data_dir),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "optimizer": "AdamW",
        "model": "resnet18",
        "num_classes": 37,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if len(set(args.experiments)) != len(args.experiments):
        raise ValueError("experiments cannot contain duplicates")


def _run_one_experiment(
    args: argparse.Namespace,
    experiment: str,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(args.seed)
    bundle: DataBundle = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        experiment=experiment,
    )
    model = build_model(num_classes=len(bundle.class_names), pretrained=True).to(device)
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("All model parameters must remain trainable")

    train_criterion, val_criterion = build_criteria(experiment)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    checkpoint_path = args.artifact_dir / "checkpoints" / f"{experiment}.pth"
    history_path = args.artifact_dir / "histories" / f"{experiment}.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir / experiment))
    history: list[dict[str, Any]] = []
    best_accuracy = float("-inf")
    best_epoch = 0
    started = time.perf_counter()

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model,
                bundle.train_loader,
                train_criterion,
                optimizer,
                device,
            )
            val_result = evaluate_model(
                model,
                bundle.val_loader,
                val_criterion,
                device,
                num_classes=len(bundle.class_names),
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.top1_accuracy,
            }
            history.append(row)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar("Loss/val", val_result.loss, epoch)
            writer.add_scalar("Accuracy/val", val_result.top1_accuracy, epoch)

            saved = ""
            if val_result.top1_accuracy > best_accuracy:
                best_accuracy = val_result.top1_accuracy
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "experiment": experiment,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_result.top1_accuracy,
                        "class_names": bundle.class_names,
                        "config": _checkpoint_config(args, experiment),
                    },
                    checkpoint_path,
                )
                saved = " | saved checkpoint"
            print(
                f"{experiment} Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Train Loss {train_loss:.4f} | Train Acc {train_accuracy:.2%} | "
                f"Val Loss {val_result.loss:.4f} | Val Acc {val_result.top1_accuracy:.2%}"
                f"{saved}",
                flush=True,
            )
    finally:
        writer.flush()
        writer.close()

    train_seconds = time.perf_counter() - started
    _write_csv(
        history_path,
        history,
        ("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"),
    )
    return {
        "experiment": experiment,
        "changed_variable": CHANGED_VARIABLE[experiment],
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "train_seconds": train_seconds,
        "checkpoint": str(checkpoint_path),
    }


def run_training(
    args: argparse.Namespace,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Train requested experiments without accessing their test split."""

    _validate_args(args)
    if device is None:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device(args.device)

    summary_path = args.artifact_dir / "training_summary.csv"
    merged_rows: dict[str, dict[str, Any]] = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", newline="") as stream:
            for existing in csv.DictReader(stream):
                experiment = existing.get("experiment", "")
                if experiment in EXPERIMENTS and experiment not in args.experiments:
                    merged_rows[experiment] = {
                        field: existing.get(field, "") for field in TRAINING_SUMMARY_FIELDS
                    }

    rows: list[dict[str, Any]] = []
    for experiment in args.experiments:
        row = _run_one_experiment(args, experiment, device)
        rows.append(row)
        merged_rows[experiment] = row
        _write_csv(
            summary_path,
            [merged_rows[name] for name in EXPERIMENTS if name in merged_rows],
            TRAINING_SUMMARY_FIELDS,
        )
    return rows


def main() -> None:
    args = parse_args()
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    run_training(args, device)


if __name__ == "__main__":
    main()
