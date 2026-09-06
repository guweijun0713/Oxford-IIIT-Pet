"""Backward-compatible Day 2 experiment entry point.

New work should use ``train.py`` followed by ``evaluate.py`` directly.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import torch

from data.dataset import build_transforms, seed_everything
from evaluate import run_evaluation
from train import (
    build_criteria,
    parse_args as _parse_train_args,
    run_training,
    train_one_epoch,
)
from utils.metrics import EvaluationResult, evaluate_model


def build_train_transform(experiment: str):
    return build_transforms(experiment)[0]


evaluate_with_predictions = evaluate_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    return _parse_train_args(
        ["--experiments", "randaugment", "label_smoothing", *arguments]
    )


def run_day2(args: argparse.Namespace, device: torch.device | None = None):
    """Train both Day 2 variants and then run the shared independent evaluator."""

    run_training(args, device)
    evaluation_args = argparse.Namespace(
        experiments=["baseline", "randaugment", "label_smoothing"],
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    return run_evaluation(evaluation_args, device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_everything(args.seed)
    run_day2(args, device)


if __name__ == "__main__":
    main()
