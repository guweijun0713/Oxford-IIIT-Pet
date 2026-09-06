"""ResNet-18 model and safe checkpoint utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

REQUIRED_CHECKPOINT_FIELDS = {
    "epoch",
    "experiment",
    "model_state_dict",
    "optimizer_state_dict",
    "val_accuracy",
    "class_names",
    "config",
}


def build_model(num_classes: int = 37, pretrained: bool = True) -> nn.Module:
    """Create a fully trainable ResNet-18 classifier."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    num_classes: int = 37,
) -> tuple[nn.Module, dict[str, Any]]:
    """Safely load and validate a project checkpoint."""

    checkpoint_path = Path(path)
    # Evaluation does not need AdamW moment tensors on the GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} must contain a dictionary")
    missing = REQUIRED_CHECKPOINT_FIELDS - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint missing required fields: {sorted(missing)}")
    class_names = checkpoint["class_names"]
    if not isinstance(class_names, list) or len(class_names) != num_classes:
        raise ValueError(
            f"Checkpoint class_names must contain exactly {num_classes} entries"
        )

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint
