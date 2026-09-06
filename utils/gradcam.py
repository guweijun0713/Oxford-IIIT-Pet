"""Grad-CAM computation, sample selection, and rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.dataset import IMAGENET_MEAN, IMAGENET_STD


def select_gradcam_examples(
    targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    *,
    top_pair: tuple[int, int],
) -> tuple[int, int]:
    """Select confident correct and incorrect examples tied to the top pair."""

    if not (len(targets) == len(predictions) == len(confidences)):
        raise ValueError("Targets, predictions, and confidences must have equal lengths")
    pair_mask = np.isin(targets, top_pair)
    pair_prediction_mask = np.isin(predictions, top_pair)
    correct_candidates = np.flatnonzero((targets == predictions) & pair_mask)
    incorrect_candidates = np.flatnonzero(
        (targets != predictions) & pair_mask & pair_prediction_mask
    )
    if correct_candidates.size == 0:
        correct_candidates = np.flatnonzero(targets == predictions)
    if incorrect_candidates.size == 0:
        incorrect_candidates = np.flatnonzero(targets != predictions)
    if correct_candidates.size == 0 or incorrect_candidates.size == 0:
        raise ValueError("Grad-CAM requires at least one correct and one incorrect prediction")
    correct = int(correct_candidates[np.argmax(confidences[correct_candidates])])
    incorrect = int(incorrect_candidates[np.argmax(confidences[incorrect_candidates])])
    return correct, incorrect


def compute_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    image: torch.Tensor,
    *,
    target_class: int,
) -> torch.Tensor:
    """Compute a normalized Grad-CAM map and remove temporary hooks."""

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def capture_activation(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    handle = target_layer.register_forward_hook(capture_activation)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(image)
            if target_class < 0 or target_class >= logits.shape[1]:
                raise ValueError("target_class is outside the model output range")
            logits[:, target_class].sum().backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hook did not capture activations and gradients")
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam - cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam = cam / maximum
        return cam.detach().cpu()
    finally:
        handle.remove()
        if was_training:
            model.train()


def save_gradcam_figure(
    normalized_image: torch.Tensor,
    cam: torch.Tensor,
    *,
    true_name: str,
    predicted_name: str,
    confidence: float,
    output_path: Path,
) -> None:
    """Save the CenterCrop model input and its spatially aligned heatmap overlay."""

    image = normalized_image.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image_array = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    heatmap = plt.get_cmap("jet")(cam.detach().cpu().numpy())[..., :3]
    overlay = np.clip(0.55 * image_array + 0.45 * heatmap, 0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(image_array)
    axes[0].set_title("Model input (224 x 224)")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"True: {true_name} | Predicted: {predicted_name} | Confidence: {confidence:.2%}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
