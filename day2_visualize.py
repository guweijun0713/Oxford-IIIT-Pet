"""Backward-compatible Day 2 visualization entry point.

All analysis code is implemented in ``evaluate.py`` and ``utils``.
"""

from __future__ import annotations

from evaluate import main
from utils.gradcam import compute_gradcam, save_gradcam_figure, select_gradcam_examples
from utils.metrics import (
    compute_confusion_tables,
    save_ablation_plot,
    save_confusion_matrix,
    save_local_confusion_matrix,
    top_confusion_pairs,
)

__all__ = [
    "compute_confusion_tables",
    "compute_gradcam",
    "save_ablation_plot",
    "save_confusion_matrix",
    "save_gradcam_figure",
    "save_local_confusion_matrix",
    "select_gradcam_examples",
    "top_confusion_pairs",
]


if __name__ == "__main__":
    main()
