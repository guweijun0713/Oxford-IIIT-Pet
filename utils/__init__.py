"""Evaluation metrics and model-analysis utilities."""

import os
import tempfile
from pathlib import Path

# The desktop sandbox cannot write the default user-level Matplotlib cache.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pet37-matplotlib"))
