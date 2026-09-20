"""Reproducible random-number seeding.

``set_global_seed`` seeds every RNG the platform uses (Python ``random``,
NumPy, PyTorch CPU/CUDA, and the hash seed) so that a run with a fixed seed is
byte-for-byte reproducible.
"""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic runs.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where supported (warn-only so unsupported ops
    # don't hard-fail on CPU).
    torch.use_deterministic_algorithms(True, warn_only=True)