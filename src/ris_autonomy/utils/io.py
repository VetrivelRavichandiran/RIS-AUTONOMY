"""Filesystem, JSON/CSV/Parquet I/O, run manifests, and version capture.

Run directories are timestamped and never overwritten, so previous experiments
are always preserved.  ``run_manifest`` writes the config + seed + software
versions needed to reproduce a run.
"""
from __future__ import annotations

import datetime
import json
import os
import platform

import pandas as pd


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if missing; return it."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str) -> None:
    """Write ``obj`` to ``path`` as indented JSON (parents created)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: str) -> str:
    """Save a DataFrame to Parquet (if the path ends in ``.parquet`` and
    pyarrow is available) or CSV otherwise.  Returns the written path."""
    ensure_dir(os.path.dirname(path) or ".")
    if path.endswith(".parquet"):
        try:
            df.to_parquet(path)
            return path
        except Exception:
            path = os.path.splitext(path)[0] + ".csv"
    df.to_csv(path, index=False)
    return path


def software_versions() -> dict:
    """Capture the software versions for reproducibility records."""
    import gymnasium
    import numpy
    import stable_baselines3
    import torch

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "sb3": stable_baselines3.__version__,
    }


def run_manifest(run_dir: str, config, seed: int, extra: dict | None = None) -> None:
    """Write ``config.yaml`` + ``run_manifest.json`` into a run directory.

    Parameters
    ----------
    run_dir:
        Target run directory (created if missing).
    config:
        The validated configuration for the run.
    seed:
        The global seed used.
    extra:
        Optional extra fields (e.g. ``{"pipeline": "train"}``).
    """
    ensure_dir(run_dir)
    config.to_yaml(os.path.join(run_dir, "config.yaml"))
    manifest = {
        "seed": seed,
        "timestamp": datetime.datetime.now().isoformat(),
        "versions": software_versions(),
        **(extra or {}),
    }
    save_json(manifest, os.path.join(run_dir, "run_manifest.json"))