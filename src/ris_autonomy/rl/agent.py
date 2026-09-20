"""PPO agent factory, model persistence, and metadata.

``create_agent`` builds a Stable-Baselines3 PPO agent from the config (with a
device that gracefully falls back to CPU when CUDA is unavailable).  Models are
saved as SB3 ``.zip`` files with a JSON metadata sidecar recording the
algorithm, seed, timesteps, full config, timestamp, and software versions so a
run can be reproduced.
"""
from __future__ import annotations

import json
import logging
import os
import time

from stable_baselines3 import PPO

from ..utils.io import software_versions
from .policies import build_policy_net

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a model file is missing or corrupted."""


def _resolve_device(requested: str | None) -> str:
    """Resolve the device, falling back to CPU with a warning if needed."""
    import torch

    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA unavailable, falling back to CPU")
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, falling back to CPU")
        return "cpu"
    return requested


def create_agent(config, env, device: str | None = None) -> PPO:
    """Create a PPO agent from the config and environment.

    Parameters
    ----------
    config:
        Validated configuration (RL hyperparameters + seed).
    env:
        The Gymnasium environment to train on.
    device:
        ``auto``/``cpu``/``cuda``; falls back to CPU when CUDA is unavailable.

    Returns
    -------
    stable_baselines3.PPO
    """
    dev = _resolve_device(device or config.rl.device)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.rl.learning_rate,
        n_steps=config.rl.n_steps,
        batch_size=min(config.rl.batch_size, config.rl.n_steps),
        n_epochs=config.rl.n_epochs,
        gamma=config.rl.gamma,
        gae_lambda=config.rl.gae_lambda,
        ent_coef=config.rl.ent_coef,
        clip_range=config.rl.clip_range,
        policy_kwargs=build_policy_net(config),
        seed=config.experiment.seed,
        device=dev,
        verbose=0,
    )
    logger.info("Agent created: PPO device=%s obs_dim=%d n_ris=%d", dev, env.observation_space.shape[0], env.n)
    return model


def save_agent(model, path: str, metadata: dict) -> None:
    """Save the model (``.zip``) plus a JSON metadata sidecar."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    meta = dict(metadata)
    meta.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("versions", software_versions())
    model.save(path)
    with open(os.path.join(directory, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Model saved: %s", path)


def load_agent(path: str, config, device: str = "cpu"):
    """Load a model and its metadata.

    Parameters
    ----------
    path:
        Path to the ``.zip`` model (with or without extension).
    config:
        Configuration (used for context in error messages).
    device:
        Device to load onto.

    Returns
    -------
    tuple
        ``(model, metadata_dict)``.

    Raises
    ------
    ModelLoadError
        If the file is missing or cannot be loaded.
    """
    candidate = path if os.path.exists(path) else (path + ".zip" if os.path.exists(path + ".zip") else path)
    if not os.path.exists(candidate):
        raise ModelLoadError(f"missing model: {path}")
    try:
        model = PPO.load(candidate, device=device)
    except Exception as exc:  # corrupted / incompatible checkpoint
        raise ModelLoadError(f"could not load model {path}: {exc}") from exc
    sidecar = os.path.join(os.path.dirname(candidate) or ".", "model_metadata.json")
    metadata = json.load(open(sidecar)) if os.path.exists(sidecar) else {}
    logger.info("Model loaded: %s", candidate)
    return model, metadata