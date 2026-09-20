"""PPO training pipeline for RIS-AUTONOMY.

``train`` seeds everything, builds the environment + PPO agent, runs
``model.learn`` with metrics/best-model/checkpoint callbacks, then saves the
final model, training statistics, and real training-curve plots.  All numbers
in the outputs come from the actual run.
"""
from __future__ import annotations

import logging
import os
import time

from ..environment.wireless_env import RISAUTONOMYEnv
from ..evaluation.evaluator import evaluate_agent
from ..utils.io import save_json
from ..utils.seeding import set_global_seed
from .agent import create_agent, save_agent
from .callbacks import BestModelCallback, EpisodeMetricsCallback

logger = logging.getLogger(__name__)


def _resolve_device(requested: str | None) -> str:
    """Resolve the training device, falling back to CPU with a warning."""
    import torch

    if requested in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU")
        return "cpu"
    return requested


def train(
    config,
    seed: int,
    timesteps: int,
    output_dir: str,
    device: str | None = None,
    model_path: str | None = None,
    resume: bool = False,
) -> dict:
    """Train a PPO agent and persist all run artifacts.

    Parameters
    ----------
    config:
        A validated :class:`~ris_autonomy.config.Config`.
    seed:
        Global random seed (deterministic when fixed).
    timesteps:
        Total environment steps for ``model.learn``.
    output_dir:
        Run directory (created if missing).
    device:
        ``auto``/``cpu``/``cuda``; falls back to CPU when CUDA is unavailable.
    model_path:
        Optional path to resume from.
    resume:
        If True and ``model_path`` is given, continue from that model.

    Returns
    -------
    dict
        Summary with model path, output dir, duration, and final eval metrics.
    """
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    set_global_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    config = config.clone_with_overrides({"experiment.seed": seed})

    # SB3 requires batch <= rollout; shrink rollout for tiny requested jobs.
    if timesteps < config.rl.n_steps:
        config = config.clone_with_overrides(
            {
                "rl.n_steps": max(8, min(timesteps, config.rl.n_steps)),
                "rl.batch_size": min(config.rl.batch_size, max(8, min(timesteps, config.rl.n_steps))),
            }
        )

    dev = _resolve_device(device)
    logger.info(
        "Training: algorithm=%s device=%s timesteps=%d seed=%d",
        config.rl.algorithm, dev, timesteps, seed,
    )
    env = RISAUTONOMYEnv(config, seed)
    model = create_agent(config, env, device=dev)
    if resume and model_path:
        from .agent import load_agent

        model, _ = load_agent(model_path, config, device=dev)

    metrics_csv = os.path.join(output_dir, "metrics", "episode_metrics.csv")
    best_zip = os.path.join(output_dir, "models", "final", "best_model.zip")
    callback_list = [
        EpisodeMetricsCallback(metrics_csv),
        BestModelCallback(best_zip, window=10),
    ]
    if config.rl.checkpoint_every > 0:
        callback_list.append(
            CheckpointCallback(
                save_freq=max(1, config.rl.checkpoint_every),
                save_path=os.path.join(output_dir, "models", "checkpoints"),
                name_prefix="ppo",
            )
        )
    callbacks = CallbackList(callback_list)

    start = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    duration = time.time() - start
    logger.info("Training finished in %.1fs", duration)

    final_zip = os.path.join(output_dir, "models", "final", "final_model.zip")
    metadata = {
        "algorithm": "PPO",
        "seed": seed,
        "timesteps": timesteps,
        "device": dev,
        "config": config.to_dict(),
    }
    save_agent(model, final_zip, metadata)
    # Ensure a best model exists even if the callback never fired (very short runs).
    if not os.path.exists(best_zip):
        save_agent(model, best_zip, metadata)

    final_eval = evaluate_agent(model, config, 1, seed=seed)["aggregates"]
    stats = {
        "total_timesteps": timesteps,
        "seed": seed,
        "device": dev,
        "duration_s": duration,
        "final_eval": final_eval,
    }
    save_json(stats, os.path.join(output_dir, "training_stats.json"))

    # Real training-curve plots (reward, rate, SINR, EE).
    try:
        from ..visualization import plot_training_curves, save_figure

        fig = plot_training_curves(metrics_csv)
        save_figure(fig, os.path.join(output_dir, "plots", "training.png"))
    except Exception as exc:  # plotting must never break training
        logger.warning("Could not plot training curves: %s", exc)

    return {"model_path": final_zip, "output_dir": output_dir, **stats}