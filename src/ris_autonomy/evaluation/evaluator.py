"""Agent / baseline evaluation harness.

``evaluate_agent`` runs a model (or random policy) for ``episodes`` episodes,
accumulates per-step metrics (rate, SINR, energy efficiency, BER, outage,
fairness) and measures per-step inference latency.  It returns per-episode rows
plus aggregates and, when ``output_dir`` is given, writes ``metrics.csv`` /
``metrics.json`` and evaluation plots.  All values come from actual execution.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pandas as pd

from ..environment.wireless_env import RISAUTONOMYEnv

logger = logging.getLogger(__name__)


def evaluate_agent(
    model,
    config,
    episodes: int,
    deterministic: bool = True,
    seed: int | None = None,
    output_dir: str | None = None,
    baseline=None,
) -> dict:
    """Evaluate a model (or baseline) over ``episodes`` episodes.

    Parameters
    ----------
    model:
        A loaded SB3 model, or ``None`` for a random policy.
    config:
        Validated configuration.
    episodes:
        Number of episodes to run.
    deterministic:
        Use deterministic actions (``model.predict(deterministic=True)``).
    seed:
        Base seed; episode ``e`` uses ``seed + e``.
    output_dir:
        Optional directory for ``metrics.csv``/``metrics.json`` and plots.
    baseline:
        Optional baseline controller (used instead of ``model`` if given).

    Returns
    -------
    dict
        ``{"episodes": [per-episode rows], "aggregates": {metric: mean}}``.
    """
    base_seed = seed if seed is not None else config.experiment.seed
    rows = []
    all_rates: list[float] = []
    all_latencies: list[float] = []

    for e in range(episodes):
        env = RISAUTONOMYEnv(config, base_seed + e)
        obs, _ = env.reset()
        vals = []
        lats = []
        done = False
        while not done:
            if baseline is not None:
                t0 = time.perf_counter()
                action = baseline.action(env)
            else:
                t0 = time.perf_counter()
                action = model.predict(obs, deterministic=deterministic)[0] if model is not None else env.action_space.sample()
            lats.append((time.perf_counter() - t0) * 1000.0)
            obs, _, term, trunc, info = env.step(action)
            vals.append(info)
            all_rates.append(float(info["rate_sum"]))
            done = term or trunc
        all_latencies.extend(lats)

        rows.append(
            {
                "rate_sum": float(np.mean([x["rate_sum"] for x in vals])),
                "sinr_db": float(np.mean([np.mean(x["sinr_db"]) for x in vals])),
                "ee": float(np.mean([x["ee"] for x in vals])),
                "ber": float(np.mean([np.mean(x["ber"]) for x in vals])),
                "outage": float(np.mean([x["outage"] for x in vals])),
                "fairness": float(np.mean([x["fairness"] for x in vals])),
                "switching": float(np.mean([x["switching_cost"] for x in vals])),
                "inference_latency_ms": float(np.median(lats)),
            }
        )
        env.close()

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out = {"episodes": rows, "aggregates": agg}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        json.dump(out, open(os.path.join(output_dir, "metrics.json"), "w"), indent=2)
        # Real evaluation plots: rate CDF + inference-latency distribution.
        try:
            from ..visualization import plot_inference_latency, plot_rate_cdf, save_figure

            plots = os.path.join(output_dir, "plots")
            save_figure(plot_rate_cdf(all_rates, "CDF of sum rate"), os.path.join(plots, "rate_cdf.png"))
            save_figure(plot_inference_latency(all_latencies), os.path.join(plots, "inference_latency.png"))
        except Exception as exc:  # plotting must never break evaluation
            logger.warning("Could not plot evaluation results: %s", exc)

    return out