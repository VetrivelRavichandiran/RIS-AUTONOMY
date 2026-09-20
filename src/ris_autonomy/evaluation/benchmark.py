"""Fair five-controller benchmark orchestration.

Evaluates no-RIS, random-RIS, greedy, conventional, and the trained DRL agent
on the *same* scenario and *same* channel seed, then writes a comparison table
(CSV/JSON) and real comparison plots.  The DRL model is trained on the fly if
none is supplied (``benchmark.train_drl_if_missing``).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from ..baselines import make_baseline
from ..environment.wireless_env import RISAUTONOMYEnv
from ..rl.agent import load_agent
from ..rl.training import train
from .evaluator import evaluate_agent

logger = logging.getLogger(__name__)

_BASELINES = ("no_ris", "random_ris", "greedy", "conventional")
_METRICS = ("rate_sum", "sinr_db", "ee", "outage", "fairness")


def _eval_controller(controller, config, episodes: int, seed: int) -> dict:
    """Run a baseline controller for ``episodes`` episodes; return mean metrics."""
    rows = []
    for ep in range(episodes):
        env = RISAUTONOMYEnv(config, seed + ep)
        obs, _ = env.reset()
        vals = []
        done = False
        while not done:
            action = controller.action(env)
            obs, _, term, trunc, info = env.step(action)
            vals.append(info)
            done = term or trunc
        row = {}
        for k in _METRICS:
            series = [np.mean(x[k]) if isinstance(x[k], np.ndarray) else x[k] for x in vals]
            row[k] = float(np.mean(series))
        rows.append(row)
        env.close()
    return {k: float(np.mean([r[k] for r in rows])) for k in _METRICS}


def run_benchmark(config, output_dir: str, seed: int = 42) -> pd.DataFrame:
    """Run the full five-controller benchmark and persist artifacts.

    Parameters
    ----------
    config:
        Validated configuration.
    output_dir:
        Run directory for artifacts.
    seed:
        Channel/evaluation seed (shared by all controllers for fairness).

    Returns
    -------
    pandas.DataFrame
        Comparison table (index = baseline, columns = metrics).
    """
    os.makedirs(output_dir, exist_ok=True)
    # The benchmark's purpose is to distinguish the controllers.  Greedy (sum-rate)
    # and conventional (per-user) optimization are mathematically identical for a
    # single user, so the comparison runs with at least 2 users to make multi-user
    # interference (and thus the two optimizers) meaningful.  The DRL model is
    # trained on this same comparison config so every controller is compared fairly.
    cmp_config = config.clone_with_overrides(
        {"users.count": max(config.users.count, 2)}
    )
    if cmp_config.users.count != config.users.count:
        logger.info(
            "Benchmark uses %d users (config had %d) so greedy vs conventional differ.",
            cmp_config.users.count, config.users.count,
        )
    config = cmp_config

    model_path = config.benchmark.drl_model_path
    if not model_path or not os.path.exists(model_path):
        if not config.benchmark.train_drl_if_missing:
            raise FileNotFoundError("no DRL model provided and train_drl_if_missing is false")
        logger.info("No DRL model found; training a quick model (%d steps)", config.benchmark.drl_timesteps_if_missing)
        model_path = train(config, seed, config.benchmark.drl_timesteps_if_missing, output_dir)["model_path"]
    model, _ = load_agent(model_path, config)

    results: dict[str, dict] = {}
    for name in _BASELINES:
        results[name] = _eval_controller(make_baseline(name), config, config.benchmark.episodes, seed)
    results["drl"] = evaluate_agent(model, config, config.benchmark.episodes, seed=seed)["aggregates"]

    df = pd.DataFrame(results).T
    df.index.name = "baseline"
    df.to_csv(os.path.join(output_dir, "comparison.csv"))
    df.to_json(os.path.join(output_dir, "comparison.json"), indent=2)
    json.dump(results, open(os.path.join(output_dir, "summary.json"), "w"), indent=2)

    # Real comparison plots (rate, SINR, EE, outage) + rate CDF.
    try:
        from ..visualization import (
            plot_baseline_comparison,
            plot_rate_cdf,
            save_figure,
        )

        plots = os.path.join(output_dir, "plots")
        save_figure(plot_baseline_comparison(df, "rate_sum", "Sum rate (bps)", "Achievable rate"), os.path.join(plots, "comparison_rate.png"))
        save_figure(plot_baseline_comparison(df, "sinr_db", "Mean SINR (dB)", "SINR"), os.path.join(plots, "comparison_sinr.png"))
        save_figure(plot_baseline_comparison(df, "ee", "Energy efficiency (bps/W)", "Energy efficiency"), os.path.join(plots, "comparison_ee.png"))
        save_figure(plot_baseline_comparison(df, "outage", "Outage probability", "Outage"), os.path.join(plots, "comparison_outage.png"))
        # Rate CDF across all baselines (per-episode rates).
        samples = []
        for name in _BASELINES:
            env = RISAUTONOMYEnv(config, seed)
            obs, _ = env.reset()
            b = make_baseline(name)
            while True:
                a = b.action(env)
                obs, _, term, trunc, info = env.step(a)
                samples.append(info["rate_sum"])
                if term or trunc:
                    break
            env.close()
        save_figure(plot_rate_cdf(samples, "CDF of sum rate (all baselines)"), os.path.join(plots, "rate_cdf.png"))
    except Exception as exc:  # plotting must never break the benchmark
        logger.warning("Could not plot benchmark comparison: %s", exc)

    return df