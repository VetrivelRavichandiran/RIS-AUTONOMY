"""Command-line interface for RIS-AUTONOMY.

Subcommands (also exposed via ``scripts/*.py`` thin wrappers):

* ``train``             -- train a PPO agent
* ``evaluate``          -- evaluate a trained model
* ``benchmark``         -- five-controller comparison
* ``experiment``        -- run a named parameter sweep or generalization
* ``generate-channels`` -- batch-generate channel realizations
* ``reproduce``         -- re-run a saved run from its manifest
* ``dashboard``         -- launch the Streamlit dashboard

Every command sets up logging, creates a unique run directory, saves a run
manifest (config + seed + versions), and prints a final summary with paths.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from .config import Config
from .rl.training import train
from .utils.io import run_manifest

logger = logging.getLogger(__name__)


def _new_run_dir(base: str, name: str) -> str:
    """Create a unique, never-overwritten run directory."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(base, f"{stamp}_{name}")
    os.makedirs(path, exist_ok=False)
    return path


def _save(fn, out: str) -> None:
    """Build a figure with ``fn`` and save it to ``out`` (never raises)."""
    try:
        from .visualization import save_figure

        save_figure(fn(), out)
    except Exception as exc:  # plotting must never break a pipeline
        logger.warning("Could not plot %s: %s", out, exc)


def _cmd_train(a) -> None:
    from .visualization import plot_training_curves

    c = Config.from_yaml(a.config)
    d = _new_run_dir(a.output_dir or "results", "train")
    run_manifest(d, c, a.seed, {"pipeline": "train"})
    result = train(c, a.seed, a.timesteps or c.rl.timesteps, d, device=a.device)
    _save(
        lambda: plot_training_curves(os.path.join(d, "metrics", "episode_metrics.csv")),
        os.path.join(d, "plots", "training.png"),
    )
    print(json.dumps(result, indent=2, default=str))


def _cmd_evaluate(a) -> None:
    from .evaluation.evaluator import evaluate_agent
    from .rl.agent import load_agent

    c = Config.from_yaml(a.config)
    model, _ = load_agent(a.model, c)
    d = _new_run_dir(a.output_dir or "results", "evaluate")
    run_manifest(d, c, a.seed, {"pipeline": "evaluate"})
    result = evaluate_agent(model, c, a.episodes, deterministic=not a.stochastic, seed=a.seed, output_dir=d)
    print(json.dumps(result["aggregates"], indent=2, default=str))


def _cmd_benchmark(a) -> None:
    from .evaluation.benchmark import run_benchmark

    c = Config.from_yaml(a.config)
    d = _new_run_dir(a.output_dir or "results", "benchmark")
    run_manifest(d, c, a.seed, {"pipeline": "benchmark"})
    df = run_benchmark(c, d, a.seed)
    print(df.to_string())


def _plot_sweep(name: str, df: pd.DataFrame, out: str) -> None:
    """Plot a sweep DataFrame using the matching performance plot."""
    from .visualization import (
        plot_fading_comparison,
        plot_rate_vs_ris_size,
        plot_rate_vs_snr,
        plot_vs_csi_error,
        plot_vs_distance,
        plot_vs_phase_bits,
        plot_vs_power,
        plot_vs_users,
        plot_vs_velocity,
    )

    x = df["value"].values
    if name == "snr":
        fig = plot_rate_vs_snr(df["snr_db"].values, df["rate_sum"].values)
    elif name == "ris_size":
        fig = plot_rate_vs_ris_size(x, df["rate_sum"].values)
    elif name == "phase_bits":
        fig = plot_vs_phase_bits(x, df["rate_sum"].values)
    elif name == "user_count":
        fig = plot_vs_users(x, df["rate_sum"].values)
    elif name == "velocity":
        fig = plot_vs_velocity(x, df["rate_sum"].values)
    elif name == "csi_error":
        fig = plot_vs_csi_error(x, df["rate_sum"].values)
    elif name == "power":
        fig = plot_vs_power(x, df["rate_sum"].values)
    elif name == "distance":
        fig = plot_vs_distance(x, df["rate_sum"].values)
    elif name == "fading":
        fig = plot_fading_comparison(x, df["rate_sum"].values)
    else:
        return
    _save(lambda: fig, out)


def _cmd_experiment(a) -> None:
    from .evaluation.experiments import _sweep_dispatch, run_generalization
    from .environment.state import observation_dim
    from .rl.agent import load_agent
    from .rl.training import train

    c = Config.from_yaml(a.config)
    d = _new_run_dir(a.output_dir or "results", a.name)
    run_manifest(d, c, a.seed, {"pipeline": "experiment", "name": a.name})
    name = a.name.replace("_sweep", "")

    if name == "generalization":
        if not a.model:
            raise SystemExit("experiment generalization requires --model")
        model, _ = load_agent(a.model, c)
        gen = c.generalization
        result = run_generalization(
            model,
            c,
            gen.get("train_condition", {}),
            gen.get("test_condition", {}),
            a.episodes,
            output_dir=d,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    model = load_agent(a.model, c)[0] if a.model else None
    model_factory = None
    if model is None:
        steps = c.experiment.sweep_train_timesteps
        if steps <= 0:
            raise SystemExit(
                "no --model supplied and experiment.sweep_train_timesteps is 0; "
                "pass --model or set experiment.sweep_train_timesteps > 0"
            )
        # Sweeps that change the observation dimension (user_count, ris_size)
        # need a freshly trained model per value; all others share one model.
        obs_dim_vary = name in ("user_count", "ris_size")
        if not obs_dim_vary:
            model = load_agent(train(c, a.seed, steps, d)["model_path"], c)[0]
        else:
            def model_factory(cfg, _name=name):
                val = cfg.to_dict().get("users", {}).get("count", "n")
                sub = os.path.join(d, "sweep_models", f"{_name}_{val}")
                return load_agent(train(cfg, a.seed, steps, sub)["model_path"], cfg)[0]

    df = _sweep_dispatch(name, model, c, a.episodes, output_dir=d, model_factory=model_factory)
    _plot_sweep(name, df, os.path.join(d, "plots", f"{name}.png"))
    print(df.to_string())


def _cmd_generate_channels(a) -> None:
    from .channels.channel_generator import ChannelGenerator
    from .environment.mobility import random_initial_positions

    c = Config.from_yaml(a.config)
    rng = np.random.default_rng(a.seed)
    pos = random_initial_positions(rng, c.users.count, c.scenario.ue_region)
    chs = [ChannelGenerator(c, rng).generate(pos) for _ in range(a.num)]
    rows = []
    for i, ch in enumerate(chs):
        rows.append(
            {
                "realization": i,
                "direct_real": float(ch.h_direct.real.mean()),
                "direct_imag": float(ch.h_direct.imag.mean()),
                "hbr_real": float(ch.H_br.real.mean()),
                "hbr_imag": float(ch.H_br.imag.mean()),
                "hru_real": float(ch.h_ru.real.mean()),
                "hru_imag": float(ch.h_ru.imag.mean()),
                "distances": json.dumps(ch.metadata.get("distances", [])),
            }
        )
    os.makedirs(a.out or "data/processed", exist_ok=True)
    out = os.path.join(a.out or "data/processed", "channels.parquet")
    pd.DataFrame(rows).to_parquet(out)
    print(out)


def _cmd_reproduce(a) -> None:
    d = a.run_dir
    c = Config.from_yaml(os.path.join(d, "config.yaml"))
    manifest = json.load(open(os.path.join(d, "run_manifest.json")))
    out = _new_run_dir(a.output_dir or "results", "reproduce")
    run_manifest(out, c, manifest["seed"], {"pipeline": "reproduce", "source": d})
    result = train(c, manifest["seed"], c.rl.timesteps, out)
    print(json.dumps(result, indent=2, default=str))


def _cmd_dashboard(a) -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("streamlit is not installed; cannot launch the dashboard.")
        return
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"], check=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all subcommands."""
    p = argparse.ArgumentParser(prog="ris_autonomy", description="RIS-AUTONOMY CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ["train", "evaluate", "benchmark", "experiment", "generate-channels", "reproduce", "dashboard"]:
        q = sub.add_parser(cmd)
        q.add_argument("--config")
        q.add_argument("--output-dir", default="results")
        q.add_argument("--model")
        q.add_argument("--timesteps", type=int)
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--episodes", type=int, default=3)
        q.add_argument("--name")
        q.add_argument("--num", type=int, default=1)
        q.add_argument("--out")
        q.add_argument("--run-dir")
        q.add_argument("--device")
        q.add_argument("--stochastic", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {
        "train": _cmd_train,
        "evaluate": _cmd_evaluate,
        "benchmark": _cmd_benchmark,
        "experiment": _cmd_experiment,
        "generate-channels": _cmd_generate_channels,
        "reproduce": _cmd_reproduce,
        "dashboard": _cmd_dashboard,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()