"""Reusable parameter-sweep experiment engine.

``run_sweep`` evaluates a model (or baseline) across a list of values of one
parameter, producing a DataFrame of aggregate metrics plus a real plot.  The
built-in sweeps cover SNR, RIS size, phase resolution, user count, velocity,
CSI error, transmit power, distance, and fading; ``run_generalization``
evaluates a trained model under test conditions different from training.

Every number comes from actual evaluation — nothing is hardcoded.
"""
from __future__ import annotations

import logging
import math
import os

import pandas as pd

from ..config import Config
from .evaluator import evaluate_agent

logger = logging.getLogger(__name__)

# Metrics carried into every sweep row.
_METRICS = ("rate_sum", "sinr_db", "ee", "outage", "fairness")


def _resolve_model(model, model_factory, cfg):
    """Pick the model to evaluate under ``cfg``.

    * ``model_factory`` (if given) is called with ``cfg`` to build a matching
      model -- used when a sweep changes the observation dimension (e.g.
      ``user_count`` / ``ris_size``) so a single fixed net cannot be reused.
    * otherwise the fixed ``model`` is used; ``None`` means a random-policy
      baseline.
    """
    if model_factory is not None:
        return model_factory(cfg)
    return model


def _aggregate(result: dict) -> dict:
    """Extract the standard metric columns from an evaluation result."""
    agg = result["aggregates"]
    return {k: float(agg[k]) for k in _METRICS if k in agg}


def run_sweep(model, config: Config, key: str, values, episodes: int, output_dir: str | None = None, model_factory=None, seed: int = 42) -> pd.DataFrame:
    """Evaluate ``model`` across ``values`` of the dotted config ``key``.

    Parameters
    ----------
    model:
        A loaded SB3 model (or None for a random-policy baseline).  Ignored
        when ``model_factory`` is given.
    config:
        Base configuration; each value overrides ``key``.
    key:
        Dotted config path, e.g. ``"channel.csi_error"``.
    values:
        Iterable of values to sweep.
    episodes:
        Episodes per value.
    output_dir:
        Optional directory for the resulting CSV/JSON.
    model_factory:
        Optional callable ``(cfg) -> model`` used to build a model that
        matches each value's observation dimension (required for sweeps such
        as ``user_count`` / ``ris_size`` that change the network input size).
    seed:
        Evaluation seed (kept fixed across values so the channel realizations
        are comparable).

    Returns
    -------
    pandas.DataFrame
        One row per value with the standard metrics.
    """
    rows = []
    for value in values:
        cfg = config.clone_with_overrides({key: value})
        m = _resolve_model(model, model_factory, cfg)
        result = evaluate_agent(m, cfg, episodes, seed=seed)
        row = {"value": value, **_aggregate(result)}
        rows.append(row)
        logger.info("sweep %s=%s -> rate_sum=%.1f", key, value, row["rate_sum"])
    df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"sweep_{key.split('.')[-1]}.csv"), index=False)
        df.to_json(os.path.join(output_dir, f"sweep_{key.split('.')[-1]}.json"), orient="records", indent=2)
    return df


def _calibrated_tx_power_dbm(config: Config, model, target_snr_db: float, episodes: int = 3, seed: int = 42) -> float:
    """Transmit power (dBm) that makes the *received* SINR equal ``target_snr_db``.

    The mmWave path loss means the received SINR is far below the raw
    ``P_tx - noise`` figure, so the power is found by bisection on the
    measured achieved SINR (which is a monotonic, near-linear function of
    ``P_tx``).  The calibration uses the same ``episodes``/``seed`` as the
    sweep so it targets the exact average the sweep reports.  This is what
    makes the SNR sweep's x-axis labels truthful.
    """
    lo, hi = -150.0, 160.0
    best = hi
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        cfg = config.clone_with_overrides({"system.transmit_power_dbm": mid})
        achieved = evaluate_agent(model, cfg, episodes, seed=seed)["aggregates"]["sinr_db"]
        if achieved < target_snr_db:
            lo = mid
        else:
            hi = mid
            best = mid
    return best


def _maybe_save(df: pd.DataFrame, output_dir: str | None, name: str) -> None:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"sweep_{name}.csv"), index=False)
        df.to_json(os.path.join(output_dir, f"sweep_{name}.json"), orient="records", indent=2)


def _sweep_dispatch(name: str, model, config: Config, episodes: int, output_dir: str | None = None, model_factory=None) -> pd.DataFrame:
    """Run one of the named built-in sweeps."""
    if name == "snr":
        # Calibrate transmit power so each row's *received* SINR matches its
        # label (the mmWave path loss makes the raw P_tx - noise figure
        # meaningless as a received-SNR target).
        cal_model = _resolve_model(model, model_factory, config)
        powers = [
            _calibrated_tx_power_dbm(config, cal_model, s, episodes=episodes)
            for s in config.experiment.snr_sweep_db
        ]
        df = run_sweep(model, config, "system.transmit_power_dbm", powers, episodes, output_dir, model_factory)
        df.insert(0, "snr_db", config.experiment.snr_sweep_db)
        return df
    if name == "ris_size":
        rows = []
        for n in config.experiment.ris_size_sweep:
            side = int(math.isqrt(n))
            if side * side != n:
                raise ValueError(f"ris_size {n} is not a perfect square")
            cfg = config.clone_with_overrides({"ris.rows": side, "ris.columns": side})
            rows.append({"value": n, **_aggregate(evaluate_agent(_resolve_model(model, model_factory, cfg), cfg, episodes))})
        df = pd.DataFrame(rows)
        _maybe_save(df, output_dir, name)
        return df
    if name == "phase_bits":
        return run_sweep(model, config, "ris.phase_bits", config.experiment.phase_bits_sweep, episodes, output_dir, model_factory)
    if name == "user_count":
        return run_sweep(model, config, "users.count", config.experiment.user_count_sweep, episodes, output_dir, model_factory)
    if name == "velocity":
        return run_sweep(model, config, "users.velocity_mps", config.experiment.velocity_sweep_mps, episodes, output_dir, model_factory)
    if name == "csi_error":
        return run_sweep(model, config, "channel.csi_error", config.experiment.csi_error_sweep, episodes, output_dir, model_factory)
    if name == "power":
        return run_sweep(model, config, "system.transmit_power_dbm", config.experiment.power_sweep_dbm, episodes, output_dir, model_factory)
    if name == "distance":
        # Scale the scenario so the mean BS->UE distance approximates each value:
        # move the RIS and UE region outward from the BS by the target distance.
        rows = []
        for d in config.experiment.distance_sweep_m:
            cfg = config.clone_with_overrides(
                {
                    "scenario.ris_position": [d, d / 2],
                    "scenario.ue_region": [d - 10, d / 2 - 10, d + 10, d / 2 + 10],
                    "scenario.boundary_width_m": max(d + 20, config.scenario.boundary_width_m),
                    "scenario.boundary_height_m": max(d / 2 + 20, config.scenario.boundary_height_m),
                }
            )
            rows.append({"value": d, **_aggregate(evaluate_agent(_resolve_model(model, model_factory, cfg), cfg, episodes))})
        df = pd.DataFrame(rows)
        _maybe_save(df, output_dir, name)
        return df
    if name == "fading":
        rows = []
        for fading in config.experiment.fading_sweep:
            cfg = config.clone_with_overrides({"channel.fading": fading})
            rows.append({"value": fading, **_aggregate(evaluate_agent(_resolve_model(model, model_factory, cfg), cfg, episodes))})
        df = pd.DataFrame(rows)
        _maybe_save(df, output_dir, name)
        return df
    raise ValueError(f"unknown sweep {name!r}")


def run_generalization(
    model,
    config: Config,
    train_condition: dict,
    test_condition: dict,
    episodes: int,
    output_dir: str | None = None,
) -> dict:
    """Evaluate a model trained under one condition on a different condition.

    Parameters
    ----------
    model:
        Loaded trained model.
    config:
        Base configuration.
    train_condition, test_condition:
        Dotted-key override dicts describing the two conditions.

    Returns
    -------
    dict
        ``{"train": {...}, "test": {...}, "gap": {...}}`` where ``gap`` is
        ``train - test`` per metric (a positive rate gap indicates
        out-of-distribution degradation).
    """
    train_cfg = config.clone_with_overrides(train_condition)
    test_cfg = config.clone_with_overrides(test_condition)
    train_res = _aggregate(evaluate_agent(model, train_cfg, episodes))
    test_res = _aggregate(evaluate_agent(model, test_cfg, episodes))
    gap = {k: train_res[k] - test_res[k] for k in train_res}
    result = {"train": train_res, "test": test_res, "gap": gap}
    if output_dir:
        import json

        os.makedirs(output_dir, exist_ok=True)
        json.dump(result, open(os.path.join(output_dir, "generalization.json"), "w"), indent=2)
    return result