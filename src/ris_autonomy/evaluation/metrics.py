"""Evaluation metrics: fairness, outage, and per-step metric accumulation.

These are the standard multi-user performance metrics used by the evaluator,
benchmark, and experiment engine.  Everything is computed from actual
per-step values returned by the environment — nothing is hardcoded.
"""
from __future__ import annotations

import numpy as np


def fairness_jain(rates: np.ndarray) -> float:
    """Jain's fairness index of a rate vector.

    ``J = (sum(r))^2 / (K * sum(r^2))``; equals 1 for perfectly equal rates
    and approaches ``1/K`` when one user dominates.

    Parameters
    ----------
    rates:
        Per-user rates (any shape; flattened).

    Returns
    -------
    float
        Fairness index in ``(0, 1]``; 0 if all rates are zero.
    """
    r = np.asarray(rates, dtype=float).ravel()
    denom = (r * r).sum()
    if denom == 0:
        return 0.0
    return float(r.sum() ** 2 / (len(r) * denom))


def outage_probability(rates: np.ndarray, target_bps: float) -> float:
    """Fraction of users whose rate falls below the target.

    Parameters
    ----------
    rates:
        Per-user rates (bps).
    target_bps:
        Minimum acceptable rate (bps).

    Returns
    -------
    float
        Outage probability in ``[0, 1]``.
    """
    r = np.asarray(rates, dtype=float)
    return float(np.mean(r < target_bps))


class MetricsAccumulator:
    """Accumulate per-step environment ``info`` dicts into summary statistics.

    Feed one ``info`` dict per step via :meth:`update`; call :meth:`summary`
    for mean/median/std/min/max of each metric.  Inference latency is tracked
    separately (milliseconds, median reported).

    Parameters
    ----------
    keys:
        Metric names to accumulate (all present in each ``info`` dict).
    """

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys
        self.values: dict[str, list[float]] = {k: [] for k in keys}
        self.latencies_ms: list[float] = []

    def update(self, info: dict, latency_ms: float | None = None) -> None:
        """Record one step's metrics.

        Parameters
        ----------
        info:
            Per-step metric dict from the environment.
        latency_ms:
            Optional wall-clock time (ms) of the action call.
        """
        for key in self.keys:
            value = info[key]
            if isinstance(value, np.ndarray):
                value = float(np.mean(value))
            self.values[key].append(float(value))
        if latency_ms is not None:
            self.latencies_ms.append(float(latency_ms))

    def summary(self) -> dict:
        """Return per-metric summary statistics.

        Returns
        -------
        dict
            ``{metric: {"mean": ..., "median": ..., "std": ..., "min": ...,
            "max": ...}}`` plus ``inference_latency_ms`` (median).
        """
        out: dict = {}
        for key, vals in self.values.items():
            arr = np.asarray(vals)
            out[key] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        if self.latencies_ms:
            out["inference_latency_ms"] = float(np.median(self.latencies_ms))
        return out