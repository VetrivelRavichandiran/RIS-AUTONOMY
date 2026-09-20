"""Configurable multi-objective reward.

The reward is a weighted sum of normalized communication metrics minus a
switching-cost penalty (and a reserved constraint term)::

    R = w_rate * (rate_sum / rate_norm)
      + w_sinr * (mean(SINR_dB) / sinr_norm_db)
      + w_ee   * (EE / ee_norm)
      - w_switch * switch_fraction
      - w_constraint * violation      # reserved, 0 by default

Each metric is divided by its ``*_norm`` scale so no single term numerically
dominates; all scales are configurable.
"""
from __future__ import annotations

import numpy as np


class RewardFunction:
    """Compute the multi-objective reward from a step's metric ``info`` dict.

    Parameters
    ----------
    config:
        Validated configuration (reward weights + normalization scales).
    """

    def __init__(self, config) -> None:
        self.c = config

    def __call__(self, info: dict) -> tuple[float, dict]:
        """Return ``(reward, breakdown)`` where ``breakdown`` sums to ``reward``.

        Parameters
        ----------
        info:
            Per-step metrics from the environment (``rate_sum``, ``sinr_db``,
            ``ee``, ``switching_cost``).
        """
        c = self.c
        parts = {
            "rate": c.reward.rate_weight * info["rate_sum"] / c.reward.rate_norm,
            "sinr": c.reward.sinr_weight * float(np.mean(info["sinr_db"])) / c.reward.sinr_norm_db,
            "energy": c.reward.energy_weight * info["ee"] / c.reward.ee_norm,
            "switching": -c.reward.switching_weight * info["switching_cost"],
            "constraint": 0.0,
        }
        return float(sum(parts.values())), parts