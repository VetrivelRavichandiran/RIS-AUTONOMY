"""Stable-Baselines3 callbacks for deterministic, auditable training artifacts.

* :class:`EpisodeMetricsCallback` writes one row per episode (reward, rate, SINR,
  energy efficiency, switching cost) to a CSV so training curves can be plotted.
* :class:`BestModelCallback` tracks the best mean reward over a rolling window
  and saves the model **only when strictly better** than the recorded best, so
  the best model is never overwritten by an equal-or-worse checkpoint.
"""
from __future__ import annotations

import csv
import json
import os

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """Append per-episode metrics to a CSV file.

    Parameters
    ----------
    path:
        Destination CSV path (parent directories are created on training end).
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.rows: list[list[float]] = []
        self.rewards: list[float] = []
        self.length = 0

    def _on_step(self) -> bool:  # noqa: D102 (SB3 hook)
        self.rewards.append(float(self.locals["rewards"][0]))
        self.length += 1
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            self.rows.append(
                [
                    len(self.rows),
                    sum(self.rewards) / len(self.rewards),
                    info["rate_sum"],
                    float(info["sinr_db"].mean()),
                    info["ee"],
                    info["switching_cost"],
                    self.length,
                ]
            )
            self.rewards = []
            self.length = 0
        return True

    def _on_training_end(self) -> None:  # noqa: D102
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "mean_reward", "rate_sum", "sinr_db", "ee", "switching", "episode_length"])
            writer.writerows(self.rows)


class BestModelCallback(BaseCallback):
    """Save the model when the rolling-window mean reward is a new best.

    The best value is persisted to a JSON sidecar next to the model so a
    resumed run never overwrites a better model with a worse one.

    Parameters
    ----------
    path:
        Destination ``.zip`` path for the best model.
    window:
        Number of recent episodes used for the rolling mean reward.
    """

    def __init__(self, path: str, window: int = 10) -> None:
        super().__init__()
        self.path = path
        self.window = window
        self.sidecar = os.path.join(os.path.dirname(path) or ".", "best_model.json")
        self.rewards: list[float] = []
        self.best: float = self._load_best()

    def _load_best(self) -> float:
        if os.path.exists(self.sidecar):
            try:
                with open(self.sidecar) as f:
                    return float(json.load(f).get("best_reward", float("-inf")))
            except (json.JSONDecodeError, ValueError, OSError):
                return float("-inf")
        return float("-inf")

    def _on_step(self) -> bool:  # noqa: D102
        self.rewards.append(float(self.locals["rewards"][0]))
        if self.locals["dones"][0]:
            window = self.rewards[-self.window :]
            mean_reward = sum(window) / len(window)
            if mean_reward > self.best:
                self.best = mean_reward
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                self.model.save(self.path)
                json.dump(
                    {"best_reward": self.best, "timestep": self.num_timesteps},
                    open(self.sidecar, "w"),
                    indent=2,
                )
        return True