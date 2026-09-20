"""Training-curve and convergence plots (from episode-metrics CSV)."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def _load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"training metrics not found: {path}")
    return pd.read_csv(path)


def plot_training_curves(csv_path: str, title: str = "Training progress") -> plt.Figure:
    """Four-panel training curve: reward, sum rate, SINR, energy efficiency.

    Parameters
    ----------
    csv_path:
        Path to ``episode_metrics.csv`` (columns: episode, mean_reward, rate_sum,
        sinr_db, ee, switching, episode_length).

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = _load(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=110, sharex=True)
    panels = [
        ("mean_reward", "Mean reward", "#4c72b0"),
        ("rate_sum", "Sum rate (bps)", "#55a868"),
        ("sinr_db", "Mean SINR (dB)", "#dd8452"),
        ("ee", "Energy efficiency (bps/W)", "#8172b3"),
    ]
    for ax, (col, label, color) in zip(axes.ravel(), panels):
        if col in df.columns:
            ax.plot(df["episode"], df[col], color=color, lw=1.2, alpha=0.85)
            ax.set_ylabel(label)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("episode")
    axes[1, 1].set_xlabel("episode")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_convergence(csv_path: str, window: int = 10, title: str = "Reward convergence") -> plt.Figure:
    """Raw vs windowed-mean episode reward (convergence view).

    Parameters
    ----------
    csv_path:
        Path to ``episode_metrics.csv``.
    window:
        Rolling-window size for the smoothed curve.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = _load(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    ax.plot(df["episode"], df["mean_reward"], color="#4c72b0", alpha=0.3, lw=0.8, label="episode reward")
    if "mean_reward" in df.columns and len(df) >= 2:
        smooth = df["mean_reward"].rolling(window, min_periods=1).mean()
        ax.plot(df["episode"], smooth, color="#c44e52", lw=2.0, label=f"mean ({window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig