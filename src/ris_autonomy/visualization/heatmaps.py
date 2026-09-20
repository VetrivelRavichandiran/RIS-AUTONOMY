"""2D spatial heatmaps (e.g. rate / SINR over a UE position grid)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _grid_heatmap(values_2d: np.ndarray, x: np.ndarray, y: np.ndarray, title: str,
                  cmap: str, label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=110)
    im = ax.imshow(np.atleast_2d(values_2d), origin="lower", aspect="auto", cmap=cmap,
                   extent=[x[0], x[-1], y[0], y[-1]])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    return fig


def plot_rate_heatmap(values_2d: np.ndarray, x: np.ndarray, y: np.ndarray,
                      title: str = "Achievable rate heatmap") -> plt.Figure:
    """Heatmap of sum rate over a 2D grid of UE positions."""
    return _grid_heatmap(values_2d, np.atleast_1d(x), np.atleast_1d(y), title, "viridis", "Sum rate (bps)")


def plot_sinr_heatmap(values_2d: np.ndarray, x: np.ndarray, y: np.ndarray,
                      title: str = "SINR heatmap") -> plt.Figure:
    """Heatmap of mean SINR (dB) over a 2D grid of UE positions."""
    return _grid_heatmap(values_2d, np.atleast_1d(x), np.atleast_1d(y), title, "magma", "Mean SINR (dB)")