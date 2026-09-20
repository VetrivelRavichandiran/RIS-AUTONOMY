"""RIS phase-configuration heatmaps and histograms."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_phase_grid(
    theta: np.ndarray,
    rows: int,
    cols: int,
    title: str = "RIS phase configuration",
    annotate: bool = True,
) -> plt.Figure:
    """Render the RIS phase vector as a rows×cols heatmap.

    Parameters
    ----------
    theta:
        Phase vector of length ``rows*cols`` in radians [0, 2π).
    rows, cols:
        RIS grid dimensions.
    title:
        Figure title.
    annotate:
        Whether to label each element with its index.

    Returns
    -------
    matplotlib.figure.Figure
    """
    theta = np.asarray(theta, dtype=float).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=110)
    im = ax.imshow(theta, origin="lower", cmap="twilight", vmin=0, vmax=2 * np.pi)
    if annotate:
        for r in range(rows):
            for c in range(cols):
                ax.text(c, r, f"{r * cols + c}", ha="center", va="center",
                        fontsize=max(4, 14 - rows // 2), color="w")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("phase (rad)")
    fig.tight_layout()
    return fig


def plot_phase_histogram(theta: np.ndarray, levels: int, title: str = "Phase distribution") -> plt.Figure:
    """Histogram of element phases binned on the quantization grid.

    Parameters
    ----------
    theta:
        Phase vector (rad).
    levels:
        Number of quantization levels (2**bits).

    Returns
    -------
    matplotlib.figure.Figure
    """
    theta = np.asarray(theta, dtype=float)
    step = 2 * np.pi / levels
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=110)
    counts, edges = np.histogram(theta, bins=np.linspace(0, 2 * np.pi, levels + 1))
    ax.bar(edges[:-1], counts, width=step * 0.9, color="#4c72b0", edgecolor="k", alpha=0.85)
    ax.set_xlabel("phase (rad)")
    ax.set_ylabel("element count")
    ax.set_title(title)
    ax.set_xticks(np.linspace(0, 2 * np.pi, levels + 1))
    ax.set_xticklabels([f"{i * step:.2f}" for i in range(levels + 1)], rotation=30, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig