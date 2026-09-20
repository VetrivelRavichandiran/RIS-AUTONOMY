"""2D wireless topology visualization (BS, RIS, UEs, links)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:  # plotly is optional at import time; degrade gracefully
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    _HAS_PLOTLY = False


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def plot_topology(data: dict, title: str = "Wireless topology") -> plt.Figure:
    """Draw the 2D topology: BS (triangle), RIS (rectangle), UEs (circles).

    Parameters
    ----------
    data:
        Dict with keys ``bs`` (2,), ``ris`` (2,), ``ues`` (K, 2).  Optional key
        ``distances`` (K,) gives BS-UE distances for labels.
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bs = np.asarray(data["bs"], dtype=float)
    ris = np.asarray(data["ris"], dtype=float)
    ues = np.atleast_2d(np.asarray(data["ues"], dtype=float))
    dists = data.get("distances")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=110)
    # direct links BS -> UE (solid)
    for i, ue in enumerate(ues):
        ax.plot([bs[0], ue[0]], [bs[1], ue[1]], "-", color="#1f77b4", alpha=0.55, lw=1.2)
        if dists is not None:
            mid = (bs + ue) / 2
            ax.text(mid[0], mid[1], f"{dists[i]:.0f} m", fontsize=7, color="#1f77b4", ha="center")
    # RIS links BS -> RIS -> UE (dashed)
    ax.plot([bs[0], ris[0]], [bs[1], ris[1]], "--", color="#d62728", alpha=0.7, lw=1.2)
    for ue in ues:
        ax.plot([ris[0], ue[0]], [ris[1], ue[1]], "--", color="#d62728", alpha=0.45, lw=1.0)

    # nodes
    ax.scatter(*bs, marker="^", s=220, c="#2ca02c", edgecolor="k", zorder=5)
    ax.text(bs[0], bs[1] - 4, "BS", ha="center", fontsize=9, fontweight="bold")
    ax.add_patch(
        plt.Rectangle(
            (ris[0] - 3, ris[1] - 3), 6, 6, fill=True, facecolor="#ff9f43",
            edgecolor="k", zorder=5,
        )
    )
    ax.text(ris[0], ris[1] + 5, "RIS", ha="center", fontsize=9, fontweight="bold")
    ax.scatter(*ues.T, marker="o", s=90, c="#9467bd", edgecolor="k", zorder=5)
    for i, ue in enumerate(ues):
        ax.text(ue[0] + 2, ue[1], f"UE {i}", fontsize=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_topology_interactive(data: dict, title: str = "Wireless topology") -> "go.Figure":
    """Interactive (Plotly) topology with hover labels.

    Raises
    ------
    RuntimeError
        If plotly is not installed.
    """
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly is not installed; cannot build interactive topology")
    bs = np.asarray(data["bs"], dtype=float)
    ris = np.asarray(data["ris"], dtype=float)
    ues = np.atleast_2d(np.asarray(data["ues"], dtype=float))

    fig = go.Figure()
    for i, ue in enumerate(ues):
        fig.add_trace(
            go.Scatter(
                x=[bs[0], ue[0]], y=[bs[1], ue[1]], mode="lines",
                line=dict(color="#1f77b4", width=1.5),
                name=f"direct BS→UE{i}", hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[bs[0], ris[0], ue[0]], y=[bs[1], ris[1], ue[1]], mode="lines",
                line=dict(color="#d62728", width=1.2, dash="dash"),
                name=f"RIS path UE{i}", hoverinfo="skip",
            )
        )
    fig.add_trace(go.Scatter(x=[bs[0]], y=[bs[1]], mode="markers", name="BS",
                             marker=dict(symbol="triangle", size=16, color="#2ca02c")))
    fig.add_trace(go.Scatter(x=[ris[0]], y=[ris[1]], mode="markers", name="RIS",
                             marker=dict(symbol="square", size=16, color="#ff9f43")))
    fig.add_trace(go.Scatter(x=ues[:, 0], y=ues[:, 1], mode="markers", name="UE",
                             marker=dict(symbol="circle", size=11, color="#9467bd")))
    fig.update_layout(title=title, aspectratio=dict(x=1, y=1), showlegend=True,
                      xaxis_title="x (m)", yaxis_title="y (m)")
    return fig