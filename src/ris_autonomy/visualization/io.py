"""Figure persistence helpers (PNG always; optional Plotly HTML)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe default; callers may override for display

import matplotlib.pyplot as plt  # noqa: E402


def save_figure(fig, path_png: str, path_html: str | None = None) -> str:
    """Save a matplotlib figure to PNG (and a Plotly figure to HTML if given).

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` (saved to ``path_png``) or a
        ``plotly.graph_objects.Figure`` (saved to ``path_html`` if provided).
    path_png:
        Destination PNG path.
    path_html:
        Optional destination HTML path for Plotly figures.

    Returns
    -------
    str
        The PNG path (or HTML path for pure-Plotly figures).
    """
    is_plotly = type(fig).__module__.startswith("plotly")
    if is_plotly:
        if path_html is None:
            path_html = os.path.splitext(path_png)[0] + ".html"
        os.makedirs(os.path.dirname(path_html) or ".", exist_ok=True)
        fig.write_html(path_html)
        return path_html

    os.makedirs(os.path.dirname(path_png) or ".", exist_ok=True)
    fig.savefig(path_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path_png