"""Performance / sweep / comparison plots.

All functions take plain arrays or pandas DataFrames (real experiment data) and
return a matplotlib Figure.  Nothing here fabricates data: if a requested series
is missing, the panel shows "N/A".
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c", "#ccb974"]


def _line(x, y, xlabel, ylabel, title, marker="o", log_y=False):
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=110)
    if y is None or len(np.atleast_1d(y)) == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    else:
        ax.plot(np.atleast_1d(x), np.atleast_1d(y), marker=marker, color=_COLORS[0], lw=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rate_vs_snr(snr_db, rate_sum, title="Sum rate vs SNR") -> plt.Figure:
    return _line(snr_db, rate_sum, "SNR (dB)", "Sum rate (bps)", title)


def plot_sinr_vs_snr(snr_db, sinr_db, title="Mean SINR vs SNR") -> plt.Figure:
    return _line(snr_db, sinr_db, "SNR (dB)", "Mean SINR (dB)", title)


def plot_rate_vs_ris_size(n_elements, rate_sum, title="Sum rate vs RIS elements") -> plt.Figure:
    return _line(n_elements, rate_sum, "RIS elements", "Sum rate (bps)", title)


def plot_vs_csi_error(csi_error, values, xlabel="CSI error (fraction)", ylabel="Sum rate (bps)",
                      title="Performance vs CSI error") -> plt.Figure:
    return _line(csi_error, values, xlabel, ylabel, title)


def plot_vs_velocity(velocity, values, ylabel="Sum rate (bps)", title="Performance vs user velocity") -> plt.Figure:
    return _line(velocity, values, "Velocity (m/s)", ylabel, title)


def plot_vs_phase_bits(bits, values, ylabel="Sum rate (bps)", title="Performance vs phase resolution") -> plt.Figure:
    return _line(bits, values, "Phase bits", ylabel, title)


def plot_vs_users(count, values, ylabel="Sum rate (bps)", title="Performance vs number of users") -> plt.Figure:
    return _line(count, values, "Number of users", ylabel, title)


def plot_vs_power(power_dbm, values, ylabel="Sum rate (bps)", title="Performance vs transmit power") -> plt.Figure:
    return _line(power_dbm, values, "Transmit power (dBm)", ylabel, title)


def plot_vs_distance(distance, values, ylabel="Sum rate (bps)", title="Performance vs distance") -> plt.Figure:
    return _line(distance, values, "Distance (m)", ylabel, title)


def plot_fading_comparison(fading, values, ylabel="Sum rate (bps)", title="Performance vs fading") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=110)
    ax.bar(np.atleast_1d(fading), np.atleast_1d(values), color=_COLORS[: len(np.atleast_1d(fading))])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_baseline_comparison(df: pd.DataFrame, metric: str = "rate_sum", ylabel: str = "Sum rate (bps)",
                            title: str = "Baseline comparison") -> plt.Figure:
    """Grouped bar chart of one metric across baselines.

    Parameters
    ----------
    df:
        Comparison table with one row per baseline (index = baseline name) and a
        column per metric.
    metric:
        Column to plot.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=110)
    if metric not in df.columns:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    else:
        order = list(df.index)
        ax.bar(order, df[metric].values, color=_COLORS[: len(order)])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rate_cdf(rate_samples, title="CDF of achievable rate", label: str = "rate") -> plt.Figure:
    """Empirical CDF of a collection of per-step sum-rate samples."""
    samples = np.asarray(rate_samples, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=110)
    if samples.size == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    else:
        s = np.sort(samples)
        cdf = np.arange(1, s.size + 1) / s.size
        ax.plot(s, cdf, color=_COLORS[0], lw=1.6, label=label)
        ax.set_xlabel("Sum rate (bps)")
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.3)
        ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_ee_comparison(df: pd.DataFrame, title="Energy-efficiency comparison") -> plt.Figure:
    return plot_baseline_comparison(df, metric="ee", ylabel="Energy efficiency (bps/W)", title=title)


def plot_inference_latency(latency_ms, title="Inference latency") -> plt.Figure:
    """Distribution of per-step inference latency (ms)."""
    lat = np.asarray(latency_ms, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=110)
    if lat.size == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    else:
        ax.hist(lat, bins=min(30, max(5, lat.size // 2)), color=_COLORS[3], edgecolor="k", alpha=0.85)
        ax.axvline(float(np.median(lat)), color="k", ls="--", lw=1,
                   label=f"median {np.median(lat):.3f} ms")
        ax.set_xlabel("latency (ms)")
        ax.set_ylabel("count")
        ax.legend()
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig