"""Achievable-rate computations for the MISO RIS link.

All rates use the Shannon formula ``R = B * log2(1 + SINR)`` with the SINR in
linear units.  Functions are pure NumPy and vectorized over the user axis.
"""
from __future__ import annotations

import numpy as np


def rate_bps(sinr_lin: np.ndarray, bandwidth_hz: float) -> np.ndarray:
    """Per-user achievable rate in bits/s.

    Parameters
    ----------
    sinr_lin:
        Linear (not dB) SINR per user, shape ``(K,)``.
    bandwidth_hz:
        Channel bandwidth in Hz.

    Returns
    -------
    numpy.ndarray
        Achievable rate per user, shape ``(K,)``.
    """
    sinr = np.asarray(sinr_lin, dtype=float)
    return bandwidth_hz * np.log2(1.0 + sinr)


def sum_rate(sinr_lin: np.ndarray, bandwidth_hz: float) -> float:
    """Sum achievable rate (bps) across all users.

    Parameters
    ----------
    sinr_lin:
        Linear SINR per user, shape ``(K,)``.
    bandwidth_hz:
        Channel bandwidth in Hz.

    Returns
    -------
    float
        Sum of the per-user rates.
    """
    return float(rate_bps(sinr_lin, bandwidth_hz).sum())


def spectral_efficiency_bps_hz(sinr_lin: np.ndarray) -> np.ndarray:
    """Spectral efficiency in bits/s/Hz per user.

    Parameters
    ----------
    sinr_lin:
        Linear SINR per user, shape ``(K,)``.

    Returns
    -------
    numpy.ndarray
        ``log2(1 + SINR)`` per user, shape ``(K,)``.
    """
    return np.log2(1.0 + np.asarray(sinr_lin, dtype=float))