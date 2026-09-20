"""SINR / SNR dB-conversion helpers.

All conversions are pure NumPy and guard against zero/negative inputs so the
logarithm never produces ``-inf`` or NaN.
"""
from __future__ import annotations

import numpy as np

# Floor for the linear argument of the log so 0 dB is finite.
_EPS = 1e-300


def sinr_db(sinr_lin: np.ndarray) -> np.ndarray:
    """Convert linear SINR to dB.

    Parameters
    ----------
    sinr_lin:
        Linear SINR, shape ``(K,)`` (or scalar).

    Returns
    -------
    numpy.ndarray
        SINR in dB, same shape as input.
    """
    x = np.asarray(sinr_lin, dtype=float)
    return 10.0 * np.log10(np.maximum(x, _EPS))


def snr_db(noise_power_w: float, p_tx_w: float, k_users: int = 1) -> float:
    """Per-user SNR in dB for a total transmit power split over K users.

    Parameters
    ----------
    noise_power_w:
        Thermal noise power (W).
    p_tx_w:
        Total transmit power (W).
    k_users:
        Number of users the power is split over.

    Returns
    -------
    float
        Per-user SNR in dB.
    """
    return 10.0 * np.log10((p_tx_w / k_users) / noise_power_w)