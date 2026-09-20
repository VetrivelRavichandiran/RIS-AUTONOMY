"""MRT precoding and per-user SINR for the MISO effective channel.

The transmitter uses maximum-ratio transmit (MRT) on the *observed*
(CSI-errored) effective channel; the SINR is then evaluated on the *true*
effective channel, matching a realistic system that precodes with imperfect
channel state information.
"""
from __future__ import annotations

import numpy as np


def mrt_precoding(h_eff_tilde: np.ndarray) -> np.ndarray:
    """Maximum-ratio transmit precoder.

    Each user's beamformer is the conjugate of its effective channel row,
    normalized to unit norm so every user carries ``P_tx / K`` of the power.

    Parameters
    ----------
    h_eff_tilde:
        Observed effective channel, shape ``(K, M)`` (K users, M antennas).

    Returns
    -------
    numpy.ndarray
        Precoder ``W``, shape ``(M, K)``; column ``k`` is the unit-norm beam
        for user ``k``.
    """
    h = np.asarray(h_eff_tilde, dtype=complex)
    norms = np.linalg.norm(h, axis=1)
    norms = np.maximum(norms, 1e-15)
    return np.conj(h).T / norms


def received_signal(
    h_eff: np.ndarray,
    w: np.ndarray,
    p_tx_w: float,
    noise_power_w: float,
) -> tuple[np.ndarray, dict]:
    """Per-user SINR for a given precoder on the true effective channel.

    User ``k`` receives ``|h_k . W[:,k]|^2 * P/K`` of signal power and
    ``sum_{j != k} |h_k . W[:,j]|^2 * P/K`` of interference, plus thermal
    noise ``noise_power_w``.

    Parameters
    ----------
    h_eff:
        True effective channel, shape ``(K, M)``.
    w:
        Precoder, shape ``(M, K)``.
    p_tx_w:
        Total transmit power (W); split equally over the K users.
    noise_power_w:
        Thermal noise power at the receiver (W).

    Returns
    -------
    tuple
        ``(sinr_lin, powers)`` where ``sinr_lin`` has shape ``(K,)`` and
        ``powers`` is a dict with ``signal`` / ``interference`` / ``noise``
        arrays (each ``(K,)``).
    """
    h = np.asarray(h_eff, dtype=complex)
    k = h.shape[0]
    gain = np.abs(h @ w) ** 2 * (p_tx_w / k)
    signal = np.diag(gain)
    interference = gain.sum(axis=1) - signal
    sinr = signal / (interference + noise_power_w)
    return sinr, {
        "signal": signal,
        "interference": interference,
        "noise": noise_power_w,
    }