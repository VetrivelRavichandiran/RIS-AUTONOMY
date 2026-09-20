"""Path-loss models.

All functions return path loss in **dB** and validate that distances and the
carrier frequency are positive (raising :class:`ValueError` otherwise).
"""
from __future__ import annotations

import numpy as np

_C_LIGHT = 299_792_458.0  # m/s


def _validate(d, f_hz: float) -> None:
    """Raise if any distance is non-positive or the frequency is non-positive."""
    if np.any(np.asarray(d) <= 0) or f_hz <= 0:
        raise ValueError("distance and frequency must be positive")


def free_space_db(d_m, f_hz: float):
    """Free-space path loss: ``20 log10(4 pi d / lambda)`` (dB).

    Parameters
    ----------
    d_m:
        Distance(s) in meters (scalar or array).
    f_hz:
        Carrier frequency in Hz.

    Returns
    -------
    float or np.ndarray
        Path loss in dB.
    """
    _validate(d_m, f_hz)
    d = np.asarray(d_m, dtype=float)
    return 20.0 * np.log10(4.0 * np.pi * d * f_hz / _C_LIGHT)


def log_distance_db(d_m, f_hz: float, n: float, d_ref: float = 1.0, pl_ref: float | None = None):
    """Log-distance path loss: ``PL(d_ref) + 10 n log10(d / d_ref)`` (dB).

    Parameters
    ----------
    d_m:
        Distance(s) in meters.
    f_hz:
        Carrier frequency in Hz.
    n:
        Path-loss exponent.
    d_ref:
        Reference distance (m); FSPL is used at ``d_ref`` unless ``pl_ref`` given.
    pl_ref:
        Optional reference path loss (dB) at ``d_ref``.
    """
    _validate(d_m, f_hz)
    d = np.asarray(d_m, dtype=float)
    base = free_space_db(d_ref, f_hz) if pl_ref is None else pl_ref
    return base + 10.0 * n * np.log10(d / d_ref)


def three_gpp_um_db(d_m, f_hz: float, los):
    """3GPP TR 38.901 UMa path loss (dB).

    LOS:  ``32.45 + 21.7 log10(d) + 20 log10(f_GHz)``
    NLOS: LOS + ``16.7 log10(d)``

    Parameters
    ----------
    d_m:
        Distance(s) in meters.
    f_hz:
        Carrier frequency in Hz.
    los:
        Boolean array (same shape as ``d_m``); True = LOS, False = NLOS.
    """
    _validate(d_m, f_hz)
    d = np.asarray(d_m, dtype=float)
    base = 32.45 + 21.7 * np.log10(d) + 20.0 * np.log10(f_hz / 1e9)
    return np.where(np.asarray(los), base, base + 16.7 * np.log10(d))


def pathloss_db(d_m, f_hz: float, model: str, los=True, exponent: float = 2.5):
    """Dispatch to the named path-loss model.

    Parameters
    ----------
    model:
        One of ``"free_space"``, ``"log_distance"``, ``"3gpp_um"``.
    los:
        LOS flag(s) used by ``3gpp_um``.
    exponent:
        Path-loss exponent used by ``log_distance``.
    """
    models = {
        "free_space": lambda: free_space_db(d_m, f_hz),
        "log_distance": lambda: log_distance_db(d_m, f_hz, exponent),
        "3gpp_um": lambda: three_gpp_um_db(d_m, f_hz, los),
    }
    if model not in models:
        raise ValueError(f"unknown path-loss model {model!r}")
    return models[model]()