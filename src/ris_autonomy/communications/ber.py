"""Bit-error-rate (BER) approximations for Gray-mapped QAM.

Standard closed-form approximations (Gray mapping, no coding):

* QPSK:   ``0.5 * erfc(sqrt(SINR))``
* 16-QAM: ``(8/15) * erfc(sqrt(3*SINR/13)) * (1 - (2/5) * sqrt(3*SINR/13) / (sqrt(pi) * (1+SINR)))``
* 64-QAM: ``(15/64) * erfc(sqrt(2*SINR/7)) * (1 - (2/7) * sqrt(2*SINR/7) / (sqrt(pi) * (1+SINR)))``

These are the usual first-term union-bound approximations; they are monotone
decreasing in SINR and lie in ``[0, 1]``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

# Argument of erfc for each modulation, in terms of SINR.
_ERFC_ARG = {
    "qpsk": lambda s: np.sqrt(s),
    "16qam": lambda s: np.sqrt(3.0 * s / 13.0),
    "64qam": lambda s: np.sqrt(2.0 * s / 7.0),
}

# Leading coefficient and the correction factor for the QAM approximations.
_QAM_COEF = {"16qam": 8.0 / 15.0, "64qam": 15.0 / 64.0}
_QAM_CORR = {"16qam": 2.0 / 5.0, "64qam": 2.0 / 7.0}


def ber_approx(sinr_lin: np.ndarray, modulation: str) -> np.ndarray:
    """Approximate BER per user for the given modulation.

    Parameters
    ----------
    sinr_lin:
        Linear SINR per user, shape ``(K,)``.
    modulation:
        One of ``"qpsk"``, ``"16qam"``, ``"64qam"``.

    Returns
    -------
    numpy.ndarray
        Approximate BER per user, shape ``(K,)``, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``modulation`` is not supported.
    """
    s = np.asarray(sinr_lin, dtype=float)
    if modulation not in _ERFC_ARG:
        raise ValueError(f"unsupported modulation {modulation!r}")
    x = _ERFC_ARG[modulation](s)
    if modulation == "qpsk":
        return 0.5 * erfc(x)
    coef = _QAM_COEF[modulation]
    corr = _QAM_CORR[modulation]
    return coef * erfc(x) * (1.0 - corr * x / (np.sqrt(np.pi) * (1.0 + s)))