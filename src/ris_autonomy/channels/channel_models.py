"""Channel-realization container, effective channel, and CSI error.

Tensor dimensions (K users, M BS antennas, N RIS elements)::

    h_direct : (K, M) complex   direct BS -> UE
    H_br     : (N, M) complex   BS -> RIS
    h_ru     : (K, N) complex   RIS -> UE
    theta    : (N,)     real    RIS phase vector (rad)

The effective channel is ``H_eff = h_direct + h_ru @ diag(e^{j theta}) @ H_br``
(computed vectorized as ``h_direct + (h_ru * e^{j theta}) @ H_br``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChannelRealization:
    """A full set of channel matrices for one UE-position snapshot.

    Attributes
    ----------
    h_direct, H_br, h_ru:
        Complex channel matrices (see module docstring for shapes).
    positions:
        UE positions, shape (K, 2).
    metadata:
        Free-form metadata (e.g. link distances).
    """

    h_direct: np.ndarray
    H_br: np.ndarray
    h_ru: np.ndarray
    positions: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        k, m = self.h_direct.shape
        n = self.H_br.shape[0]
        if self.H_br.shape[1] != m or self.h_ru.shape != (k, n) or self.positions.shape != (k, 2):
            raise ValueError("incompatible channel shapes")
        self.h_direct = np.asarray(self.h_direct, dtype=np.complex128)
        self.H_br = np.asarray(self.H_br, dtype=np.complex128)
        self.h_ru = np.asarray(self.h_ru, dtype=np.complex128)


def compute_effective_channel(h_direct: np.ndarray, H_br: np.ndarray, h_ru: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Effective channel ``H_eff = h_direct + h_ru @ diag(e^{j theta}) @ H_br``.

    Parameters
    ----------
    h_direct:
        (K, M) direct channels.
    H_br:
        (N, M) BS->RIS channels.
    h_ru:
        (K, N) RIS->UE channels.
    theta:
        (N,) RIS phase vector (rad).

    Returns
    -------
    np.ndarray
        (K, M) effective channels.

    Raises
    ------
    ValueError
        If the dimensions are inconsistent.
    """
    if H_br.shape[0] != len(theta) or h_ru.shape != (h_direct.shape[0], len(theta)) or H_br.shape[1] != h_direct.shape[1]:
        raise ValueError("effective-channel dimensions incompatible")
    return h_direct + (h_ru * np.exp(1j * theta)[None, :]) @ H_br


def apply_csi_error(H: np.ndarray, rng: np.random.Generator, csi_error: float) -> np.ndarray:
    """Apply multiplicative/imperfect-CSI error to a channel matrix.

    ``H_tilde = sqrt(1-eps) H + sqrt(eps) sigma W`` where ``sigma`` is the
    per-tap RMS of ``H`` and ``W ~ CN(0,1)``.  ``eps`` is the CSI error
    fraction in ``[0, 1)``.

    Parameters
    ----------
    H:
        Channel matrix to corrupt.
    rng:
        Random generator.
    csi_error:
        Error fraction in ``[0, 1)``.

    Returns
    -------
    np.ndarray
        The CSI-errored channel.
    """
    if not 0 <= csi_error < 1:
        raise ValueError("csi_error in [0,1)")
    sigma = np.linalg.norm(H) / np.sqrt(H.size)
    w = (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)) / np.sqrt(2)
    return np.sqrt(1 - csi_error) * H + np.sqrt(csi_error) * sigma * w