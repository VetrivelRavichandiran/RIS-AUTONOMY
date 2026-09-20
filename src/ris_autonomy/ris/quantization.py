"""B-bit RIS phase quantization.

A :class:`PhaseQuantizer` with ``B`` bits has ``2**B`` equally spaced levels on
``[0, 2pi)``.  ``quantize`` snaps arbitrary phases to the nearest level;
``from_action`` maps a normalized continuous action in ``[-1, 1]`` to
``[0, 2pi)`` (via ``(a+1)*pi``) and then quantizes.  All operations are
vectorized and deterministic.
"""
from __future__ import annotations

import numpy as np


class PhaseQuantizer:
    """Quantize RIS phases to a ``2**B``-level grid on ``[0, 2pi)``.

    Parameters
    ----------
    bits:
        Phase resolution in bits (1..6).

    Raises
    ------
    ValueError
        If ``bits`` is outside 1..6.
    """

    def __init__(self, bits: int) -> None:
        if not 1 <= bits <= 6:
            raise ValueError("bits must be 1..6")
        self.bits = bits
        self.levels = 2 ** bits
        self.step = 2.0 * np.pi / self.levels
        self.levels_array = np.arange(self.levels) * self.step

    def quantize(self, theta: np.ndarray) -> np.ndarray:
        """Snap phases to the nearest grid level, returned in ``[0, 2pi)``.

        Parameters
        ----------
        theta:
            Phase vector (rad), any range.

        Returns
        -------
        np.ndarray
            Quantized phases in ``[0, 2pi)``.
        """
        theta = np.asarray(theta, dtype=float)
        return (2.0 * np.pi * np.floor(theta / self.step + 0.5) / self.levels) % (2.0 * np.pi)

    def from_action(self, action: np.ndarray) -> np.ndarray:
        """Map a normalized action in ``[-1, 1]`` to quantized phases.

        The mapping is ``theta = (a + 1) * pi`` (so -1 -> 0, +1 -> 2pi) then
        quantized.  Deterministic and monotonic in the action.

        Parameters
        ----------
        action:
            Normalized action vector in ``[-1, 1]``.

        Returns
        -------
        np.ndarray
            Quantized phases in ``[0, 2pi)``.
        """
        return self.quantize((np.asarray(action, dtype=float) + 1.0) * np.pi)