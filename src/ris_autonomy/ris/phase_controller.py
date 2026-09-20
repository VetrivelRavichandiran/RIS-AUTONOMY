"""Action-to-phase mapping for the RIS controller.

Bridges the RL action space to physical RIS phases.  In continuous mode the
normalized action ``[-1, 1]`` is mapped to ``[0, 2pi)`` and quantized; in
per-element discrete mode the action is an index into the level grid.
"""
from __future__ import annotations

import numpy as np

from .quantization import PhaseQuantizer


class PhaseController:
    """Map RL actions to quantized RIS phases.

    Parameters
    ----------
    quantizer:
        A :class:`PhaseQuantizer` for the configured phase resolution.
    """

    def __init__(self, quantizer: PhaseQuantizer) -> None:
        self.quantizer = quantizer

    def action_to_phases(self, action: np.ndarray) -> np.ndarray:
        """Map a continuous normalized action in ``[-1, 1]`` to quantized phases.

        Parameters
        ----------
        action:
            Action vector in ``[-1, 1]``.

        Returns
        -------
        np.ndarray
            Quantized phases in ``[0, 2pi)``.
        """
        return self.quantizer.from_action(action)

    def discrete_phases(self, indices: np.ndarray) -> np.ndarray:
        """Map per-element discrete indices to their grid phases.

        Parameters
        ----------
        indices:
            Integer indices in ``[0, 2**bits)`` per element.

        Returns
        -------
        np.ndarray
            Phases (rad) selected from the level grid.
        """
        return self.quantizer.levels_array[np.asarray(indices, dtype=int)]