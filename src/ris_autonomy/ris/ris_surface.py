"""RIS surface state: phase vector, phase matrix, and switching cost.

The surface holds ``N = rows * columns`` elements with a current quantized
phase vector ``theta`` and the previous one.  ``switch_fraction`` (the fraction
of elements whose phase changed) drives the reward's switching-cost term.
"""
from __future__ import annotations

import numpy as np

from .quantization import PhaseQuantizer


class RISurface:
    """Stateful RIS phase configuration.

    Parameters
    ----------
    config:
        Validated configuration (``ris.rows``, ``ris.columns``,
        ``ris.phase_bits``, ``ris.amplitude``).
    """

    def __init__(self, config) -> None:
        self.n = config.ris.rows * config.ris.columns
        self.rows = config.ris.rows
        self.columns = config.ris.columns
        self.amplitude = config.ris.amplitude
        self.quantizer = PhaseQuantizer(config.ris.phase_bits)
        self.theta = np.zeros(self.n)
        self.previous_theta = self.theta.copy()

    def set_theta(self, theta: np.ndarray) -> None:
        """Apply a new (quantized) phase vector, remembering the previous one.

        Parameters
        ----------
        theta:
            Phase vector (rad); it is re-quantized to the grid.
        """
        self.previous_theta = self.theta.copy()
        self.theta = self.quantizer.quantize(theta)

    def phase_matrix(self) -> np.ndarray:
        """Return the diagonal phase matrix ``diag(a * e^{j theta})`` (N, N).

        Returns
        -------
        np.ndarray
            Complex diagonal matrix of shape (N, N).
        """
        return np.diag(self.amplitude * np.exp(1j * self.theta))

    def switch_fraction(self) -> float:
        """Fraction of elements whose quantized phase changed since the last step.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        return float(np.mean(self.theta != self.previous_theta))

    def reset(self, rng: np.random.Generator) -> None:
        """Randomize the phase vector on the quantization grid.

        Parameters
        ----------
        rng:
            Random generator.
        """
        self.theta = rng.choice(self.quantizer.levels_array, self.n)
        self.previous_theta = self.theta.copy()