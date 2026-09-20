"""Small-scale fading coefficient generators (complex, unit-variance).

All functions draw from a ``numpy.random.Generator`` and return complex128
arrays of the requested shape.  The complex Gaussian is
``(N(0,1) + j N(0,1)) / sqrt(2)`` so that ``E[|h|^2] = 1``.
"""
from __future__ import annotations

import numpy as np


def _complex_gaussian(rng: np.random.Generator, shape) -> np.ndarray:
    """i.i.d. complex Gaussian ``CN(0, 1)`` (unit variance)."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def rayleigh_fading(rng: np.random.Generator, shape) -> np.ndarray:
    """Rayleigh fading: i.i.d. ``CN(0, 1)`` per tap."""
    return _complex_gaussian(rng, shape)


def rician_fading(rng: np.random.Generator, shape, k_db: float) -> np.ndarray:
    """Rician fading with K-factor ``k_db`` (dB).

    ``h = sqrt(K/(K+1)) e^{j phi} + sqrt(1/(K+1)) CN(0,1)`` with a random
    LOS phase ``phi ~ U[0, 2pi)`` per tap.  Unit total power.
    """
    k = 10.0 ** (k_db / 10.0)
    los = np.sqrt(k / (k + 1.0)) * np.exp(1j * rng.uniform(0, 2 * np.pi, shape))
    scatter = np.sqrt(1.0 / (k + 1.0)) * _complex_gaussian(rng, shape)
    return los + scatter


def no_fading(rng: np.random.Generator, shape) -> np.ndarray:
    """No fading: all-ones (deterministic unit gain)."""
    return np.ones(shape, dtype=np.complex128)


def fading_coefficients(rng: np.random.Generator, shape, model: str, k_db: float = 10.0) -> np.ndarray:
    """Dispatch to the named fading model.

    Parameters
    ----------
    model:
        One of ``"rician"``, ``"rayleigh"``, ``"no_fading"``.
    k_db:
        Rician K-factor (dB), used only by ``"rician"``.
    """
    models = {
        "rician": lambda: rician_fading(rng, shape, k_db),
        "rayleigh": lambda: rayleigh_fading(rng, shape),
        "no_fading": lambda: no_fading(rng, shape),
    }
    if model not in models:
        raise ValueError(f"unknown fading model {model!r}")
    return models[model]()