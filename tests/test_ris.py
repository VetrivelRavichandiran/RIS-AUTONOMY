"""RIS quantization, surface, and action-mapping tests.

Covers acceptance areas 5-6:
  5. quantizer: levels=2^B; output in [0,2pi) and on the grid; from_action
     deterministic, in-range, quantized; switch_fraction in [0,1]
  6. action mapping [-1,1] -> [0,2pi) monotonic + deterministic
"""
from __future__ import annotations

import numpy as np
import pytest

from ris_autonomy.config import Config
from ris_autonomy.ris.phase_controller import PhaseController
from ris_autonomy.ris.quantization import PhaseQuantizer
from ris_autonomy.ris.ris_surface import RISurface


def test_quantizer_levels_and_grid():
    for bits in (1, 2, 3, 4):
        q = PhaseQuantizer(bits)
        assert q.levels == 2**bits
        assert q.step == 2 * np.pi / q.levels
        theta = np.linspace(-2.0, 8.0, 99)
        out = q.quantize(theta)
        assert np.all((out >= 0) & (out < 2 * np.pi))
        # Output must lie exactly on the level grid.
        dist = np.min(np.abs(out[:, None] - q.levels_array[None, :]), axis=1)
        assert np.all(dist < 1e-9)
        # Exactly 2^B distinct values across a wide input range.
        wide = np.linspace(0, 2 * np.pi * 3, 1000)
        assert len(np.unique(np.round(q.quantize(wide), 9))) == 2**bits


def test_quantizer_rejects_bad_bits():
    with pytest.raises(ValueError):
        PhaseQuantizer(0)
    with pytest.raises(ValueError):
        PhaseQuantizer(7)


def test_from_action_deterministic_in_range_quantized():
    q = PhaseQuantizer(2)
    a = np.linspace(-1, 1, 17)
    out1 = q.from_action(a)
    out2 = q.from_action(a)
    assert np.allclose(out1, out2)
    assert np.all((out1 >= 0) & (out1 < 2 * np.pi))
    dist = np.min(np.abs(out1[:, None] - q.levels_array[None, :]), axis=1)
    assert np.all(dist < 1e-9)
    # -1 maps to phase 0; +1 wraps to phase 0.
    assert q.from_action(np.array([-1.0]))[0] == 0.0
    assert q.from_action(np.array([1.0]))[0] == 0.0


def test_action_mapping_monotonic():
    q = PhaseQuantizer(4)
    # Test on [-1, 0.9) so the +1 -> 2pi -> 0 wrap-around at the top edge is
    # excluded; within the range the mapping is monotone non-decreasing.
    a = np.linspace(-1, 0.9, 65)
    phases = q.from_action(a)
    assert np.all(np.diff(phases) >= -1e-9)
    assert phases[-1] > phases[0]


def test_surface_switch_fraction_bounds():
    c = Config.from_yaml("configs/quick_test.yaml")
    rng = np.random.default_rng(0)
    ris = RISurface(c)
    ris.reset(rng)
    assert 0.0 <= ris.switch_fraction() <= 1.0
    # Changing every phase to a different level -> switch_fraction 1.
    ris.set_theta(ris.theta + ris.quantizer.step)
    assert ris.switch_fraction() == 1.0
    # No change -> 0.
    ris.set_theta(ris.theta)
    assert ris.switch_fraction() == 0.0
    # Phase matrix is diagonal with the configured amplitude.
    n = c.ris.rows * c.ris.columns
    pm = ris.phase_matrix()
    assert pm.shape == (n, n)
    assert np.allclose(np.abs(np.diag(pm)), c.ris.amplitude)
    off = pm - np.diag(np.diag(pm))
    assert np.allclose(off, 0.0)


def test_phase_controller_discrete():
    q = PhaseQuantizer(2)
    pc = PhaseController(q)
    idx = np.array([0, 1, 2, 3])
    assert np.allclose(pc.discrete_phases(idx), q.levels_array)
    # Continuous path agrees with the quantizer.
    a = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(pc.action_to_phases(a), q.from_action(a))