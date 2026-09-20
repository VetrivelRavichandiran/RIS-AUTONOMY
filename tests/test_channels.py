"""Channel-model tests: shapes, path loss, effective channel, CSI error.

Covers acceptance areas 1-4:
  1. channel dimensions (K,M)/(N,M)/(K,N) for K=2,M=4,N=16; complex dtype
  2. path-loss monotonic in distance + known FSPL value
  3. effective-channel dimension check
  4. CSI error reduces correlation (0 < corr < 1 for small eps)
"""
from __future__ import annotations

import numpy as np
import pytest

from ris_autonomy.channels.channel_generator import ChannelGenerator
from ris_autonomy.channels.channel_models import apply_csi_error, compute_effective_channel
from ris_autonomy.channels.fading import fading_coefficients
from ris_autonomy.channels.pathloss import free_space_db, log_distance_db, pathloss_db, three_gpp_um_db
from ris_autonomy.config import Config


def _config() -> Config:
    """A 2-user, 4-antenna, 4x4-RIS config (K=2, M=4, N=16)."""
    return Config.from_yaml("configs/quick_test.yaml").clone_with_overrides({"users.count": 2})


def test_channel_shapes_and_dtype():
    c = _config()
    rng = np.random.default_rng(0)
    pos = rng.uniform([10, 10], [90, 90], (2, 2))
    ch = ChannelGenerator(c, rng).generate(pos)
    assert ch.h_direct.shape == (2, 4)
    assert ch.H_br.shape == (16, 4)
    assert ch.h_ru.shape == (2, 16)
    assert np.iscomplexobj(ch.h_direct)
    assert np.iscomplexobj(ch.H_br)
    assert np.iscomplexobj(ch.h_ru)


def test_fading_shapes_and_unit_power():
    rng = np.random.default_rng(1)
    for model, k in (("rician", 10.0), ("rayleigh", 0.0), ("no_fading", 0.0)):
        h = fading_coefficients(rng, (4, 4), model, k)
        assert h.shape == (4, 4)
        assert np.iscomplexobj(h)
    # Rayleigh is unit-variance on average: E[|h|^2] = 1.
    big = fading_coefficients(np.random.default_rng(2), (2000, 2000), "rayleigh", 0.0)
    assert abs((np.abs(big) ** 2).mean() - 1.0) < 0.05


def test_pathloss_monotonic_and_fspl_value():
    # Free-space path loss at 1 m, 1 GHz is ~32.45 dB.
    assert abs(free_space_db(1.0, 1e9) - 32.45) < 0.1
    # Path loss grows with distance for every model.
    for model in ("free_space", "log_distance", "3gpp_um"):
        near = pathloss_db(10.0, 28e9, model, True, 2.5)
        far = pathloss_db(20.0, 28e9, model, True, 2.5)
        assert far > near
    # 3GPP UMa NLOS is worse than LOS at the same distance.
    assert three_gpp_um_db(50.0, 28e9, False) > three_gpp_um_db(50.0, 28e9, True)
    # Log-distance model reduces to FSPL at the reference distance.
    ref = 1.0
    assert abs(log_distance_db(ref, 1e9, 2.5, ref) - free_space_db(ref, 1e9)) < 1e-6


def test_pathloss_rejects_invalid():
    with pytest.raises(ValueError):
        free_space_db(0.0, 1e9)
    with pytest.raises(ValueError):
        free_space_db(10.0, 0.0)


def test_effective_channel_dimensions():
    rng = np.random.default_rng(3)
    h = rng.standard_normal((2, 4)) + 1j * rng.standard_normal((2, 4))
    H = rng.standard_normal((16, 4)) + 1j * rng.standard_normal((16, 4))
    r = rng.standard_normal((2, 16)) + 1j * rng.standard_normal((2, 16))
    theta = rng.uniform(0, 2 * np.pi, 16)
    h_eff = compute_effective_channel(h, H, r, theta)
    assert h_eff.shape == (2, 4)
    assert np.iscomplexobj(h_eff)
    # With all-zero phases the reflected path is r @ H (no conjugation).
    h_eff0 = compute_effective_channel(h, H, r, np.zeros(16))
    assert np.allclose(h_eff0, h + r @ H)


def test_csi_error_reduces_correlation():
    rng = np.random.default_rng(4)
    h = rng.standard_normal((2, 4)) + 1j * rng.standard_normal((2, 4))
    for eps in (0.05, 0.2):
        h_tilde = apply_csi_error(h, np.random.default_rng(100), eps)
        corr = abs(np.vdot(h_tilde, h)) / (np.linalg.norm(h_tilde) * np.linalg.norm(h))
        assert 0.0 < corr < 1.0

    def _mean_corr(eps, n=50):
        cs = []
        for i in range(n):
            ht = apply_csi_error(h, np.random.default_rng(i), eps)
            cs.append(abs(np.vdot(ht, h)) / (np.linalg.norm(ht) * np.linalg.norm(h)))
        return float(np.mean(cs))

    # Larger CSI error -> lower correlation (in expectation).
    assert _mean_corr(0.05) > _mean_corr(0.3)