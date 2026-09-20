"""Physical-layer metric tests: SINR, rate, BER, EE, fairness, outage.

Covers acceptance areas 7-10:
  7. SINR formula vs a hand-computed 2-user / 2-antenna case
  8. rate = B*log2(1+SINR) exact
  9. BER monotonic decreasing in SINR, in [0,1]
  10. EE = rate/power exact; Jain fairness = 1 for equal rates; outage threshold
"""
from __future__ import annotations

import numpy as np
import pytest

from ris_autonomy.communications.achievable_rate import rate_bps, sum_rate
from ris_autonomy.communications.ber import ber_approx
from ris_autonomy.communications.energy_efficiency import energy_efficiency, total_power_w
from ris_autonomy.communications.signal_model import mrt_precoding, received_signal
from ris_autonomy.communications.sinr import sinr_db
from ris_autonomy.evaluation.metrics import fairness_jain, outage_probability


def test_sinr_hand_computed_two_user_case():
    # Identity effective channel, MRT -> identity precoder, P=1 W split over 2 users.
    h_eff = np.eye(2, dtype=complex)
    w = mrt_precoding(h_eff)
    p_tx_w = 1.0
    noise_w = 0.1
    sinr, powers = received_signal(h_eff, w, p_tx_w, noise_w)
    # Per-user signal power = |h_k . W[:,k]|^2 * P/K = 1 * 1/2 = 0.5; no interference.
    assert np.allclose(sinr, [0.5 / noise_w, 0.5 / noise_w])
    assert np.allclose(powers["signal"], [0.5, 0.5])
    assert np.allclose(powers["interference"], [0.0, 0.0])
    assert powers["noise"] == noise_w


def test_sinr_db_conversion():
    assert abs(sinr_db(np.array([1.0]))[0]) < 1e-9
    assert abs(sinr_db(np.array([10.0]))[0] - 10.0) < 1e-9
    # Zero SINR is finite (floored), not -inf.
    assert np.isfinite(sinr_db(np.array([0.0]))[0])


def test_rate_exact():
    # R = B*log2(1+SINR); at SINR=1, B=100 -> 100 bps.
    assert rate_bps(np.array([1.0]), 100.0)[0] == 100.0
    assert rate_bps(np.array([3.0]), 100.0)[0] == pytest.approx(100.0 * np.log2(4.0))
    assert sum_rate(np.array([1.0, 3.0]), 100.0) == pytest.approx(100.0 * (1 + 2))


def test_ber_monotonic_and_bounded():
    for mod in ("qpsk", "16qam", "64qam"):
        snrs = np.array([0.1, 1.0, 5.0, 20.0])
        ber = ber_approx(snrs, mod)
        assert np.all((ber >= 0) & (ber <= 1))
        # BER decreases as SINR increases.
        assert np.all(np.diff(ber) < 0)
    # At high SINR every modulation's BER is tiny.
    assert ber_approx(np.array([100.0]), "qpsk")[0] < 1e-6


def test_ber_rejects_unknown_modulation():
    with pytest.raises(ValueError):
        ber_approx(np.array([1.0]), "unknown")


def test_energy_efficiency_exact():
    assert energy_efficiency(10.0, 2.0) == 5.0
    # total_power = tx + N*p_elem + circuit.
    assert total_power_w(1.0, 4, 0.5, 0.25) == 1.0 + 4 * 0.5 + 0.25


def test_fairness_jain_equal_and_unequal():
    # Equal rates -> fairness 1.
    assert fairness_jain(np.array([5.0, 5.0, 5.0])) == pytest.approx(1.0)
    # One user dominates -> fairness < 1, approaching 1/K.
    assert fairness_jain(np.array([100.0, 1.0, 1.0])) < 1.0
    # All-zero rates -> 0 (no division by zero).
    assert fairness_jain(np.array([0.0, 0.0])) == 0.0


def test_outage_probability_threshold():
    rates = np.array([0.5e6, 1.5e6, 2.0e6])
    # Target 1e6: one user below -> 1/3.
    assert outage_probability(rates, 1e6) == pytest.approx(1.0 / 3.0)
    # Target above all -> everyone in outage.
    assert outage_probability(rates, 1e9) == 1.0
    # Target below all -> no outage.
    assert outage_probability(rates, 1e3) == 0.0