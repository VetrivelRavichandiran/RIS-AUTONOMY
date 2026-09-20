"""Reward-function tests.

Covers acceptance area 11:
  reward is finite; the breakdown sums to the reward; zero weights give zero
  contribution; the switching penalty is active when phases change.
"""
from __future__ import annotations

import numpy as np
import pytest

from ris_autonomy.config import Config
from ris_autonomy.rl.rewards import RewardFunction


def _info(rate_sum=1e7, sinr_db=(20.0,), ee=1e6, switching=0.0):
    return {
        "rate_sum": rate_sum,
        "sinr_db": np.asarray(sinr_db, dtype=float),
        "ee": ee,
        "switching_cost": switching,
    }


def test_reward_finite_and_breakdown_sums_to_reward():
    rf = RewardFunction(Config())
    reward, breakdown = rf(_info())
    assert np.isfinite(reward)
    assert reward == pytest.approx(sum(breakdown.values()), abs=1e-12)
    # The rate term dominates and is positive.
    assert breakdown["rate"] > 0


def test_zero_weights_give_zero_contribution():
    c = Config().clone_with_overrides(
        {
            "reward.rate_weight": 0.0,
            "reward.sinr_weight": 0.0,
            "reward.energy_weight": 0.0,
            "reward.switching_weight": 0.0,
            "reward.constraint_weight": 0.0,
        }
    )
    rf = RewardFunction(c)
    reward, breakdown = rf(_info())
    assert reward == 0.0
    assert all(v == 0.0 for v in breakdown.values())


def test_switching_penalty_active_when_phases_change():
    base = Config()
    no_switch, _ = RewardFunction(base)(_info(switching=0.0))
    with_switch, breakdown = RewardFunction(base)(_info(switching=1.0))
    # A full phase change subtracts the switching term.
    assert with_switch < no_switch
    assert breakdown["switching"] == pytest.approx(-base.reward.switching_weight * 1.0)


def test_rate_term_scales_with_rate():
    base = Config()
    low, _ = RewardFunction(base)(_info(rate_sum=1e5))
    high, _ = RewardFunction(base)(_info(rate_sum=1e7))
    assert high > low