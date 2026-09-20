"""Baseline controller tests.

Covers acceptance area 13:
  each baseline runs 5 steps without error and returns valid phases; no_ris
  makes H_eff == h_direct; greedy and conventional produce DIFFERENT phase
  vectors (proving they are genuinely distinct optimizers).
"""
from __future__ import annotations

import numpy as np
import pytest

from ris_autonomy.baselines import make_baseline
from ris_autonomy.config import Config
from ris_autonomy.environment.wireless_env import RISAUTONOMYEnv

_NAMES = ("no_ris", "random_ris", "greedy", "conventional")


def _config(users: int = 2) -> Config:
    return Config.from_yaml("configs/quick_test.yaml").clone_with_overrides({"users.count": users})


def _run_steps(name: str, users: int, steps: int = 5):
    env = RISAUTONOMYEnv(_config(users), 1)
    env.reset()
    controller = make_baseline(name)
    actions = []
    for _ in range(steps):
        action = controller.action(env)
        assert action.shape == (env.n,)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        actions.append(action)
    env.close()
    return env, actions


@pytest.mark.parametrize("name", _NAMES)
def test_each_baseline_runs_and_returns_valid_phases(name):
    env, actions = _run_steps(name, users=2)
    # Every action is a valid continuous action in [-1, 1].
    for a in actions:
        assert np.all((a >= -1.0) & (a <= 1.0))
    # The resulting RIS phases lie on the quantization grid.
    levels = env.ris.quantizer.levels_array
    dist = np.min(np.abs(env.ris.theta[:, None] - levels[None, :]), axis=1)
    assert np.all(dist < 1e-9)


def test_no_ris_disables_reflected_path():
    env = RISAUTONOMYEnv(_config(2), 1)
    env.reset()
    controller = make_baseline("no_ris")
    # Capture the channel the reward is measured on (before step regenerates it).
    ch_before = env.last_channel
    action = controller.action(env)
    obs, _, _, _, info = env.step(action)
    # With the RIS disabled the effective channel must equal the direct channel.
    assert np.allclose(info["H_eff"], ch_before.h_direct)
    env.close()


def test_greedy_and_conventional_differ():
    # With 2 users the two optimizers (sum-rate vs per-user) must produce
    # different phase configurations.
    env_g = RISAUTONOMYEnv(_config(2), 7)
    env_g.reset()
    theta_g = make_baseline("greedy").action(env_g)

    env_c = RISAUTONOMYEnv(_config(2), 7)
    env_c.reset()
    theta_c = make_baseline("conventional").action(env_c)

    # Same seed -> same channel, so the difference is purely the optimizer.
    assert np.allclose(env_g.last_channel.h_direct, env_c.last_channel.h_direct)
    assert not np.allclose(theta_g, theta_c)
    env_g.close()
    env_c.close()


def test_make_baseline_rejects_unknown():
    with pytest.raises(ValueError):
        make_baseline("does_not_exist")