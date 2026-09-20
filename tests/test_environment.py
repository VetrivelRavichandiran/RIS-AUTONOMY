"""Gymnasium environment tests: reset/step contract, bounds, truncation, render.

Covers acceptance area 12:
  reset returns (obs, info) with correct shape/dtype/bounds; step returns the
  5-tuple; obs stays bounded after 20 random steps; episode truncates at
  n_steps_per_episode; render() and close() work; invalid config raises ConfigError.
"""
from __future__ import annotations

import numpy as np
import pytest

from ris_autonomy.config import Config, ConfigError
from ris_autonomy.environment.wireless_env import RISAUTONOMYEnv


def _env(seed: int = 1, render_mode: str | None = None) -> RISAUTONOMYEnv:
    return RISAUTONOMYEnv(Config.from_yaml("configs/quick_test.yaml"), seed, render_mode)


def test_reset_returns_obs_info_with_correct_shape():
    env = _env()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert "episode" in info and "snr_db" in info and "n_ris_elements" in info
    assert info["n_ris_elements"] == env.n
    env.close()


def test_step_returns_five_tuple_and_obs_stays_bounded():
    env = _env()
    env.reset()
    for _ in range(20):
        out = env.step(env.action_space.sample())
        assert len(out) == 5
        obs, reward, terminated, truncated, info = out
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not terminated  # episodes truncate, never terminate early
        # Metrics are finite.
        for key in ("rate_sum", "ee", "fairness", "outage"):
            assert np.isfinite(info[key])
    env.close()


def test_episode_truncates_at_n_steps():
    env = _env()
    n_steps = env.config.rl.n_steps_per_episode
    env.reset()
    truncated_at = None
    for t in range(n_steps + 5):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if truncated:
            truncated_at = t + 1
            break
    assert truncated_at == n_steps
    assert not terminated
    env.close()


def test_render_and_close():
    env = _env(render_mode="human")
    env.reset()
    fig = env.render()
    assert fig is not None
    env.close()
    # rgb_array mode returns a numeric array.
    env2 = _env(render_mode="rgb_array")
    env2.reset()
    arr = env2.render()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3 and arr.shape[2] == 4
    env2.close()


def test_get_topology():
    env = _env()
    env.reset()
    topo = env.get_topology()
    assert set(topo) >= {"bs", "ris", "ues", "distances"}
    assert topo["ues"].shape == (env.config.users.count, 2)
    env.close()


def test_invalid_config_raises():
    with pytest.raises(ConfigError):
        Config.from_dict({"ris": {"phase_bits": 0}})
    with pytest.raises(ConfigError):
        Config.from_dict({"system": {"bandwidth_hz": -1.0}})
    with pytest.raises(ConfigError):
        Config.from_dict({"channel": {"csi_error": 1.5}})
    with pytest.raises(ConfigError):
        Config.from_dict({"users": {"mobility": "teleport"}})


def test_action_mode_discrete_space():
    c = Config.from_yaml("configs/quick_test.yaml").clone_with_overrides(
        {"rl.action_mode": "per_element_discrete"}
    )
    env = RISAUTONOMYEnv(c, 1)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
    env.close()