"""End-to-end smoke test: env loop + a short PPO run that saves and reloads.

Covers acceptance area 16 (part): env -> reset -> random actions -> finite
metrics -> close; plus a short PPO training run that completes and whose model
loads back.
"""
from __future__ import annotations

import numpy as np

from ris_autonomy.config import Config
from ris_autonomy.environment.wireless_env import RISAUTONOMYEnv
from ris_autonomy.rl.agent import load_agent
from ris_autonomy.rl.training import train


def test_env_random_actions_finite():
    c = Config.from_yaml("configs/quick_test.yaml")
    env = RISAUTONOMYEnv(c, 123)
    obs, _ = env.reset()
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        assert np.isfinite(info["rate_sum"])
        if terminated or truncated:
            break
    env.close()


def test_short_ppo_run_saves_and_loads(tmp_path):
    c = Config.from_yaml("configs/quick_test.yaml")
    result = train(c, seed=3, timesteps=32, output_dir=str(tmp_path))
    assert result["model_path"].endswith(".zip")
    model, meta = load_agent(result["model_path"], c)
    assert model is not None
    assert isinstance(meta, dict)
    # The best model (callback or fallback) must also exist and load.
    import os

    best = os.path.join(str(tmp_path), "models", "final", "best_model.zip")
    assert os.path.exists(best)
    model_b, _ = load_agent(best, c)
    assert model_b is not None