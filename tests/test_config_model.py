"""Model persistence and configuration tests.

Covers acceptance areas 14-15:
  14. model save -> load -> identical deterministic prediction on the same obs
  15. config validation: bad values raise ConfigError; YAML round-trip preserves
      values
"""
from __future__ import annotations

import numpy as np
import pytest

from ris_autonomy.config import Config, ConfigError
from ris_autonomy.environment.wireless_env import RISAUTONOMYEnv
from ris_autonomy.rl.agent import create_agent, load_agent, save_agent


def test_model_save_load_identical_prediction(tmp_path):
    c = Config.from_yaml("configs/quick_test.yaml")
    env = RISAUTONOMYEnv(c, 5)
    model = create_agent(c, env, device="cpu")
    obs, _ = env.reset()
    # Deterministic prediction before saving.
    a1, _ = model.predict(obs, deterministic=True)
    path = str(tmp_path / "m.zip")
    save_agent(model, path, {"seed": 5, "timesteps": 0})
    model2, _ = load_agent(path, c, device="cpu")
    a2, _ = model2.predict(obs, deterministic=True)
    assert np.allclose(a1, a2)
    env.close()


def test_load_missing_model_raises(tmp_path):
    c = Config.from_yaml("configs/quick_test.yaml")
    from ris_autonomy.rl.agent import ModelLoadError

    with pytest.raises(ModelLoadError):
        load_agent(str(tmp_path / "nope.zip"), c)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ris": {"phase_bits": 0}},
        {"ris": {"phase_bits": 7}},
        {"system": {"bandwidth_hz": -1.0}},
        {"system": {"carrier_frequency_hz": 0.0}},
        {"channel": {"csi_error": 1.5}},
        {"channel": {"csi_error": -0.1}},
        {"users": {"mobility": "teleport"}},
        {"channel": {"model": "bogus"}},
        {"channel": {"fading": "bogus"}},
        {"system": {"modulation": "bogus"}},
        {"users": {"count": 0}},
        {"rl": {"n_steps_per_episode": 4}},
    ],
)
def test_invalid_config_raises(overrides):
    with pytest.raises(ConfigError):
        Config.from_dict(overrides)


def test_yaml_round_trip_preserves_values(tmp_path):
    c = Config.from_yaml("configs/quick_test.yaml")
    path = str(tmp_path / "roundtrip.yaml")
    c.to_yaml(path)
    c2 = Config.from_yaml(path)
    assert c.to_dict() == c2.to_dict()


def test_clone_with_overrides_nested():
    c = Config.from_yaml("configs/quick_test.yaml")
    c2 = c.clone_with_overrides(
        {"channel.csi_error": 0.25, "ris.rows": 6, "users.count": 3}
    )
    assert c2.channel.csi_error == 0.25
    assert c2.ris.rows == 6
    assert c2.users.count == 3
    # Original is unchanged (deep copy).
    assert c.channel.csi_error == 0.0
    assert c.ris.rows == 4


def test_snr_is_derived():
    c = Config.from_yaml("configs/quick_test.yaml")
    # snr_db = tx_power_dbm - noise_power_dbm.
    noise_dbm = 10.0 * np.log10(c.noise_power_w) + 30.0
    assert c.snr_db == pytest.approx(c.system.transmit_power_dbm - noise_dbm)
    # Higher transmit power -> higher SNR.
    c_hi = c.clone_with_overrides({"system.transmit_power_dbm": c.system.transmit_power_dbm + 10})
    assert c_hi.snr_db > c.snr_db