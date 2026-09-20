"""Observation-vector construction for the RIS environment.

The observation is a fixed-layout float32 vector, clipped to ``[-3, 3]``.
Each feature group is enabled by ``state.components`` and normalized by a
per-group scale before concatenation.  The exact layout (in order) is:

Per user ``k`` (K users, M BS antennas, N RIS elements):

* ``direct_magnitude``  -- ``|h_direct[k]|^2``            (M values, scale 1)
* ``direct_phase``      -- ``angle(h_direct[k])``          (M values, scale pi)
* ``effective_magnitude``-- ``|H_eff[k]|^2``               (M values, scale 1)
* ``effective_phase``   -- ``angle(H_eff[k])``             (M values, scale pi)
* ``sinr``              -- linear SINR of user k           (1 value, scale 1)
* ``rate``              -- rate of user k                  (1 value, scale rate_norm)
* ``position``          -- (x, y) of user k                (2 values, scale boundary)
* ``velocity``          -- (vx, vy) of user k              (2 values, scale 30 m/s)

Global:

* ``current_phase``     -- theta (N values, scale pi)
* ``previous_phase``    -- theta_prev (N values, scale pi)
* ``snr``               -- link SNR in dB                  (1 value, scale 20)
* ``csi_quality``       -- configured CSI error            (1 value, scale 1)
* ``interference``      -- per-user interference power (K values, scale p_tx/K)

``observation_dim`` is the single source of truth for the vector length.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepData:
    """All per-step quantities needed to build an observation.

    Attributes
    ----------
    h_direct_tilde:
        Observed direct channel, shape ``(K, M)`` complex.
    H_eff_tilde:
        Observed effective channel, shape ``(K, M)`` complex.
    theta:
        Current RIS phases (rad), shape ``(N,)``.
    theta_prev:
        Previous RIS phases (rad), shape ``(N,)``.
    positions:
        UE positions, shape ``(K, 2)``.
    velocities:
        UE velocities, shape ``(K, 2)``.
    sinr_lin:
        Linear SINR per user, shape ``(K,)``.
    rates:
        Achievable rate per user (bps), shape ``(K,)``.
    snr_db:
        Link SNR in dB (scalar).
    csi_error:
        Configured CSI error fraction (scalar).
    interference_power:
        Per-user interference power (W), shape ``(K,)``.
    """

    h_direct_tilde: object
    H_eff_tilde: object
    theta: object
    theta_prev: object
    positions: object
    velocities: object
    sinr_lin: object
    rates: object
    snr_db: float
    csi_error: float
    interference_power: object


class StateBuilder:
    """Build the normalized, clipped observation vector from :class:`StepData`.

    Parameters
    ----------
    config:
        Validated configuration (component toggles, scales, boundary, power).
    """

    def __init__(self, config) -> None:
        self.config = config

    def build(self, d: StepData) -> np.ndarray:
        """Return the observation vector (float32, clipped to [-3, 3]).

        Parameters
        ----------
        d:
            Per-step data to encode.

        Returns
        -------
        numpy.ndarray
            1-D float32 observation, length ``observation_dim(config)``.
        """
        c = self.config
        q = c.state.components
        parts: list[float] = []

        def add(key: str, x: object, scale: float | list = 1.0) -> None:
            """Append a normalized, flattened feature group if enabled."""
            if q.get(key, False):
                parts.extend((np.asarray(x) / scale).ravel())

        add("direct_magnitude", np.abs(d.h_direct_tilde) ** 2)
        add("direct_phase", np.angle(d.h_direct_tilde), np.pi)
        add("effective_magnitude", np.abs(d.H_eff_tilde) ** 2)
        add("effective_phase", np.angle(d.H_eff_tilde), np.pi)
        add("sinr", d.sinr_lin)
        add("rate", d.rates, c.reward.rate_norm)
        add("position", d.positions, [c.scenario.boundary_width_m, c.scenario.boundary_height_m])
        add("velocity", d.velocities, 30.0)
        add("current_phase", d.theta, np.pi)
        add("previous_phase", d.theta_prev, np.pi)
        add("snr", [d.snr_db], 20.0)
        add("csi_quality", [d.csi_error])
        p_tx_w = 10.0 ** ((c.system.transmit_power_dbm - 30.0) / 10.0)
        add("interference", d.interference_power, p_tx_w / c.users.count)

        obs = np.clip(np.asarray(parts, dtype=np.float32), -3.0, 3.0)
        if c.state.observation_noise_std:
            obs = np.clip(
                obs + np.random.normal(0.0, c.state.observation_noise_std, obs.shape),
                -3.0,
                3.0,
            )
        return obs


def observation_dim(config) -> int:
    """Return the observation-vector length for the enabled components.

    This is the single source of truth for the environment's
    ``observation_space`` shape.

    Parameters
    ----------
    config:
        Validated configuration.

    Returns
    -------
    int
        Number of observation features.
    """
    n = config.ris.rows * config.ris.columns
    k = config.users.count
    m = config.bs.antennas
    q = config.state.components
    per_user_per_antenna = k * m * (
        q["direct_magnitude"]
        + q["direct_phase"]
        + q["effective_magnitude"]
        + q["effective_phase"]
    )
    per_user = k * (
        q["sinr"] + q["rate"] + 2 * q["position"] + 2 * q["velocity"] + q["interference"]
    )
    per_ris = n * (q["current_phase"] + q["previous_phase"])
    global_terms = q["snr"] + q["csi_quality"]
    return per_user_per_antenna + per_user + per_ris + global_terms