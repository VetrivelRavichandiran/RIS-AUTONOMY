"""Gymnasium environment for adaptive RIS phase optimization.

The environment models a BS -> RIS -> UE (plus direct BS -> UE) MISO system in
which a deep-RL agent chooses the RIS phase vector each step to maximize a
communication reward.

Channel-coherence semantics (important for correctness): within a single step
the channel is fixed.  The agent observes the (CSI-errored) channel, chooses a
phase vector, and the achievable rate is measured on that *same* channel with
the new phases applied.  Only after the reward is computed does the environment
advance the UEs and regenerate the channel for the *next* observation.  This
guarantees that a controller (DRL or conventional) always acts on the channel
it observed, matching a real system operating within a coherence interval.

Step order:

1. map the action to a quantized RIS phase vector (deterministic);
2. measure the rate/reward on the current channel with the new phases;
3. advance UE positions (mobility) and regenerate the channel;
4. build the next observation from the new channel + current phases.

Observation layout (see :mod:`ris_autonomy.environment.state`) is a float32
vector clipped to [-3, 3].  Action space is a continuous Box in [-1, 1]^N (mapped
to [0, 2π) then quantized) or, in ``per_element_discrete`` mode, a
MultiDiscrete([2**bits] * N).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..channels.channel_generator import ChannelGenerator
from ..channels.channel_models import apply_csi_error, compute_effective_channel
from ..communications.achievable_rate import rate_bps
from ..communications.ber import ber_approx
from ..communications.energy_efficiency import energy_efficiency, ris_power_w, total_power_w
from ..communications.signal_model import mrt_precoding, received_signal
from ..communications.sinr import sinr_db
from ..rl.rewards import RewardFunction
from .mobility import MobilityModel, random_initial_positions
from .state import StateBuilder, StepData, observation_dim
from ..ris.ris_surface import RISurface
from ..ris.phase_controller import PhaseController


class RISAUTONOMYEnv(gym.Env):
    """Adaptive RIS optimization environment (Gymnasium API)."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config, seed: int | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.seed0 = seed if seed is not None else config.experiment.seed
        self.render_mode = render_mode
        self.n = config.ris.rows * config.ris.columns
        self.ris = RISurface(config)
        self.controller = PhaseController(self.ris.quantizer)
        self.builder = StateBuilder(config)
        self.reward_fn = RewardFunction(config)
        self.p_tx_w = 10.0 ** ((config.system.transmit_power_dbm - 30.0) / 10.0)
        self.noise_power_w = config.noise_power_w

        self.observation_space = spaces.Box(-3.0, 3.0, (observation_dim(config),), np.float32)
        if config.rl.action_mode == "continuous":
            self.action_space = spaces.Box(-1.0, 1.0, (self.n,), np.float32)
        else:
            self.action_space = spaces.MultiDiscrete([2 ** config.ris.phase_bits] * self.n)

        self.episode = -1
        self.rng = np.random.default_rng(self.seed0)
        self.positions = np.zeros((config.users.count, 2))
        self.velocities = np.zeros((config.users.count, 2))
        self.t = 0
        self.last_channel = None
        self._fig = None
        # When False the RIS path is disabled and H_eff = h_direct (NoRIS baseline).
        self.ris_enabled = True

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment; returns ``(obs, info)``."""
        super().reset(seed=seed)
        self.episode += 1
        self.rng = np.random.default_rng(seed if seed is not None else self.seed0 + self.episode)
        c = self.config
        self.positions = random_initial_positions(self.rng, c.users.count, c.scenario.ue_region)
        speed = c.users.velocity_mps
        if not isinstance(speed, list):
            speed = np.asarray([speed] * c.users.count)
        else:
            speed = np.asarray(speed)
        angles = self.rng.uniform(0, 2 * np.pi, c.users.count)
        self.velocities = np.column_stack([np.cos(angles), np.sin(angles)]) * np.asarray(speed).reshape(-1, 1)
        self.mobility = MobilityModel(c, self.rng)
        self.ris.reset(self.rng)
        self.t = 0
        self._regenerate_channel()
        obs, info = self._measure()
        return obs, {"episode": self.episode, "snr_db": c.snr_db, "n_ris_elements": self.n, **info}

    # ------------------------------------------------------------------ channel
    def _regenerate_channel(self) -> None:
        """Generate the channel for the current UE positions (stored on the env)."""
        self.last_channel = ChannelGenerator(self.config, self.rng).generate(self.positions)

    # ----------------------------------------------------------------- measure
    def _measure(self):
        """Compute metrics + observation from the *current* channel and phases.

        Uses ``self.last_channel`` (the channel the agent observed) and
        ``self.ris.theta`` (the current phases).  The rate is measured on the
        true channel; the observation is built from the CSI-errored channel.
        Returns ``(obs, info)``.
        """
        c = self.config
        ch = self.last_channel
        theta = self.ris.theta
        if self.ris_enabled:
            h_eff = compute_effective_channel(ch.h_direct, ch.H_br, ch.h_ru, theta)
        else:
            # NoRIS: the reflected path is disabled, so only the direct link counts.
            h_eff = ch.h_direct
        h_tilde = apply_csi_error(h_eff, self.rng, c.channel.csi_error)
        w = mrt_precoding(h_tilde)
        sinr, powers = received_signal(h_eff, w, self.p_tx_w, self.noise_power_w)
        rates = rate_bps(sinr, c.system.bandwidth_hz)
        sum_rate = float(rates.sum())
        power = total_power_w(self.p_tx_w, self.n, c.system.ris_element_power_w, c.system.bs_circuit_power_w)

        info = {
            "rate_sum": sum_rate,
            "rate_per_user": rates,
            "sinr_db": sinr_db(sinr),
            "ee": energy_efficiency(sum_rate, power),
            "ber": ber_approx(sinr, c.system.modulation),
            "outage": float(np.mean(rates < c.experiment.outage_target_bps)),
            "fairness": float(sum_rate * sum_rate / (len(rates) * np.sum(rates * rates))) if np.sum(rates * rates) > 0 else 0.0,
            "switching_cost": self.ris.switch_fraction(),
            "p_tx_w": self.p_tx_w,
            "p_ris_w": ris_power_w(self.n, c.system.ris_element_power_w),
            "interference_power": powers["interference"],
            "channel": ch,
            "H_eff": h_eff,
            "H_eff_tilde": h_tilde,
        }
        d = StepData(
            ch.h_direct, h_tilde, self.ris.theta, self.ris.previous_theta,
            self.positions, self.velocities, sinr, rates, c.snr_db,
            c.channel.csi_error, powers["interference"],
        )
        return self.builder.build(d), info

    # -------------------------------------------------------------------- step
    def step(self, action):
        """Advance one step; returns ``(obs, reward, terminated, truncated, info)``.

        The reward is measured on the channel the agent observed (current
        channel with the new phases); the returned observation reflects the
        *next* channel (after UE motion + regeneration).
        """
        if self.config.rl.action_mode == "continuous":
            self.ris.set_theta(self.controller.action_to_phases(action))
        else:
            self.ris.set_theta(self.controller.discrete_phases(np.asarray(action)))

        # Reward: current channel + new phases (the channel the agent observed).
        _, info = self._measure()
        reward, breakdown = self.reward_fn(info)
        info["reward_breakdown"] = breakdown

        # Advance: move UEs, regenerate the channel, build the next observation.
        self.t += 1
        self.positions = self.mobility.update(self.positions, self.velocities, self.config.scenario.step_dt_s)
        self._regenerate_channel()
        obs_next, _ = self._measure()

        truncated = self.t >= self.config.rl.n_steps_per_episode
        return obs_next, float(reward), False, truncated, info

    # ----------------------------------------------------------------- topology
    def get_topology(self) -> dict:
        """Return topology data for visualization (positions + distances)."""
        bs = np.asarray(self.config.scenario.bs_position)
        ues = np.asarray(self.positions)
        return {
            "bs": bs,
            "ris": np.asarray(self.config.scenario.ris_position),
            "ues": ues,
            "distances": np.linalg.norm(ues - bs, axis=1),
        }

    # ------------------------------------------------------------------- render
    def render(self):
        """Render the topology.

        ``render_mode="human"`` returns a matplotlib Figure;
        ``render_mode="rgb_array"`` returns a small RGBA array.
        """
        from ..visualization.topology import plot_topology

        data = self.get_topology()
        if self.render_mode == "rgb_array":
            fig = plot_topology(data, title=f"RIS topology (episode {self.episode})")
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            return buf
        self._fig = plot_topology(data, title=f"RIS topology (episode {self.episode})")
        return self._fig

    # -------------------------------------------------------------------- close
    def close(self) -> None:
        """Release any rendering resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None