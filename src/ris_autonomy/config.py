"""Validated, dataclass-based configuration for RIS-AUTONOMY.

Everything tunable about a simulation lives here so that no magic constants are
scattered through the code.  A :class:`Config` is built from YAML (or a dict),
validated (raising :class:`ConfigError` with a precise message), and can be
round-tripped back to YAML or deep-copied with nested overrides
(:meth:`Config.clone_with_overrides`, used by the experiment engine).

SNR is *derived*, not stored: ``snr_db = transmit_power_dbm - noise_power_dbm``
where the noise power is thermal noise over the bandwidth plus the noise figure.
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import yaml

logger = logging.getLogger(__name__)

# Physical constants
K_BOLZMANN = 1.380649e-23  # J
T0 = 290.0  # K (reference noise temperature)


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


def _default_state_components() -> dict:
    """All observation components enabled by default."""
    return {
        "direct_magnitude": True,
        "direct_phase": True,
        "effective_magnitude": True,
        "effective_phase": True,
        "sinr": True,
        "rate": True,
        "position": True,
        "velocity": True,
        "current_phase": True,
        "previous_phase": True,
        "snr": True,
        "csi_quality": True,
        "interference": True,
    }


@dataclass
class System:
    """RF system parameters."""

    carrier_frequency_hz: float = 28e9
    bandwidth_hz: float = 100e6
    transmit_power_dbm: float = 30.0
    noise_figure_db: float = 5.0
    modulation: str = "qpsk"  # qpsk | 16qam | 64qam
    ris_element_power_w: float = 0.005
    bs_circuit_power_w: float = 0.0


@dataclass
class BS:
    """Base-station parameters."""

    antennas: int = 4


@dataclass
class RIS:
    """Reconfigurable intelligent surface parameters."""

    rows: int = 8
    columns: int = 8
    phase_bits: int = 2
    amplitude: float = 1.0
    switching_cost: float = 0.1


@dataclass
class Users:
    """User-equipment parameters."""

    count: int = 2
    mobility: str = "static"  # static | linear | random_waypoint
    velocity_mps: object = 2.0  # float or list[float] (one per user)


@dataclass
class Scenario:
    """2-D scenario geometry (meters)."""

    boundary_width_m: float = 100.0
    boundary_height_m: float = 100.0
    bs_position: list = field(default_factory=lambda: [0.0, 0.0])
    ris_position: list = field(default_factory=lambda: [50.0, 50.0])
    ue_region: list = field(default_factory=lambda: [10.0, 10.0, 90.0, 90.0])
    bs_height_m: float = 25.0
    step_dt_s: float = 1.0


@dataclass
class Channel:
    """Channel-model parameters.

    ``direct_link`` selects the direct BS->UE condition:

    * ``"los"``     -- distance-based LOS/NLOS (``los_distance_m`` threshold).
    * ``"blocked"`` -- the direct path is obstructed (NLOS plus ``blockage_db``
      of extra loss) while the two RIS hops stay LOS.  This is the canonical
      mmWave RIS setting in which the reflected path is the useful one.
    """

    model: str = "3gpp_um"  # 3gpp_um | free_space | log_distance
    fading: str = "rician"  # rician | rayleigh | no_fading
    rician_k_db: float = 10.0
    pathloss_exponent: float = 2.5
    csi_error: float = 0.0
    los_distance_m: float = 60.0
    direct_link: str = "blocked"  # los | blocked
    blockage_db: float = 40.0


@dataclass
class State:
    """Observation-vector composition and noise."""

    components: dict = field(default_factory=_default_state_components)
    observation_noise_std: float = 0.0


@dataclass
class RL:
    """Reinforcement-learning hyperparameters."""

    algorithm: str = "ppo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01
    clip_range: float = 0.2
    timesteps: int = 100_000
    n_steps_per_episode: int = 100
    action_mode: str = "continuous"  # continuous | per_element_discrete
    device: str = "auto"  # auto | cpu | cuda
    net_arch: str = "default"  # default | small | large
    checkpoint_every: int = 10_000


@dataclass
class Reward:
    """Multi-objective reward weights and normalization scales.

    The reward is ``sum(weight_i * metric_i / norm_i) - switching_weight *
    switch_fraction - constraint_weight * violation``.  The ``*_norm`` scales
    keep every term O(1) so no single metric numerically dominates.
    """

    rate_weight: float = 1.0
    sinr_weight: float = 0.5
    energy_weight: float = 0.2
    switching_weight: float = 0.1
    constraint_weight: float = 0.0
    rate_norm: float = 1e4
    sinr_norm_db: float = 20.0
    ee_norm: float = 1e4


@dataclass
class Experiment:
    """Experiment / sweep ranges and evaluation settings."""

    seed: int = 42
    episodes: int = 10
    snr_sweep_db: list = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    ris_size_sweep: list = field(default_factory=lambda: [16, 32, 64, 128])
    phase_bits_sweep: list = field(default_factory=lambda: [1, 2, 3, 4])
    user_count_sweep: list = field(default_factory=lambda: [1, 2, 3, 4])
    velocity_sweep_mps: list = field(default_factory=lambda: [0, 1, 5, 10, 20])
    csi_error_sweep: list = field(default_factory=lambda: [0, 0.05, 0.1, 0.2, 0.3])
    power_sweep_dbm: list = field(default_factory=lambda: [10, 20, 30, 40])
    distance_sweep_m: list = field(default_factory=lambda: [20, 40, 60, 80])
    fading_sweep: list = field(default_factory=lambda: ["rician", "rayleigh"])
    outage_target_bps: float = 1e6
    # Timesteps used to train a DRL model for a sweep when none is supplied
    # via ``--model``.  0 means "reuse the supplied model only" (a sweep whose
    # values change the observation dimension then raises).  When >0 the
    # experiment trains a model at the base config (or per value, for sweeps
    # that change the observation dimension such as user_count / ris_size).
    sweep_train_timesteps: int = 0


@dataclass
class Benchmark:
    """Benchmark (baseline comparison) settings."""

    drl_model_path: object = None
    train_drl_if_missing: bool = True
    drl_timesteps_if_missing: int = 5000
    episodes: int = 5


@dataclass
class Config:
    """Top-level configuration aggregating every subsystem."""

    system: System = field(default_factory=System)
    bs: BS = field(default_factory=BS)
    ris: RIS = field(default_factory=RIS)
    users: Users = field(default_factory=Users)
    scenario: Scenario = field(default_factory=Scenario)
    channel: Channel = field(default_factory=Channel)
    state: State = field(default_factory=State)
    rl: RL = field(default_factory=RL)
    reward: Reward = field(default_factory=Reward)
    experiment: Experiment = field(default_factory=Experiment)
    benchmark: Benchmark = field(default_factory=Benchmark)
    generalization: dict = field(
        default_factory=lambda: {"enabled": False, "train_condition": {}, "test_condition": {}}
    )
    logging: dict = field(default_factory=lambda: {"level": "INFO", "file": None})

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ validate
    def validate(self) -> None:
        """Validate every field, raising :class:`ConfigError` on the first problem."""
        # transmit_power_dbm may be negative (low but still positive power);
        # only reject it when it is so low it is no longer a real power
        # (-150 dBm ~ 1e-18 W) or non-finite.
        p_dbm = self.system.transmit_power_dbm
        if (
            self.system.carrier_frequency_hz <= 0
            or self.system.bandwidth_hz <= 0
            or not math.isfinite(p_dbm)
            or p_dbm < -150.0
        ):
            raise ConfigError("frequency and bandwidth must be positive; transmit power must be a finite value >= -150 dBm")
        if not 1 <= self.ris.phase_bits <= 6:
            raise ConfigError("phase_bits must be in 1..6")
        if self.ris.rows < 1 or self.ris.columns < 1 or self.users.count < 1:
            raise ConfigError("rows, columns and users.count must be >= 1")
        if not 0 <= self.channel.csi_error < 1:
            raise ConfigError("csi_error must be in [0,1)")
        if self.users.mobility not in ("static", "linear", "random_waypoint"):
            raise ConfigError("invalid mobility")
        if self.channel.model not in ("3gpp_um", "free_space", "log_distance"):
            raise ConfigError("invalid channel model")
        if self.channel.fading not in ("rician", "rayleigh", "no_fading"):
            raise ConfigError("invalid fading model")
        if self.system.modulation not in ("qpsk", "16qam", "64qam"):
            raise ConfigError("invalid modulation")
        if self.channel.direct_link not in ("los", "blocked"):
            raise ConfigError("direct_link must be 'los' or 'blocked'")
        if self.channel.blockage_db < 0:
            raise ConfigError("blockage_db must be >= 0")
        velocity = self.users.velocity_mps
        if isinstance(velocity, list):
            if min(velocity) < 0:
                raise ConfigError("velocity must be nonnegative")
        elif velocity < 0:
            raise ConfigError("velocity must be nonnegative")
        if self.rl.n_steps_per_episode < 5:
            raise ConfigError("n_steps_per_episode must be >= 5")

    # -------------------------------------------------------------------- derived
    @property
    def noise_power_w(self) -> float:
        """Thermal noise power (W) over the bandwidth plus the noise figure."""
        return K_BOLZMANN * T0 * self.system.bandwidth_hz * 10.0 ** (self.system.noise_figure_db / 10.0)

    @property
    def snr_db(self) -> float:
        """Derived link SNR (dB): transmit power minus noise power."""
        return self.system.transmit_power_dbm - (10.0 * math.log10(self.noise_power_w) + 30.0)

    # ------------------------------------------------------------------- (de)serial
    def to_dict(self) -> dict:
        """Convert to a plain (YAML-serializable) dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Build a Config from a dict, warning on unknown keys."""
        d = deepcopy(d)
        mapping = {
            "system": System, "bs": BS, "ris": RIS, "users": Users, "scenario": Scenario,
            "channel": Channel, "state": State, "rl": RL, "reward": Reward,
            "experiment": Experiment, "benchmark": Benchmark,
        }
        kw = {}
        for key, value in d.items():
            if key in mapping:
                valid = mapping[key].__dataclass_fields__
                extra = set(value) - set(valid)
                if extra:
                    logger.warning("Unknown config keys in %s: %s", key, extra)
                kw[key] = mapping[key](**{a: b for a, b in value.items() if a in valid})
            elif key in ("generalization", "logging"):
                kw[key] = value
            else:
                logger.warning("Unknown config key: %s", key)
        return cls(**kw)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load and validate a config from a YAML file."""
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_yaml(self, path: str) -> None:
        """Write this config to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    # --------------------------------------------------------------------- overrides
    def clone_with_overrides(self, overrides: dict) -> "Config":
        """Return a deep copy with dotted-key overrides applied.

        Example: ``clone_with_overrides({"channel.csi_error": 0.1, "ris.rows": 8})``.
        """
        d = self.to_dict()
        for key, value in overrides.items():
            node = d
            bits = key.split(".")
            for b in bits[:-1]:
                node = node[b]
            node[bits[-1]] = value
        return Config.from_dict(d)