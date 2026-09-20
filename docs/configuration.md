# RIS-AUTONOMY — Configuration Reference

Every tunable lives in a single validated `Config` (see `src/ris_autonomy/
config.py`).  A config is a YAML file (or dict) with the sections below.
`Config.from_yaml` validates on load and raises `ConfigError` with a precise
message on the first problem.  Unknown keys log a warning and are ignored.
`clone_with_overrides({"dotted.key": value})` deep-copies and applies nested
overrides — the mechanism the experiment engine uses to sweep parameters.

**SNR is derived, not stored.** `snr_db = transmit_power_dbm - noise_power_dbm`
where `noise_power_dbm = 10*log10(k_B * T0 * bandwidth_hz) + noise_figure_db`
(`k_B = 1.380649e-23 J`, `T0 = 290 K`).  The `config.noise_power_w` and
`config.snr_db` properties expose the derived values.

---

## `system` — RF system

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `carrier_frequency_hz` | float | `28e9` | Must be > 0. |
| `bandwidth_hz` | float | `100e6` | Must be > 0. |
| `transmit_power_dbm` | float | `30` | Must be ≥ 0. |
| `noise_figure_db` | float | `5` | Added to thermal noise. |
| `modulation` | str | `qpsk` | `qpsk` \| `16qam` \| `64qam`. |
| `ris_element_power_w` | float | `0.005` | Power per RIS element. |
| `bs_circuit_power_w` | float | `0.0` | BS circuit power. |

## `bs` — base station

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `antennas` | int | `4` | `M`, number of BS antennas. |

## `ris` — reconfigurable surface

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `rows` | int | `8` | ≥ 1. |
| `columns` | int | `8` | ≥ 1. |
| `phase_bits` | int | `2` | `1..6`; `2**bits` levels. |
| `amplitude` | float | `1.0` | Reflection amplitude. |
| `switching_cost` | float | `0.1` | (reserved scale). |

## `users` — user equipment

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `count` | int | `2` | `K`, ≥ 1. |
| `mobility` | str | `static` | `static` \| `linear` \| `random_waypoint`. |
| `velocity_mps` | float or list | `2.0` | ≥ 0; list = one per user. |

## `scenario` — 2-D geometry (meters)

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `boundary_width_m` | float | `100` | X extent. |
| `boundary_height_m` | float | `100` | Y extent. |
| `bs_position` | list | `[0, 0]` | BS (x, y). |
| `ris_position` | list | `[50, 50]` | RIS (x, y). |
| `ue_region` | list | `[10, 10, 90, 90]` | `[x1, y1, x2, y2]`. |
| `bs_height_m` | float | `25` | 3-D distance uses this. |
| `step_dt_s` | float | `1.0` | Mobility time step. |

## `channel` — channel model

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `model` | str | `3gpp_um` | `3gpp_um` \| `free_space` \| `log_distance`. |
| `fading` | str | `rician` | `rician` \| `rayleigh` \| `no_fading`. |
| `rician_k_db` | float | `10.0` | Rician K-factor. |
| `pathloss_exponent` | float | `2.5` | Used by `log_distance`. |
| `csi_error` | float | `0.0` | Fraction in `[0, 1)`. |
| `los_distance_m` | float | `60.0` | LOS/NLOS threshold (`los`). |
| `direct_link` | str | `blocked` | `los` \| `blocked`. |
| `blockage_db` | float | `40.0` | Extra loss when `blocked`. |

`direct_link: blocked` is the canonical mmWave RIS setting: the direct path is
obstructed (NLOS + `blockage_db`) while the two RIS hops stay LOS, so the
reflected path is the useful one and RIS phase control matters.

## `state` — observation composition

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `components` | dict | all `true` | Toggle each feature group. |
| `observation_noise_std` | float | `0.0` | Added after normalization. |

`components` keys: `direct_magnitude`, `direct_phase`, `effective_magnitude`,
`effective_phase`, `sinr`, `rate`, `position`, `velocity`, `current_phase`,
`previous_phase`, `snr`, `csi_quality`, `interference`.  The observation length
is computed by `observation_dim(config)` from these toggles.

## `rl` — reinforcement learning

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `algorithm` | str | `ppo` | Only `ppo` is wired. |
| `learning_rate` | float | `3e-4` | |
| `gamma` | float | `0.99` | |
| `gae_lambda` | float | `0.95` | |
| `n_steps` | int | `2048` | Rollout length. |
| `batch_size` | int | `64` | |
| `n_epochs` | int | `10` | |
| `ent_coef` | float | `0.01` | Entropy coefficient. |
| `clip_range` | float | `0.2` | PPO clip. |
| `timesteps` | int | `100000` | Default training budget. |
| `n_steps_per_episode` | int | `100` | ≥ 5; episode length. |
| `action_mode` | str | `continuous` | `continuous` \| `per_element_discrete`. |
| `device` | str | `auto` | `auto` \| `cpu` \| `cuda`. |
| `net_arch` | str | `default` | `default` \| `small` \| `large`. |
| `checkpoint_every` | int | `10000` | Timesteps; 0 disables. |

## `reward` — multi-objective weights

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `rate_weight` | float | `1.0` | |
| `sinr_weight` | float | `0.5` | |
| `energy_weight` | float | `0.2` | |
| `switching_weight` | float | `0.1` | |
| `constraint_weight` | float | `0.0` | Reserved (violation = 0). |
| `rate_norm` | float | `1e4` | Normalization scale. |
| `sinr_norm_db` | float | `20.0` | |
| `ee_norm` | float | `1e4` | |

Reward: `R = w_rate*(rate/rate_norm) + w_sinr*(mean(sinr_db)/sinr_norm_db) +
w_ee*(ee/ee_norm) - w_switch*switch_fraction - w_constraint*violation`.

## `experiment` — sweeps and evaluation

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `seed` | int | `42` | Global seed. |
| `episodes` | int | `10` | Episodes per evaluation. |
| `snr_sweep_db` | list | `[0,5,10,15,20,25,30]` | |
| `ris_size_sweep` | list | `[16,32,64,128]` | Perfect squares. |
| `phase_bits_sweep` | list | `[1,2,3,4]` | |
| `user_count_sweep` | list | `[1,2,3,4]` | |
| `velocity_sweep_mps` | list | `[0,1,5,10,20]` | |
| `csi_error_sweep` | list | `[0,0.05,0.1,0.2,0.3]` | |
| `power_sweep_dbm` | list | `[10,20,30,40]` | |
| `distance_sweep_m` | list | `[20,40,60,80]` | |
| `fading_sweep` | list | `[rician,rayleigh]` | |
| `outage_target_bps` | float | `1e6` | Outage threshold. |

## `benchmark` — baseline comparison

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `drl_model_path` | str or null | `null` | If null, train a quick model. |
| `train_drl_if_missing` | bool | `true` | |
| `drl_timesteps_if_missing` | int | `5000` | |
| `episodes` | int | `5` | Episodes per controller. |

The benchmark runs with **at least 2 users** so that greedy (sum-rate) and
conventional (per-user) optimization produce genuinely different results.

## `generalization` — out-of-distribution test

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `enabled` | bool | `false` | |
| `train_condition` | dict | `{}` | Dotted-key overrides. |
| `test_condition` | dict | `{}` | Dotted-key overrides. |

## `logging`

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `level` | str | `INFO` | |
| `file` | str or null | `null` | Optional log file. |

---

## Shipped configs

| File | Scenario |
|------|----------|
| `default.yaml` | 28 GHz, 100 MHz, 8×8 RIS, 2-bit, 2 users, static, 100k steps. |
| `quick_test.yaml` | 4×4 RIS, 1 user, 100 MHz, 50-step episodes, 5k steps — fast E2E. |
| `single_user.yaml` | Default but 1 user. |
| `multi_user.yaml` | Default but 4 users, velocity 5. |
| `stress_test.yaml` | 12×12 RIS, 4 users, random_waypoint v=10, csi_error 0.1, 200k steps. |

All five pass `Config.from_yaml` validation.