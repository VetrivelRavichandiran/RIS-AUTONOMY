# RIS-AUTONOMY — Architecture

RIS-AUTONOMY is a deep-reinforcement-learning platform for adaptive
**Reconfigurable Intelligent Surface (RIS)** phase optimization in a
millimeter-wave BS → RIS → UE MISO system.  A PPO agent chooses the RIS phase
vector each coherence interval to maximize a communication reward (sum rate,
SINR, energy efficiency, minus a switching cost).  The platform is a clean
layered engine: configuration → channel model → RIS model → physical-layer
metrics → Gymnasium environment → RL training → evaluation/benchmarking, with a
CLI, scripts, and a Streamlit dashboard on top.

This document describes the modules, the data flow, and the exact tensor
dimensions.  See `configuration.md` for the full config reference and
`user_guide.md` for step-by-step usage.

---

## 1. Layered overview

```
                 ┌───────────────────────────────────────────────┐
   UI layer      │  app/dashboard.py        (Streamlit)          │
                 │  scripts/*.py            (thin CLI wrappers)  │
                 │  src/ris_autonomy/cli.py (argparse subcommands)│
                 └───────────────┬───────────────────────────────┘
                                 │
   Orchestration ┌───────────────▼───────────────────────────────┐
                 │  rl/training.py   (train, save, plot)          │
                 │  evaluation/{evaluator,benchmark,experiments}  │
                 └───────────────┬───────────────────────────────┘
                                 │
   RL layer      ┌───────────────▼───────────────────────────────┐
                 │  rl/{agent,policies,rewards,callbacks}         │
                 │  baselines/  (no_ris, random, greedy, conv.)   │
                 └───────────────┬───────────────────────────────┘
                                 │
   Environment ┌─────────────────▼───────────────────────────────┐
                 │  environment/wireless_env.py  (Gymnasium Env)  │
                 │  environment/{state,mobility}                  │
                 └───────────────┬───────────────────────────────┘
                                 │
   Physics     ┌─────────────────▼───────────────────────────────┐
                 │  channels/   (pathloss, fading, generator)     │
                 │  ris/        (quantization, surface, ctrl)     │
                 │  communications/ (signal, sinr, rate, ber, ee) │
                 └───────────────┬───────────────────────────────┘
                                 │
   Foundation  ┌─────────────────▼───────────────────────────────┐
                 │  config.py (validated dataclasses + YAML)       │
                 │  utils/{seeding,logging,io}                    │
                 └───────────────────────────────────────────────┘
```

Data flows **downward** for physics (config → channels → RIS → metrics) and
**upward** for control (environment → RL → orchestration → UI).

---

## 2. Tensor dimensions (single source of truth)

| Symbol | Meaning | Default |
|--------|---------|---------|
| `K`    | users (UEs) | 2 |
| `M`    | BS antennas | 4 |
| `N`    | RIS elements = `rows * columns` | 64 |
| `B`    | RIS phase bits → `2**B` levels | 2 |

Channel matrices (all complex128):

| Tensor | Shape | Link |
|--------|-------|------|
| `h_direct` | `(K, M)` | BS → UE (direct) |
| `H_br`     | `(N, M)` | BS → RIS |
| `h_ru`     | `(K, N)` | RIS → UE |
| `theta`    | `(N,)` real | RIS phase vector (rad) |

The **effective channel** is

```
H_eff = h_direct + h_ru @ diag(e^{j theta}) @ H_br
      = h_direct + (h_ru * e^{j theta}[None, :]) @ H_br      # vectorized
```

with shape `(K, M)`.  The transmitter precodes on the *observed* (CSI-errored)
`H_eff_tilde` and the SINR is evaluated on the *true* `H_eff`.

---

## 3. Module reference

### 3.1 `config.py`
A single validated `Config` dataclass aggregating nested dataclasses
(`System`, `BS`, `RIS`, `Users`, `Scenario`, `Channel`, `State`, `RL`,
`Reward`, `Experiment`, `Benchmark`) plus `generalization` and `logging`
dicts.  `from_yaml` / `to_yaml` / `from_dict` / `to_dict` round-trip; unknown
keys log a warning.  `validate()` raises `ConfigError` with a precise message.
`clone_with_overrides({"dotted.key": value})` deep-copies and applies nested
overrides (used by the experiment engine).  SNR is **derived**:
`snr_db = transmit_power_dbm - noise_power_dbm` where the noise is thermal
noise over the bandwidth plus the noise figure (`noise_power_w` property).

### 3.2 `channels/`
- `pathloss.py` — `free_space_db`, `log_distance_db`, `three_gpp_um_db`, and
  the `pathloss_db` dispatcher.  All return dB and reject non-positive
  distance/frequency.
- `fading.py` — `rician_fading` (random LOS phase per tap), `rayleigh_fading`
  (i.i.d. `CN(0,1)`), `no_fading`, and `fading_coefficients`.  Complex
  Gaussian is `(N(0,1)+jN(0,1))/sqrt(2)` so `E[|h|^2]=1`.
- `channel_models.py` — `ChannelRealization` (validates shapes),
  `compute_effective_channel`, and `apply_csi_error`
  (`H_tilde = sqrt(1-eps) H + sqrt(eps) sigma W`).
- `channel_generator.py` — `ChannelGenerator(config, rng).generate(positions)`
  builds the three links from 2-D geometry (3-D distance with BS height),
  applies distance-based path loss and the configured fading, and returns the
  true channels.  `direct_link` is `"los"` (distance threshold) or `"blocked"`
  (NLOS + `blockage_db`, the canonical mmWave RIS setting).

### 3.3 `ris/`
- `quantization.py` — `PhaseQuantizer(B)`: `levels=2**B`, `step=2pi/levels`,
  `quantize` (snap to nearest level, result in `[0,2pi)`), `from_action`
  (maps `[-1,1]` → `(a+1)pi` → quantize).  Vectorized, deterministic.
- `ris_surface.py` — `RISurface`: holds current/previous `theta`,
  `set_theta` (re-quantizes), `phase_matrix` (`diag(a e^{j theta})`),
  `switch_fraction` (fraction of elements that changed), `reset`.
- `phase_controller.py` — `PhaseController`: `action_to_phases` (continuous)
  and `discrete_phases` (per-element index mode).

### 3.4 `communications/`
- `signal_model.py` — `mrt_precoding` (per-UE unit-norm beamformer) and
  `received_signal` (per-user SINR = signal / (interference + noise)).
- `sinr.py` — `sinr_db`, `snr_db` (floored log, always finite).
- `achievable_rate.py` — `rate_bps = B*log2(1+SINR)`, `sum_rate`,
  `spectral_efficiency_bps_hz`.
- `ber.py` — Gray-mapped QPSK/16-QAM/64-QAM BER approximations.
- `energy_efficiency.py` — `ris_power_w`, `total_power_w`,
  `energy_efficiency = rate / power`.

### 3.5 `environment/`
- `wireless_env.py` — `RISAUTONOMYEnv(gym.Env)`.  Observation space
  `Box(-3,3,(obs_dim,),float32)`; action space `Box(-1,1,(N,),float32)`
  (continuous) or `MultiDiscrete([2**B]*N)` (per-element discrete).
  `reset` returns `(obs, info)`; `step` returns the 5-tuple.  `render`
  (human → matplotlib Figure, rgb_array → ndarray) and `close` are provided.
  `get_topology()` returns positions + distances for visualization.
- `state.py` — `StateBuilder` assembles the normalized, clipped observation
  vector from a `StepData` dataclass; `observation_dim` is the single source of
  truth for its length.  Layout (in order): per user `|h_direct|²`,
  `∠h_direct`, `|H_eff|²`, `∠H_eff` (M each), `sinr`, `rate`, `position`,
  `velocity`; then global `theta`, `theta_prev`, `snr`, `csi_quality`,
  `interference`.  Each group is divided by its scale, concatenated, clipped to
  `[-3,3]`, cast to float32, and (optionally) noise is added.
- `mobility.py` — `MobilityModel` for `static`, `linear` (reflecting), and
  `random_waypoint`.

**Channel-coherence semantics.** Within a step the channel is fixed: the agent
observes the (CSI-errored) channel, acts, and the reward is measured on that
*same* channel with the new phases.  Only after the reward is computed does the
environment advance the UEs and regenerate the channel for the *next*
observation.  This guarantees every controller acts on the channel it observed.

### 3.6 `rl/`
- `rewards.py` — `RewardFunction`: `R = w_rate*(rate/rate_norm) +
  w_sinr*(mean(sinr_db)/sinr_norm_db) + w_ee*(ee/ee_norm) -
  w_switch*switch_fraction - w_constraint*violation`.  Returns `(reward,
  breakdown)` where the breakdown sums to the reward.
- `policies.py` — `build_policy_net` → SB3 `policy_kwargs` for `default`/
  `small`/`large` MLP sizes.
- `callbacks.py` — `EpisodeMetricsCallback` (per-episode CSV) and
  `BestModelCallback` (saves the model **only when strictly better** than the
  recorded rolling-window best, with a JSON sidecar).
- `agent.py` — `create_agent` (PPO, device auto-fallback to CPU),
  `save_agent` (zip + metadata sidecar), `load_agent` (with `ModelLoadError`).
- `training.py` — `train(config, seed, timesteps, output_dir, ...)`: seeds,
  builds env + agent, runs `model.learn` with the callbacks, saves the final
  model, `training_stats.json`, and real training-curve plots.

### 3.7 `baselines/`
Conventional (non-DRL) controllers sharing one interface
(`action(env) -> np.ndarray` in `[-1,1]^N`), evaluated by the same harness on
the same channel seed for fairness:
- `NoRIS` — disables the RIS so `H_eff = h_direct`.
- `RandomRIS` — random continuous action each step.
- `GreedyRIS` — coordinate descent over RIS elements maximizing **sum rate**
  (vectorized partial update, 2 passes).
- `ConventionalRIS` — **alternating per-user** coordinate descent maximizing
  each user's rate (2 passes over users).  Genuinely different from greedy.

### 3.8 `evaluation/`
- `metrics.py` — `fairness_jain`, `outage_probability`, `MetricsAccumulator`.
- `evaluator.py` — `evaluate_agent` runs a model (or baseline) for N episodes,
  accumulates per-step metrics + inference latency, writes CSV/JSON + plots.
- `benchmark.py` — `run_benchmark` compares no_ris / random / greedy /
  conventional / drl on the same scenario and seed; writes a comparison table
  (CSV/JSON) and real plots.  Runs with ≥2 users so the two optimizers differ.
- `experiments.py` — `run_sweep` + `_sweep_dispatch` for all 9 sweep types
  (snr, ris_size, phase_bits, user_count, velocity, csi_error, power,
  distance, fading) and `run_generalization` (train vs test condition gap).
- `comparison.py` — re-exports `run_benchmark`.

### 3.9 `visualization/`
Matplotlib for static figures (files, `render`), Plotly for interactive HTML.
Every function takes plain data (arrays/dicts/DataFrames), not env internals:
`topology` (2-D scene + interactive), `phase_map` (grid + histogram),
`heatmaps` (rate/SINR over a position grid), `training_plots` (curves +
convergence), `performance_plots` (all sweep + comparison + CDF + latency
plots), and `io.save_figure` (PNG always, optional HTML).

### 3.10 `utils/`
`seeding.set_global_seed` (Python/NumPy/PyTorch/hash), `logging.setup_logging`
(idempotent), `io` (run dirs, JSON/CSV/Parquet I/O, `run_manifest`,
`software_versions`).

---

## 4. Determinism

`set_global_seed` seeds every RNG the platform uses, so a run with a fixed
seed is byte-for-byte reproducible: two same-seed training runs produce
identical `episode_metrics.csv`.  The environment derives its per-episode RNG
from `seed + episode`, so evaluation is reproducible per episode.

## 5. Extending the platform

- **New channel/fading model** — add a function in `channels/`, register it in
  the dispatcher, and allow it in `config.validate`.
- **New baseline** — implement the `Baseline` interface in `baselines/` and add
  it to `_REGISTRY`; it is picked up by the benchmark and evaluator.
- **New sweep** — add a branch in `experiments._sweep_dispatch` and a plot in
  `performance_plots`; the CLI `experiment --name` dispatches to it.
- **New observation feature** — add a key to `state.components`, an `add(...)`
  call in `StateBuilder.build`, and the matching term in `observation_dim`.