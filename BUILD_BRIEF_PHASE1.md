# BUILD BRIEF — PHASE 1: CORE SIMULATION ENGINE (RIS-AUTONOMY)

You are building the complete core engine of **RIS-AUTONOMY**: a Deep RL platform for adaptive
Reconfigurable Intelligent Surface (RIS) optimization in 6G wireless networks.
Project root: `/opt/sandbox/workspace/RIS-AUTONOMY` (directory skeleton already exists).

## 0. Environment facts (do not re-verify endlessly)
- Python 3.11.16, pip 24.0. CPU only (no CUDA).
- Installed: torch (CPU), gymnasium, stable-baselines3, streamlit, plotly, pyyaml, pytest,
  matplotlib, pyarrow, scipy, pandas, numpy.
- No internet needed. Do NOT install anything else unless a hard import failure proves it missing.
- This is a REAL engineering build. No placeholders, no TODOs, no fake results, no mock ML.
  Every number the system reports must come from actual computation.

## 1. Repository structure (create exactly this; you may add small extra files if justified)

```
RIS-AUTONOMY/
├── README.md                  (short for now; Phase 2 rewrites it — but make it accurate)
├── LICENSE                    (MIT)
├── requirements.txt
├── pyproject.toml             (metadata + pytest + ruff config)
├── .gitignore
├── .env.example               (optional env vars, documented)
├── configs/
│   ├── default.yaml
│   ├── quick_test.yaml
│   ├── single_user.yaml
│   ├── multi_user.yaml
│   └── stress_test.yaml
├── data/{raw,processed,results}/        (keep; add .gitkeep)
├── models/{checkpoints,final}/          (keep; add .gitkeep)
├── src/ris_autonomy/
│   ├── __init__.py            (version, package docstring)
│   ├── config.py
│   ├── cli.py                 (argparse CLI: train/evaluate/benchmark/experiment/dashboard/generate-channels/reproduce)
│   ├── environment/{__init__,wireless_env,mobility,state}.py
│   ├── channels/{__init__,channel_models,fading,pathloss,channel_generator}.py
│   ├── ris/{__init__,ris_surface,phase_controller,quantization}.py
│   ├── communications/{__init__,signal_model,sinr,achievable_rate,ber,energy_efficiency}.py
│   ├── rl/{__init__,agent,policies,rewards,callbacks,training}.py
│   ├── baselines/{__init__,random_ris,no_ris,greedy,optimization}.py
│   ├── evaluation/{__init__,evaluator,metrics,experiments,comparison}.py
│   ├── visualization/{__init__,topology,phase_map,heatmaps,training_plots,performance_plots}.py
│   └── utils/{__init__,seeding,logging,io}.py
├── app/dashboard.py           (Phase 2 — create a minimal placeholder-free stub that prints
│                              "dashboard arrives in phase 2" is FORBIDDEN; instead make it a
│                              working minimal Streamlit page: title + config display + a
│                              "run quick training" button that calls the training pipeline.
│                              Phase 2 will replace it with the full dashboard.)
├── scripts/{train,evaluate,benchmark,generate_channels,reproduce,experiment}.py
├── tests/{test_channels,test_ris,test_metrics,test_environment,test_reward,test_baselines,test_smoke}.py
└── docs/{architecture.md,configuration.md,user_guide.md}   (write solid first drafts; Phase 2 polishes)
```

## 2. Configuration system (`src/ris_autonomy/config.py`)

Dataclass-based, fully validated, YAML round-trip. Top-level `Config` with nested dataclasses:

```
system:    carrier_frequency_hz (default 28e9), bandwidth_hz (100e6),
           transmit_power_dbm (30), noise_figure_db (5),
           modulation ("qpsk" | "16qam" | "64qam"),
           ris_element_power_w (0.005), bs_circuit_power_w (0.0)
bs:        antennas (4)
ris:       rows (8), columns (8), phase_bits (2), amplitude (1.0),
           switching_cost (0.1)
users:     count (2), mobility ("static"|"linear"|"random_waypoint"),
           velocity_mps (2.0)   # float or list of floats (one per user)
scenario:  boundary_width_m (100), boundary_height_m (100),
           bs_position ([0.0, 0.0]), ris_position ([50.0, 50.0]),
           ue_region ([10.0, 10.0, 90.0, 90.0])
channel:   model ("3gpp_um"|"free_space"|"log_distance"),
           fading ("rician"|"rayleigh"|"no_fading"), rician_k_db (10.0),
           pathloss_exponent (2.5, used by log_distance),
           csi_error (0.0)      # fraction in [0,1)
state:     components (dict of bools: direct_magnitude, direct_phase,
           effective_magnitude, effective_phase, sinr, rate, position, velocity,
           current_phase, previous_phase, snr, csi_quality, interference — all true by default),
           observation_noise_std (0.0)
rl:        algorithm ("ppo"), learning_rate (3e-4), gamma (0.99), gae_lambda (0.95),
           n_steps (2048), batch_size (64), n_epochs (10), ent_coef (0.01),
           clip_range (0.2), timesteps (100000), n_steps_per_episode (100),
           action_mode ("continuous"|"per_element_discrete"),
           device ("auto"|"cpu"|"cuda"),
           net_arch ("default"|"small"|"large"),
           checkpoint_every (10000)   # timesteps; 0 disables intermediate checkpoints
reward:    rate_weight (1.0), sinr_weight (0.5), energy_weight (0.2),
           switching_weight (0.1), constraint_weight (0.0),
           rate_norm (1e7), sinr_norm_db (20.0), ee_norm (1e6)
experiment: seed (42), episodes (10),
           snr_sweep_db ([0,5,10,15,20,25,30]), ris_size_sweep ([16,32,64,128]),
           phase_bits_sweep ([1,2,3,4]), user_count_sweep ([1,2,3,4]),
           velocity_sweep_mps ([0,1,5,10,20]), csi_error_sweep ([0,0.05,0.1,0.2,0.3]),
           power_sweep_dbm ([10,20,30,40]), distance_sweep_m ([20,40,60,80]),
           fading_sweep (["rician","rayleigh"]),
           outage_target_bps (1e6)
benchmark: drl_model_path (null), train_drl_if_missing (true),
           drl_timesteps_if_missing (5000), episodes (5)
generalization: enabled (false), train_condition ({}), test_condition ({})
logging:    level ("INFO"), file (null)
```

Rules:
- `Config.from_yaml(path)`, `Config.to_yaml(path)`, `Config.from_dict`, `to_dict`.
- Validation raises `ConfigError` (custom exception) with a precise message:
  positive freq/bandwidth/power, phase_bits in 1..6, rows/cols >= 1, users.count >= 1,
  csi_error in [0,1), mobility in allowed set, model/fading/modulation in allowed sets,
  velocity >= 0, n_steps_per_episode >= 5, etc.
- Unknown YAML keys → log a WARNING (do not crash).
- `Config.clone_with_overrides(dict)` → deep-copied config with nested overrides applied
  (used by experiments: e.g. override `system` SNR via `transmit_power_dbm`/noise, or
  `users.velocity_mps`, `channel.csi_error`, `ris.rows/columns`).
- SNR is DERIVED, not stored: `snr_db = transmit_power_dbm - noise_power_dbm` where
  `noise_power_dbm = 10*log10(k_B * T0 * bandwidth_hz) + noise_figure_db` (k_B=1.380649e-23,
  T0=290K). Provide `config.noise_power_w` and `config.snr_db` properties.
- NO magic constants scattered in code — everything above lives in Config.

## 3. Channels (`src/ris_autonomy/channels/`)

### pathloss.py
- `free_space_db(d_m, f_hz) -> float`: 20*log10(4*pi*d/lambda).
- `log_distance_db(d_m, f_hz, n, d_ref=1.0, pl_ref=None)`: FSPL(d_ref) + 10*n*log10(d/d_ref).
- `three_gpp_um_db(d_m, f_hz, los: bool)`: 3GPP TR 38.901 UMa:
  LOS: 32.45 + 21.7*log10(d) + 20*log10(f_GHz);
  NLOS: 32.45 + 21.7*log10(d) + 20*log10(f_GHz) + 16.7*log10(d).
  (d in meters, f in GHz.)
- `pathloss_db(d_m, f_hz, model: str, los: bool, exponent: float) -> float` dispatcher.
- All raise `ValueError` for d <= 0 or f <= 0.

### fading.py
- `rician_fading(rng, shape, k_db) -> np.ndarray` (complex128): h = sqrt(K/(K+1))*exp(j*phi)
  + CN(0, 1/(K+1)), phi ~ U[0,2pi) per tap (or fixed 0 — document your choice; use random).
- `rayleigh_fading(rng, shape)`: CN(0,1) i.i.d.
- `no_fading(rng, shape)`: ones.
- `fading_coefficients(rng, shape, model, k_db)` dispatcher.
- Complex Gaussian: `(rng.standard_normal(n) + 1j*rng.standard_normal(n))/sqrt(2)`.

### channel_models.py
- `ChannelRealization` dataclass: `h_direct (K,M)`, `H_br (N,M)`, `h_ru (K,N)`, all complex128,
  plus `positions` (K,2) float64 and `metadata` dict. Validate shapes on construction.
- `compute_effective_channel(h_direct, H_br, h_ru, theta) -> (K,M)`:
  `H_eff = h_direct + h_ru @ diag(exp(1j*theta)) @ H_br` (vectorized:
  `h_direct + (h_ru * exp(1j*theta)[None, :]) @ H_br`).
- `apply_csi_error(H, rng, csi_error)`: per matrix, `sigma = ||H||_F/sqrt(numel)`;
  `H_tilde = sqrt(1-eps)*H + sqrt(eps)*sigma*W`, W ~ CN(0,1) same shape. eps in [0,1).
- Dimension validation with clear error messages everywhere (K users, M BS antennas, N RIS elements).

### channel_generator.py
- `ChannelGenerator(config, rng)` — generates a full `ChannelRealization` for current UE
  positions:
  - distances: BS→UE (K), BS→RIS (1), RIS→UE (K) from 2D positions (3D height optional:
    BS height 25 m, RIS/UE height 0 — configurable via scenario? keep BS_HEIGHT_M=25.0 as a
    named constant in config `scenario.bs_height_m` default 25.0; 3D distance =
    sqrt(dx²+dy²+dz²)).
  - LOS probability model: simple — direct link is LOS if distance < 60 m else NLOS
    (documented heuristic; configurable `channel.los_distance_m` default 60).
  - amplitudes: `A = 10^(-PL/20)`; h = A * fading.
  - Applies nothing else. Returns true channels; caller applies CSI error.
- `generate_batch(config, rng, positions, num_realizations) -> list[ChannelRealization]`
  for batch generation (used by generate_channels.py).

## 4. RIS (`src/ris_autonomy/ris/`)

### quantization.py
- `PhaseQuantizer(bits: int)` with `levels` property (=2**bits), `step` (=2pi/levels),
  `quantize(theta: np.ndarray) -> np.ndarray` (vectorized:
  `2*pi*floor(theta/step + 0.5)/step % (2*pi)`), `levels_array` (precomputed np array of
  level phases), `from_action(a: np.ndarray) -> np.ndarray` where a in [-1,1] →
  `theta = (a+1)*pi` → quantize. Validate bits 1..6.
- Tests must verify: quantize output always in [0, 2pi), exactly 2^B distinct values,
  round-trip determinism.

### ris_surface.py
- `RISurface(config)` — holds rows, columns, N, amplitude, phase_bits, current theta (N,),
  previous theta (N,). Methods: `set_theta(theta)`, `phase_matrix() -> np.ndarray (N,N)`
  (diag(exp(1j*theta))*amplitude, built with `np.diag`), `switch_fraction() -> float`
  (fraction of elements whose quantized phase changed vs previous), `reset(rng)`.

### phase_controller.py
- `PhaseController(quantizer)` — `action_to_phases(action: np.ndarray) -> np.ndarray`
  (the deterministic [-1,1]→[0,2pi)→quantized mapping), `discrete_phases(indices)` for the
  per-element discrete action mode.

## 5. Communications (`src/ris_autonomy/communications/`)

All functions are pure NumPy, vectorized, with shape validation and `check_finite`
(raise `NumericalError` custom exception on NaN/inf).

### signal_model.py
- `mrt_precoding(H_eff_tilde: (K,M)) -> W (M,K)`: W[:,k] = conj(H_eff_tilde[k]) /
  ||H_eff_tilde[k]|| (per-UE unit norm → per-user power P_tx/K). If a row norm is ~0,
  use a random unit vector (seeded) and log a warning.
- `received_signal(H_eff: (K,M), W: (M,K), p_tx_w, rng) -> (y (K,), sinr (K,))`:
  SINR_k = |h_k·W[:,k]|²·(P/K) / (Σ_{j≠k} |h_k·W[:,j]|²·(P/K) + sigma2), sigma2 = noise power W.
  (y itself is optional; return SINR and the signal/interference/noise powers as a dict.)
- Note: precoding uses the OBSERVED (CSI-errored) channel; SINR uses the TRUE channel.

### sinr.py
- `sinr_db(sinr_lin)`, `snr_db(noise_power_w, p_tx_w, K)` helpers, `check_finite` guards.

### achievable_rate.py
- `rate_bps(sinr_lin, bandwidth_hz) -> (K,)`: B*log2(1+SINR).
- `sum_rate`, `spectral_efficiency_bps_hz`.

### ber.py
- `ber_approx(sinr_lin, modulation)`:
  qpsk: 0.5*erfc(sqrt(SINR));
  16qam: (8/15)*erfc(sqrt(3*SINR/13))*(1 - (2/5)*sqrt(3*SINR/13)/(sqrt(pi)*(1+SINR)));
  64qam: (15/64)*erfc(sqrt(2*SINR/7))*(1 - (2/7)*sqrt(2*SINR/7)/(sqrt(pi)*(1+SINR))).
  (Standard Gray-mapped QAM approximations — document them in the docstring.)

### energy_efficiency.py
- `ris_power_w(N, p_elem)`, `total_power_w(p_tx_w, N, p_elem, p_circuit)`,
  `energy_efficiency(sum_rate_bps, total_power_w) -> bps/W`.

## 6. Environment (`src/ris_autonomy/environment/`)

### mobility.py
- `MobilityModel` class: `static`, `linear` (constant velocity vector, reflect at
  boundaries), `random_waypoint` (pick random target inside boundary, move at speed, new
  target on arrival). `update(positions (K,2), velocities (K,2), dt=1.0) -> (K,2)`;
  dt configurable via `scenario.step_dt_s` default 1.0. Positions clamped to boundary.
- `random_initial_positions(rng, K, region)`.

### state.py
- `StateBuilder(config)` — builds the observation vector from a `StepData` dataclass
  (h_direct_tilde, H_eff_tilde, theta, theta_prev, positions, velocities, sinr_lin, rates,
  snr_db, csi_error, interference_power (K,)).
- Feature groups (each enabled by `state.components`):
  per user k: |h_direct[k]|² (M), ∠h_direct[k] (M), |H_eff[k]|² (M), ∠H_eff[k] (M),
  sinr_k (1), rate_k (1), position_k (2, /boundary), velocity_k (2, /30 m/s);
  global: theta (N, /pi), theta_prev (N, /pi), snr_db (1, /20), csi_error (1),
  interference_power_k (K, / (p_tx_w/K)).
- Normalization: divide each group by its scale, then `np.clip(obs, -3, 3)`, float32.
- `observation_dim(config) -> int` computed from the enabled components (single source of
  truth). Document the exact vector layout in the module docstring.
- `observation_noise_std` adds N(0, std²) AFTER normalization (only if > 0).

### wireless_env.py
- `RISAUTONOMYEnv(gymnasium.Env)` — constructor `(config, seed=None)`.
  - observation_space: Box(-3, 3, (obs_dim,), float32)
  - action_space: Box(-1, 1, (N,), float32) for continuous mode;
    MultiDiscrete([2**bits]*N) for per_element_discrete mode.
  - `reset(seed=None)`: seed internal rng (np.random.default_rng(seed) if given else
    derived from env seed + episode counter), place UEs, initialize theta (random quantized),
    generate channels, build obs. Returns (obs, info) with info containing
    `{"episode": int, "snr_db": float, "n_ris_elements": int, ...}`.
  - `step(action)`:
    1. map action → quantized theta (deterministic); update RIS (previous = old)
    2. mobility update (positions)
    3. regenerate channels (true) for new positions; apply CSI error → observed channels
    4. W = MRT(observed H_eff); metrics with TRUE H_eff: SINR (lin+dB), rates, sum rate,
       EE, BER, outage (rate_k < outage_target), fairness (Jain), switching fraction
    5. reward (see rewards.py)
    6. build observation from OBSERVED channels + metrics
    7. done when t >= n_steps_per_episode (truncated=True, terminated=False)
    Returns (obs, reward, terminated, truncated, info). info includes ALL metrics
    (rate_sum, rate_per_user, sinr_db, ee, ber, outage, fairness, switching_cost,
    p_tx_w, p_ris_w) — the RL callbacks and evaluator consume this.
  - `render()`: return a matplotlib figure (topology) or a dict of data; keep it cheap —
    `render(mode="human")` shows a plot; `render(mode="rgb_array")` returns a small array.
  - `close()`: clean up.
  - Numerical sanity: after channel gen, check finite; raise `NumericalError` otherwise.
  - Performance target: > 200 steps/s on CPU for default config (N=64, K=2, M=4).
    Vectorize; avoid per-element Python loops.
  - `env.get_topology() -> dict` for visualization (positions, distances, link powers).
- Gymnasium API correctness: `reset` returns (obs, info); `step` returns 5-tuple;
  spaces defined in `__init__`; `render_mode` parameter supported.

## 7. RL (`src/ris_autonomy/rl/`)

### rewards.py
- `RewardFunction(config)` — computes
  `R = w_rate*(R_sum/rate_norm) + w_sinr*(mean(sinr_db)/sinr_norm_db) + w_ee*(EE/ee_norm)
       - w_switch*(p_tx_w/K)/switching_ref? NO — use the raw switching FRACTION (0..1) *
       ris.switching_cost_scale` — keep it simple: `R = w_rate*rate_tilde + w_sinr*sinr_tilde
       + w_ee*ee_tilde - w_switch*switch_fraction - w_constraint*violation` where
       violation = 0.0 currently (reserved, documented). All tilde terms are the
       normalized quantities above. Return (reward, breakdown dict).

### policies.py
- `build_policy_net(config) -> dict`: "default" → dict(pi=[256,128], vf=[256,128]);
  "small" → dict(pi=[128,64], vf=[128,64]); "large" → dict(pi=[512,256], vf=[512,256]).

### callbacks.py
- `EpisodeMetricsCallback`: after each episode, append a row (episode, mean_reward,
  rate_sum, sinr_db, ee, switching, episode_length) to `metrics/episode_metrics.csv`
  (path from constructor).
- `BestModelCallback`: tracks best mean reward over a rolling window (last 10 episodes);
  saves model to `models/final/best_model.zip` ONLY when strictly better than the recorded
  best (never overwrite with equal/worse). Persists best value in a small JSON sidecar.
- `CheckpointCallback` (SB3 built-in) wired to `models/checkpoints/`.

### agent.py
- `create_agent(config, env, device=None) -> sb3.PPO`:
  - device: "auto" → cuda if torch.cuda.is_available() else cpu (log WARNING on fallback);
    "cpu"/"cuda" explicit (raise clear error if cuda requested but unavailable).
  - MlpPolicy with net_arch from policies.py, all hyperparams from config.
  - `load_agent(path, config, device)` — load with validation; raise `ModelLoadError`
    (custom) with clear message on missing/corrupt file; return (model, metadata dict)
    reading the sidecar JSON if present.
- `save_agent(model, path, metadata: dict)` — save zip + sidecar `model_metadata.json`
  (algorithm, seed, timesteps, config (full dict), timestamp, torch/sb3/gym versions).

### training.py
- `train(config, seed, timesteps, output_dir, device=None, model_path=None,
         resume=False) -> dict`:
  1. set_global_seed(seed); log config summary (env init, RIS elements, users, algorithm)
  2. build env, agent (resume loads existing model)
  3. callbacks: EpisodeMetricsCallback(output_dir/metrics), BestModelCallback(
     output_dir/models/final/best_model.zip), CheckpointCallback if configured
  4. `model.learn(timesteps, callback=...)`
  5. save final model → output_dir/models/final/final_model.zip (+ metadata)
  6. save training_stats.json (total timesteps, seed, duration, final eval quick metrics)
  7. generate plots (training_plots.py) → output_dir/plots/
  8. return summary dict.
- `quick_eval(model, config, episodes=3) -> dict` used for the final eval in step 6.

## 8. Baselines (`src/ris_autonomy/baselines/`)

Common interface: `class Baseline: name: str; def choose_phases(self, env_data, rng) -> theta (N,)`
where `env_data` is a dataclass with observed h_direct, H_br, h_ru, H_eff_tilde, positions.
All baselines are evaluated by the SAME harness (evaluation/evaluator.py) on the SAME
channel realizations (same seed) for fairness.

- `NoRIS`: theta unused; evaluator skips RIS path (H_eff = h_direct only).
- `RandomRIS`: theta ~ U[0,2pi) quantized, fixed per episode (regenerated at reset).
- `GreedyRIS`: per step, coordinate descent over elements: for element n try all 2^B
  levels, keep the one maximizing sum rate (computed with vectorized partial update:
  H_ris_base (with element n zeroed) + e^{jθ_l} * outer(h_ru[:,n], H_br[n,:])).
- `ConventionalOptimization`: alternating per-user optimization — for each user k,
  coordinate descent maximizing rate_k (interference from others treated as fixed),
  2 passes over users per step. Document as a conventional (non-DRL) optimization baseline.
- All must be fast enough: N=64, B=2 → 256 candidate evaluations per element pass;
  vectorize with NumPy (no per-level Python loops over N).

## 9. Evaluation (`src/ris_autonomy/evaluation/`)

### metrics.py
- `MetricsAccumulator`: collects per-step metrics from env info; produces
  `summary() -> dict` with mean/median/std/min/max of: rate_sum, rate_per_user (list),
  sinr_db, ee, ber, outage_prob, fairness, switching_cost; plus `inference_latency_ms`
  (measured wall-clock of the action call, median).
- `fairness_jain(rates)`, `outage_probability(rates, target)`.

### evaluator.py
- `evaluate_agent(model, config, episodes, deterministic=True, seed=None,
                  output_dir=None, baseline: Baseline|None=None) -> dict`:
  - If baseline given: run the baseline policy instead of the model (same env, same seed).
  - For each episode: reset(seed=seed+ep), loop steps, accumulate metrics, measure
    inference latency (time the model.predict / baseline.choose_phases call).
  - Returns full dict: per-episode list + aggregates.
  - `save_results(results, output_dir)`: metrics.json, metrics.csv (per-episode rows),
    summary.json (aggregates + config + seed + timestamp + versions).

### comparison.py
- `compare_baselines(results: dict[name -> eval dict]) -> DataFrame` (pandas) with rows =
  metric, columns = baseline; `comparison_table(results) -> str` (formatted text).
- Plots via performance_plots.py: grouped bar charts (rate, sinr, ee, outage), box plots
  of per-episode rates, convergence curve (from training CSV if available).

### experiments.py
- `ExperimentRunner(config, model_path)`:
  - `run_sweep(name, param_overrides: list[dict], episodes) -> DataFrame`:
    for each override dict (e.g. {"channel.csi_error": 0.1}) → build overridden config via
    `clone_with_overrides`, evaluate, collect row (param value + all aggregate metrics).
  - Built-in sweeps: `snr` (override transmit_power_dbm so snr_db matches the sweep values:
    compute required tx power from target snr), `ris_size` (rows=cols=sqrt(N), N in list),
    `phase_bits`, `user_count`, `velocity`, `csi_error`, `power`, `distance` (move RIS/UE
    region so mean BS-UE distance ≈ value — implement by scaling `scenario` positions:
    place RIS at (d, d/2) and UE region around it; document the mapping), `fading`.
  - `run_generalization(config, model_path, train_cond: dict, test_cond: dict, episodes)`:
    evaluate the trained model under test conditions different from training conditions;
    returns both-condition metrics so the report can show the gap honestly.
  - Output per experiment: `<run_dir>/<experiment_name>.csv` + `.json` + plots.

## 10. Visualization (`src/ris_autonomy/visualization/`)

Matplotlib (Agg backend for files; figures returned for Streamlit) + plotly for
interactive HTML. All functions take data (not env internals) and return `Figure`/`go.Figure`:

- topology.py: `plot_topology(data: dict) -> Figure` — 2D scatter: BS (triangle), RIS
  (rectangle), UEs (circles), direct links (solid), RIS links (dashed), labels with
  distances; `plot_topology_interactive(data) -> go.Figure`.
- phase_map.py: `plot_phase_grid(theta, rows, cols, title) -> Figure` — imshow heatmap of
  phase (circular colormap, e.g. 'hsv' or 'twilight'), annotate element indices (small),
  colorbar in radians; `plot_phase_histogram(theta, levels)`.
- heatmaps.py: `plot_sinr_heatmap` / `plot_rate_heatmap` over a 2D grid of UE positions
  (evaluate a fixed policy on a grid) — used by robustness views.
- training_plots.py: `plot_training_curves(csv_path) -> Figure` (reward + rate + sinr + ee
  vs timesteps, 4 subplots); `plot_convergence`.
- performance_plots.py: `plot_rate_vs_snr`, `plot_sinr_vs_snr`, `plot_rate_vs_ris_size`,
  `plot_vs_csi_error`, `plot_vs_velocity`, `plot_vs_phase_bits`, `plot_vs_users`,
  `plot_baseline_comparison` (bar), `plot_rate_cdf`, `plot_ee_comparison`,
  `plot_inference_latency` — all take DataFrames/arrays.
- `save_figure(fig, path_png, path_html=None)` — PNG always; HTML (plotly) when given.

## 11. Utils

- seeding.py: `set_global_seed(seed)` — random, os env, numpy (default_rng seed),
  torch.manual_seed + cuda if available, PYTHONHASHSEED. Deterministic.
- logging.py: `setup_logging(level, file=None)` — format
  `%(asctime)s %(levelname)s %(name)s: %(message)s`; idempotent (don't double-add handlers);
  `get_logger(name)`.
- io.py: `ensure_dir`, `save_json/load_json` (indent=2, default=str), `save_csv/load_csv`
  (pandas), `save_dataframe(df, path)` — parquet if path ends .parquet and pyarrow
  available, else CSV fallback with a warning; `create_run_dir(base, name) ->
  results/YYYYMMDD_HHMMSS_<name>` (never overwrite; if exists, append _1, _2...);
  `software_versions() -> dict` (python, torch, sb3, gymnasium, numpy, pandas, git hash
  if in a repo); `run_manifest(run_dir, config, seed, extra)` writes config.yaml +
  run_manifest.json (seed, versions, timestamp, config).

## 12. CLI (`src/ris_autonomy/cli.py`) + scripts/

argparse-based, subcommands (all also exposed via `scripts/*.py` thin wrappers that call
the same functions):
- `train --config PATH --seed INT --timesteps INT --device STR --output-dir PATH`
- `evaluate --model PATH --config PATH --episodes INT --deterministic/--stochastic --output-dir PATH`
- `benchmark --config PATH --output-dir PATH`
- `experiment --name {snr,ris_size,phase_bits,user_count,velocity,csi_error,power,distance,fading,generalization} --config PATH --model PATH --output-dir PATH`
- `generate-channels --config PATH --num INT --out PATH` (save batch channel realizations
  to parquet/csv: flatten complex to real/imag columns + metadata)
- `reproduce --run-dir PATH` (load saved config.yaml + seed from run manifest, re-run the
  recorded pipeline: train if a model is expected, else evaluate)
- `dashboard` (launch `streamlit run app/dashboard.py` via subprocess; clear error if
  streamlit missing)

Every command: setup logging, create run dir, save manifest, print a final summary with
paths. `python -m ris_autonomy.cli` must work (add `__main__.py` or cli `main()` guard).

## 13. Benchmark pipeline (`scripts/benchmark.py` → `ris_autonomy/evaluation/comparison.py`
+ a `run_benchmark` orchestrator, put it in `ris_autonomy/evaluation/benchmark.py` if cleaner)

`run_benchmark(config, output_dir, seed)`:
1. If no DRL model at `benchmark.drl_model_path` and `train_drl_if_missing`: train a quick
   DRL model (drl_timesteps_if_missing) into the run dir, log it.
2. Evaluate ALL of: no_ris, random_ris, greedy, conventional, drl — SAME scenario, SAME
   channel seed, SAME episodes (benchmark.episodes).
3. Produce: comparison table (CSV + JSON + printed), plots (rate/sinr/ee/outage bar charts,
   rate CDF, convergence if DRL training CSV exists), summary.json.
4. Nothing hardcoded — every number from actual execution.

## 14. Configs (configs/*.yaml)

- default.yaml: the defaults from §2 (28 GHz, 100 MHz, 30 dBm, 4 BS antennas, 8×8 RIS,
  2-bit, 2 users, static mobility, rician K=10dB, csi_error 0, PPO, 100k timesteps).
- quick_test.yaml: 4×4 RIS (16 elements), 1 user, 100 MHz, n_steps_per_episode 50,
  timesteps 5000, net_arch small, episodes 3 — MUST complete end-to-end in < ~3 min on CPU.
- single_user.yaml: default but users.count 1.
- multi_user.yaml: default but users.count 4, velocity 5.
- stress_test.yaml: 12×12 RIS (144 elements), 4 users, random_waypoint v=10, csi_error 0.1,
  timesteps 200000, n_steps_per_episode 200.
All five must pass `Config.from_yaml` validation.

## 15. Tests (pytest, tests/)

Real assertions, fast (use 4×4 RIS, short episodes, fixed seeds). At minimum:
1. test_channels: shapes (K,M)/(N,M)/(K,N) for K=2,M=4,N=16; complex dtype; path loss
   monotonic in distance; FSPL value check (known value); effective channel dimension
   check; CSI error reduces correlation (corr(H_tilde,H) < 1 and > 0 for small eps).
2. test_ris: quantizer levels = 2^B; quantize output in [0,2pi) and on the grid;
   action mapping [-1,1]→[0,2pi) monotonic and deterministic; switch_fraction in [0,1].
3. test_metrics: SINR formula vs hand-computed 2-element case; rate = B*log2(1+SINR)
   exact; BER monotonic decreasing in SINR, in [0,1]; EE = rate/power exact; Jain
   fairness = 1 for equal rates; outage threshold behavior.
4. test_environment: reset returns (obs, info) with correct obs shape & dtype & bounds;
   step returns 5-tuple; obs stays bounded after 20 random steps; episode truncates at
   n_steps; invalid config raises ConfigError.
5. test_reward: reward finite; reward breakdown sums to reward; weights respected
   (zero weights → zero contribution); switching penalty active when phases change.
6. test_baselines: each baseline runs 5 steps in the env harness without error, returns
   valid quantized phases; no_ris produces H_eff == h_direct.
7. test_smoke: create env → reset → 10 random actions → step → metrics finite → close;
   plus a 200-timestep PPO training run on quick_test config completes and saves a model
   that loads back (mark with a longer timeout; keep it fast: 200 steps).
8. Config validation: bad values (phase_bits 0, negative bandwidth, csi_error 1.5,
   unknown mobility) raise ConfigError; YAML round-trip preserves values.
9. Action mapping: deterministic, in-range, quantized (test in test_ris or test_environment).
10. Model load: save agent → load agent → same prediction on same obs (deterministic).

Run: `python -m pytest tests/ -q` — ALL must pass.

## 16. Acceptance criteria (you MUST run these and paste outputs into BUILD_REPORT.md)

1. `cd /opt/sandbox/workspace/RIS-AUTONOMY && python -m pytest tests/ -q` → all pass.
2. `python scripts/train.py --config configs/quick_test.yaml --timesteps 5000 --seed 42`
   → completes; creates results/<run>/ with config.yaml, run_manifest.json,
   metrics/episode_metrics.csv, models/final/{best_model.zip,final_model.zip,
   model_metadata.json}, plots/*.png, training_stats.json.
3. `python scripts/evaluate.py --model results/<run>/models/final/best_model.zip
   --config configs/quick_test.yaml --episodes 3` → metrics.json/csv + plots.
4. `python scripts/benchmark.py --config configs/quick_test.yaml` → 5-baseline comparison
   (trains quick DRL if missing), table + plots.
5. `python -m ris_autonomy.cli train --config configs/quick_test.yaml --timesteps 2000
   --seed 7` → works.
6. `python -m ris_autonomy.cli experiment --name csi_error_sweep --config
   configs/quick_test.yaml --model <best_model.zip>` → CSV with 5 rows + plot.
7. `python scripts/generate_channels.py --config configs/default.yaml --num 50
   --out data/processed/` → parquet/csv written.
8. `python scripts/reproduce.py --run-dir <run from step 2>` → re-runs from saved config.
9. DETERMINISM: run step 2 twice with the same seed (different run dirs) → the two
   episode_metrics.csv files must be IDENTICAL (or differences < 1e-9). Verify and report.
10. `python scripts/train.py --config configs/default.yaml --timesteps 2000 --seed 1`
   completes (sanity that the full-size config works, quick timesteps).

## 17. Quality rules (non-negotiable)

- Type hints on all public functions; numpy-style docstrings on all classes/functions.
- No `TODO`, no `pass  # later`, no placeholder returns, no fake data anywhere.
- Numerical sanity: `check_finite` on channels and metrics; custom `NumericalError`.
- Graceful degradation: no CUDA → WARNING + CPU; no pyarrow → CSV fallback; missing
  streamlit → clear CLI message; missing model → clear ModelLoadError.
- Vectorized NumPy (no per-element Python loops in the hot path).
- Logging: INFO on env init (RIS elements, users, algorithm), training start, checkpoint
  saves; WARNING on CUDA fallback; ERROR with context on failures.
- Keep functions small; no module > ~600 lines (split if needed).
- requirements.txt: pin the major deps loosely (>=) to what is installed.
- pyproject.toml: name ris-autonomy, version 0.1.0, requires-python >=3.11,
  [tool.pytest.ini_options] testpaths=["tests"], [tool.ruff] line-length 100.
- .gitignore: standard Python (venv, __pycache__, results/, models/*.zip, .env, data/*).
- LICENSE: MIT (Copyright 2026 RIS-AUTONOMY contributors).

## 18. Deliverable

When done, write `/opt/sandbox/workspace/RIS-AUTONOMY/BUILD_REPORT.md` containing:
- what was built (module by module, one line each)
- exact commands run for acceptance + their key outputs (test summary line, run dir paths,
  final metrics from the quick benchmark table)
- determinism verification result
- known limitations (honest list)
- anything you deviated from this brief and why

Then STOP. Do not start Phase 2 (dashboard/docs polish) — a later phase handles that.