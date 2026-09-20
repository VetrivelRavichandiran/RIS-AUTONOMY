# BUILD BRIEF — PHASE 2: COMPLETION + PROFESSIONALIZATION (RIS-AUTONOMY)

You are an elite senior wireless-communications + RL + scientific Python engineer.
The RIS-AUTONOMY project at `/opt/sandbox/workspace/RIS-AUTONOMY` has a WORKING core
engine (verified: physics, Gymnasium env, PPO training, model save/load, 5-baseline
benchmark harness, byte-identical determinism). Your job is to COMPLETE and
PROFESSIONALIZE it. Do NOT rewrite the working core from scratch — extend, fix, and
refactor it. Preserve the public API (class/function names) that the CLI, scripts,
tests, and dashboard depend on, unless you update all call sites.

## 0. Environment (CRITICAL — a previous attempt failed by ignoring this)
- Run EVERY Python command as: `/opt/sandbox/workspace/RIS-AUTONOMY/.venv/bin/python`
  (e.g. `.venv/bin/python -m pytest tests/ -q`, `.venv/bin/python scripts/train.py ...`,
  `.venv/bin/python -m ris_autonomy.cli ...`).
- Do NOT use bare `python3`/`python` (they don't see the deps). Do NOT create a venv.
  Do NOT pip install (space-constrained volume; everything needed is in the venv).
  If a package seems missing, verify with `.venv/bin/python -c "import <pkg>"` and report.
- CPU only: `torch.cuda.is_available()` is False. No CUDA.
- Installed: torch 2.14.0+cpu, gymnasium 1.3.0, stable-baselines3 2.9.0, streamlit,
  plotly, numpy 2.4.6, pandas 3.0.6, scipy, pytest, pyyaml, matplotlib, pyarrow,
  tensorboard.
- Read the original spec at `/opt/sandbox/workspace/RIS-AUTONOMY/BUILD_BRIEF_PHASE1.md`
  for the full intended design (config schema, math, acceptance criteria). This Phase 2
  brief supersedes it where they conflict.

## 1. What already works (do not break)
- `src/ris_autonomy/config.py` — dataclass Config, from_yaml/to_yaml/from_dict/to_dict,
  clone_with_overrides, validation raising ConfigError. (Minified but functional.)
- `channels/` — pathloss (free_space_db, log_distance_db, three_gpp_um_db, pathloss_db),
  fading (rician/rayleigh/no_fading), channel_models (ChannelRealization,
  compute_effective_channel, apply_csi_error), channel_generator (ChannelGenerator).
- `ris/` — quantization (PhaseQuantizer: levels, step, quantize, from_action),
  ris_surface (RISurface: set_theta, phase_matrix, switch_fraction, reset),
  phase_controller (PhaseController: action_to_phases, discrete_phases).
- `communications/` — signal_model (mrt_precoding, received_signal), sinr,
  achievable_rate (rate_bps), ber (ber_approx), energy_efficiency.
- `environment/` — mobility (MobilityModel, random_initial_positions), state
  (StateBuilder, StepData, observation_dim), wireless_env (RISAUTONOMYEnv with
  reset/step, observation_space, action_space, _measure, get_topology).
- `rl/` — rewards (RewardFunction), policies, callbacks (EpisodeMetricsCallback),
  agent (create_agent, save_agent, load_agent), training (train).
- `baselines/` — make_baseline + NoRIS, RandomRIS, GreedyRIS, ConventionalRIS.
- `evaluation/` — evaluator (evaluate_agent), metrics, comparison, experiments
  (run_sweep), benchmark (run_benchmark).
- `utils/` — seeding (set_global_seed), logging, io (run_manifest, etc.).
- `cli.py` + `scripts/*.py` — train/evaluate/benchmark/experiment/generate-channels/
  reproduce/dashboard subcommands.
- `configs/*.yaml` — 5 valid configs.
- `tests/` — 7 passing (but thin — expand them, §8).
- Determinism: two same-seed runs produce identical episode_metrics.csv (KEEP this).

## 2. NON-NEGOTIABLE FIXES (the user explicitly banned these; they currently exist)
1. **Remove the placeholder conventional baseline.** `ConventionalRIS(GreedyRIS): pass`
   is a literal placeholder and makes greedy/conventional show identical (fake) numbers.
   Implement a genuinely DIFFERENT conventional optimizer (see §5). After the fix,
   greedy and conventional MUST produce different (real) results in the benchmark.
2. **Remove `_plot_placeholder`** in cli.py. Replace all plotting with real functions
   from the new `visualization/` module (§6). No plot may be a placeholder.
3. **No `TODO`, no `pass  # later`, no placeholder returns, no fake data anywhere.**
   Grep the repo for `TODO`, `placeholder`, `pass$` and remove every instance.

## 3. CODE QUALITY (the user requires professional style; current code is minified)
Refactor ALL source files to professional quality:
- **Type hints** on every public function/method signature.
- **NumPy-style docstrings** on every class and public function (Args/Returns/Raises).
- **Small functions** — break single-line multi-statement bodies into readable multi-line
  functions. No 500-char lines. Line length ≤ 100 (ruff).
- **Meaningful variable names** (no single-letter locals except loop indices).
- **Module docstrings** explaining each module's role and, for the environment/state
  modules, the exact observation vector layout and every major tensor's dimensions.
- Keep modules under ~400 lines; split if larger.
- Do NOT change behavior while refactoring — the 7 existing tests must still pass
  (you will also add new ones).
- Add `render()` and `close()` to `RISAUTONOMYEnv` (spec requires both):
  - `render(mode="human")` → matplotlib topology figure (use visualization.topology);
    `render(mode="rgb_array")` → small np.ndarray. Keep it cheap.
  - `close()` → clean up any figure handles.

## 4. REAL BestModelCallback
- `rl/callbacks.py`: implement `BestModelCallback` (SB3 Callback) that tracks the best
  mean episode reward over a rolling window (last 10 episodes) and saves the model to
  `<output_dir>/models/final/best_model.zip` ONLY when strictly better than the recorded
  best (never overwrite with equal/worse). Persist the best value in a JSON sidecar
  `best_model.json`. Wire it into `training.py` alongside EpisodeMetricsCallback.
- `training.py` must save a DISTINCT best model (via the callback) and the final model
  (final weights). Do not save the same weights to both paths unconditionally.

## 5. REAL conventional + greedy baselines (vectorized, genuinely different)
- `GreedyRIS`: per step, coordinate descent over RIS elements maximizing SUM rate.
  For element n, try all 2^B quantized levels; use a vectorized partial-update:
  `H_ris_base` (contribution with element n zeroed) + `exp(1j*theta_l) * outer(h_ru[:,n],
  H_br[n,:])`, recompute sum rate for each candidate, keep the best. 2 passes over
  elements. Vectorize with NumPy (no per-level Python loops over N).
- `ConventionalRIS`: per step, ALTERNATING PER-USER optimization — for each user k,
  coordinate descent maximizing rate_k (interference from other users treated as fixed),
  2 passes over users. This is a conventional (non-DRL) optimization baseline and MUST
  differ from greedy (it optimizes per-user rate, not sum rate).
- Both must be fast enough for N=64, B=2 (256 candidate evaluations per element pass).
- Keep the common interface: `action(env) -> theta (N,)` (float32, in [-1,1] for the
  env's continuous action space, OR return phases and let the harness map — pick one,
  document it, and make the evaluator/benchmark use it consistently).
- `NoRIS` and `RandomRIS` stay as-is (already correct).

## 6. VISUALIZATION MODULE (currently EMPTY — build all of it)
`src/ris_autonomy/visualization/` — matplotlib (Agg for files; return Figures for
Streamlit) + plotly for interactive HTML. Every function takes DATA (not env internals)
and returns a `Figure`/`go.Figure`. Add real docstrings + type hints.
- `topology.py`: `plot_topology(data: dict) -> Figure` — 2D scatter: BS (triangle), RIS
  (rectangle), UEs (circles), direct links (solid), RIS links (dashed), distance labels;
  `plot_topology_interactive(data) -> go.Figure`.
- `phase_map.py`: `plot_phase_grid(theta, rows, cols, title) -> Figure` — imshow heatmap
  of phase (circular colormap), element-index annotations, colorbar in radians;
  `plot_phase_histogram(theta, levels) -> Figure`.
- `heatmaps.py`: `plot_rate_heatmap(values_2d, x, y, ...) -> Figure` and
  `plot_sinr_heatmap(...)` for robustness views (evaluate a policy on a 2D position grid).
- `training_plots.py`: `plot_training_curves(csv_path) -> Figure` (4 subplots: reward,
  rate_sum, sinr_db, ee vs episode/timesteps); `plot_convergence(...)`.
- `performance_plots.py`: `plot_rate_vs_snr`, `plot_sinr_vs_snr`, `plot_rate_vs_ris_size`,
  `plot_vs_csi_error`, `plot_vs_velocity`, `plot_vs_phase_bits`, `plot_vs_users`,
  `plot_vs_power`, `plot_vs_distance`, `plot_fading_comparison`, `plot_baseline_comparison`
  (grouped bar), `plot_rate_cdf`, `plot_ee_comparison`, `plot_inference_latency` — all
  take DataFrames/arrays.
- `io.py` (in visualization): `save_figure(fig, path_png, path_html=None)` — PNG always;
  HTML (plotly) when given.
- Wire these into cli.py / training.py / evaluator / benchmark / experiments so every
  pipeline produces REAL plots (training curves, evaluation, comparison, sweep plots).
  Remove `_plot_placeholder`.

## 7. FULL EXPERIMENT ENGINE (currently only csi_error/power/velocity)
`evaluation/experiments.py` — `ExperimentRunner` supporting ALL 9 sweep types +
generalization, each producing `<run_dir>/<name>.csv` + `.json` + a real plot:
- `snr` (override transmit_power_dbm so snr_db matches sweep values),
- `ris_size` (rows=cols=sqrt(N)),
- `phase_bits`,
- `user_count`,
- `velocity`,
- `csi_error`,
- `power`,
- `distance` (scale scenario positions so mean BS-UE distance ≈ value; document mapping),
- `fading` (rician/rayleigh),
- `generalization` (train under one condition, test under another; report the gap).
`cli.py experiment --name {snr,ris_size,phase_bits,user_count,velocity,csi_error,power,
distance,fading,generalization}` must dispatch to all of them.

## 8. EXPAND TESTS (currently 7 thin one-liners → the spec's 15 areas)
Keep the 7 passing; add real assertions covering AT MINIMUM:
1. channel dimensions (K,M)/(N,M)/(K,N) for K=2,M=4,N=16; complex dtype.
2. path-loss monotonic in distance + known FSPL value.
3. effective-channel dimension check.
4. CSI error reduces correlation (0 < corr < 1 for small eps).
5. RIS quantizer: levels=2^B; output in [0,2pi) and on the grid; from_action
   deterministic, in-range, quantized; switch_fraction in [0,1].
6. action mapping [-1,1]→[0,2pi) monotonic + deterministic.
7. SINR formula vs a hand-computed 2-element case.
8. rate = B*log2(1+SINR) exact.
9. BER monotonic decreasing in SINR, in [0,1].
10. EE = rate/power exact; Jain fairness = 1 for equal rates; outage threshold behavior.
11. reward: finite; breakdown sums to reward; zero weights → zero contribution;
    switching penalty active when phases change.
12. environment: reset returns (obs, info) correct shape/dtype/bounds; step returns
    5-tuple; obs stays bounded after 20 random steps; episode truncates at n_steps;
    render() and close() work; invalid config raises ConfigError.
13. each baseline runs 5 steps without error and returns valid phases; no_ris →
    H_eff == h_direct; greedy and conventional produce DIFFERENT phase vectors
    (assert not allclose) to prove they are distinct.
14. model save → load → identical deterministic prediction on the same obs.
15. config validation: bad values (phase_bits 0, negative bandwidth, csi_error 1.5,
    unknown mobility) raise ConfigError; YAML round-trip preserves values.
16. smoke: env → reset → 10 random actions → metrics finite → close; plus a short
    (≤300-step) PPO run on quick_test completes and the model loads back.
Run `.venv/bin/python -m pytest tests/ -q` — ALL must pass.

## 9. ENGINE DOCS (real content, not 1-liners)
Rewrite `docs/architecture.md`, `docs/configuration.md`, `docs/user_guide.md` with
substantive, accurate content (modules + data flow, env, channel model with tensor
dimensions, RIS model, RL system, evaluation, config reference table, step-by-step
usage). The README will be rewritten in Phase 3 — leave it as-is for now.

## 10. ACCEPTANCE (run with `.venv/bin/python`; paste real outputs into BUILD_REPORT.md)
1. `.venv/bin/python -m pytest tests/ -q` → all pass (report the count).
2. `.venv/bin/python scripts/train.py --config configs/quick_test.yaml --timesteps 5000
   --seed 42` → run dir with config.yaml, run_manifest.json, metrics/episode_metrics.csv,
   models/final/{best_model.zip,final_model.zip,model_metadata.json,best_model.json},
   plots/*.png (REAL training-curve plot, not a placeholder), training_stats.json.
3. `.venv/bin/python scripts/evaluate.py --model <best_model.zip> --config
   configs/quick_test.yaml --episodes 3` → metrics.json/csv + real plot.
4. `.venv/bin/python scripts/benchmark.py --config configs/quick_test.yaml` → 5-baseline
   table where **greedy != conventional** (different real numbers) + real comparison plot.
5. `.venv/bin/python -m ris_autonomy.cli experiment --name csi_error --config
   configs/quick_test.yaml --model <best_model.zip>` AND `--name snr` AND `--name
   phase_bits` AND `--name user_count` → each produces a CSV + real plot.
6. Determinism: two same-seed 5000-step runs → byte-identical episode_metrics.csv
   (report the diff result).
7. `grep -rn "TODO\|placeholder" src/ tests/ scripts/ app/` → no hits.
8. `.venv/bin/python -c "from ris_autonomy.environment.wireless_env import
   RISAUTONOMYEnv; ..."` → render('human') and close() work.
9. Every source file has type hints + docstrings (spot-check; report any file that
   lacks them).

## 11. Deliverable
Overwrite `/opt/sandbox/workspace/RIS-AUTONOMY/BUILD_REPORT.md` with: what was changed
per module, the acceptance outputs (test count, benchmark table showing greedy≠
conventional, determinism result, sweep outputs), and any deviations. Then STOP —
Phase 3 (dashboard + README) is separate.

When finished, your final message must include: (a) pytest summary line, (b) the 5-baseline
benchmark table (greedy and conventional must differ), (c) determinism result, (d) list of
sweep types verified, (e) any deviations.