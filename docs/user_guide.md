# RIS-AUTONOMY — User Guide

A step-by-step guide to installing, running, and interpreting RIS-AUTONOMY.
The engine is a Python package (`ris_autonomy`) driven by YAML configs and an
argparse CLI.  Every command creates a timestamped run directory under
`results/`, writes a `run_manifest.json` (config + seed + versions), and prints
a summary with paths.

---

## 1. Installation

Requires Python ≥ 3.11.

```bash
cd RIS-AUTONOMY
python3 -m venv .venv
.venv/bin/pip install -e .
```

This installs the package plus its dependencies (numpy, scipy, gymnasium,
stable-baselines3, torch, pyyaml, pandas, matplotlib, streamlit, plotly,
pyarrow, tensorboard).  On a machine without CUDA, the CPU build of torch is
used automatically.

Run the test suite to verify the install:

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## 2. Quick start

The fastest end-to-end path: train a small agent, evaluate it, and benchmark
it against the conventional baselines.

```bash
# 1. Train (quick-test config, 5000 steps, seed 42)
.venv/bin/python scripts/train.py --config configs/quick_test.yaml --timesteps 5000 --seed 42

# 2. Evaluate the best model (3 episodes)
.venv/bin/python scripts/evaluate.py \
    --model results/<train-run>/models/final/best_model.zip \
    --config configs/quick_test.yaml --episodes 3

# 3. Five-controller benchmark (no_ris / random / greedy / conventional / drl)
.venv/bin/python scripts/benchmark.py --config configs/quick_test.yaml
```

Every command also works through the module CLI:

```bash
.venv/bin/python -m ris_autonomy.cli train --config configs/quick_test.yaml --timesteps 2000 --seed 7
```

---

## 3. The CLI

All subcommands (and their `scripts/` wrappers) accept `--config`,
`--output-dir`, `--seed`, and command-specific flags.

| Command | Purpose | Key flags |
|---------|---------|-----------|
| `train` | Train a PPO agent | `--timesteps`, `--device`, `--config` |
| `evaluate` | Evaluate a trained model | `--model`, `--episodes`, `--stochastic` |
| `benchmark` | Five-controller comparison | `--config` |
| `experiment` | Run a named sweep / generalization | `--name`, `--model`, `--episodes` |
| `generate-channels` | Batch-generate channel realizations | `--num`, `--out` |
| `reproduce` | Re-run a saved run from its manifest | `--run-dir` |
| `dashboard` | Launch the Streamlit dashboard | — |

### 3.1 `train`

```bash
.venv/bin/python scripts/train.py --config configs/quick_test.yaml --timesteps 5000 --seed 42
```

Creates `results/<timestamp>_train/` containing:
- `config.yaml`, `run_manifest.json`
- `metrics/episode_metrics.csv` (per-episode reward/rate/SINR/EE/switching)
- `models/final/best_model.zip` (rolling-best, saved only when strictly better)
- `models/final/final_model.zip` (final weights) + `model_metadata.json`
- `models/final/best_model.json` (recorded best reward + timestep)
- `plots/training.png` (real training curves)
- `training_stats.json` (timesteps, seed, duration, final eval)

### 3.2 `evaluate`

```bash
.venv/bin/python scripts/evaluate.py --model <best_model.zip> --config configs/quick_test.yaml --episodes 3
```

Writes `metrics.csv` / `metrics.json` (per-episode rows + aggregates) and
plots (rate CDF, inference-latency distribution).  `--stochastic` samples
actions instead of using the deterministic policy.

### 3.3 `benchmark`

```bash
.venv/bin/python scripts/benchmark.py --config configs/quick_test.yaml
```

Trains a quick DRL model if none is supplied, then evaluates **all five**
controllers on the same scenario and channel seed.  Writes `comparison.csv` /
`comparison.json`, `summary.json`, and comparison plots.  The table shows
`rate_sum`, `sinr_db`, `ee`, `outage`, `fairness` per controller.  The
benchmark runs with ≥ 2 users so `greedy` (sum-rate) and `conventional`
(per-user) are genuinely different.

### 3.4 `experiment`

Run any of the nine sweeps (each produces a CSV + JSON + a real plot):

```bash
.venv/bin/python -m ris_autonomy.cli experiment --name csi_error \
    --config configs/quick_test.yaml --model <best_model.zip> --episodes 3
```

`--name` ∈ `snr`, `ris_size`, `phase_bits`, `user_count`, `velocity`,
`csi_error`, `power`, `distance`, `fading`, or `generalization`.  The sweep
ranges come from the `experiment` section of the config.  `generalization`
evaluates a model under `train_condition` vs `test_condition` and reports the
gap.

### 3.5 `generate-channels`

```bash
.venv/bin/python scripts/generate_channels.py --config configs/default.yaml --num 50 --out data/processed/
```

Generates 50 channel realizations and writes `data/processed/channels.parquet`
(flattened real/imag statistics + distances).

### 3.6 `reproduce`

```bash
.venv/bin/python scripts/reproduce.py --run-dir results/<train-run>
```

Loads the saved `config.yaml` + seed from the run manifest and re-runs the
recorded pipeline into a fresh run directory.

### 3.7 `dashboard`

```bash
.venv/bin/python -m ris_autonomy.cli dashboard
```

Launches the Streamlit dashboard (`app/dashboard.py`) in a browser.  Use it to
browse configs, run quick training, view training curves, run benchmarks and
sweeps, and inspect the topology / phase configuration.

---

## 4. Run-directory layout

```
results/<YYYYMMDD_HHMMSS_micro>_<pipeline>/
├── config.yaml              # the exact config used
├── run_manifest.json        # seed, software versions, timestamp, pipeline
├── metrics/
│   └── episode_metrics.csv  # per-episode training metrics
├── models/
│   ├── final/
│   │   ├── best_model.zip   # rolling-best (only when strictly better)
│   │   ├── best_model.json  # recorded best reward + timestep
│   │   ├── final_model.zip  # final weights
│   │   └── model_metadata.json
│   └── checkpoints/         # intermediate checkpoints (if enabled)
├── plots/                   # real matplotlib figures (PNG)
├── training_stats.json      # (train) timesteps, seed, duration, final eval
├── comparison.csv/.json     # (benchmark) five-controller table
├── summary.json             # (benchmark) per-controller aggregates
└── sweep_*.csv/.json        # (experiment) sweep results
```

Run directories are timestamped and **never overwritten**, so previous
experiments are always preserved.

---

## 5. Using the Python API

```python
from ris_autonomy.config import Config
from ris_autonomy.environment.wireless_env import RISAUTONOMYEnv
from ris_autonomy.rl.training import train
from ris_autonomy.rl.agent import load_agent
from ris_autonomy.evaluation.evaluator import evaluate_agent
from ris_autonomy.evaluation.benchmark import run_benchmark

c = Config.from_yaml("configs/quick_test.yaml")

# Train
result = train(c, seed=42, timesteps=5000, output_dir="results/my_run")
model, meta = load_agent(result["model_path"], c)

# Evaluate
res = evaluate_agent(model, c, episodes=5, seed=42)
print(res["aggregates"])

# Benchmark
df = run_benchmark(c, "results/bench", seed=42)
print(df)
```

### Environment directly

```python
import numpy as np
env = RISAUTONOMYEnv(c, seed=1)
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(info["rate_sum"], info["sinr_db"])
env.render()   # topology figure
env.close()
```

---

## 6. Interpreting the metrics

| Metric | Meaning |
|--------|---------|
| `rate_sum` | Sum achievable rate across users (bps). |
| `sinr_db` | Mean per-user SINR (dB). |
| `ee` | Energy efficiency = `rate_sum / total_power` (bps/W). |
| `outage` | Fraction of users below `outage_target_bps`. |
| `fairness` | Jain's fairness index of the per-user rates (1 = equal). |
| `switching` | Fraction of RIS elements that changed phase (cost term). |
| `inference_latency_ms` | Median wall-clock time of one action (ms). |

**Reading a benchmark.** A good RIS controller beats `no_ris` (direct link
only) and `random_ris`.  `greedy` (sum-rate) and `conventional` (per-user) are
both strong conventional baselines; the DRL agent's value is in matching or
beating them with a learned policy that generalizes across conditions (see the
sweeps and the generalization experiment).

---

## 7. Reproducibility

Fix the seed and the run is byte-for-byte reproducible:

```bash
.venv/bin/python scripts/train.py --config configs/quick_test.yaml --timesteps 5000 --seed 42
.venv/bin/python scripts/train.py --config configs/quick_test.yaml --timesteps 5000 --seed 42
# the two metrics/episode_metrics.csv files are identical
```

`set_global_seed` seeds Python `random`, NumPy, PyTorch, and the hash seed.
The environment derives its per-episode RNG from `seed + episode`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: ris_autonomy` | Run `.venv/bin/pip install -e .` from the project root. |
| `CUDA unavailable, falling back to CPU` | Expected on CPU-only machines; harmless warning. |
| `ConfigError: ...` | A config value is invalid; the message names the field. |
| `ModelLoadError: ...` | The model path is missing or corrupt; retrain or point to a valid zip. |
| Slow training | Use `configs/quick_test.yaml`, a smaller `net_arch` (`small`), or fewer `timesteps`. |
| `No space left on device` during install | Purge the pip cache (`pip cache purge`) and retry; the CPU torch build is large. |

See `docs/architecture.md` for the module/tensor reference and
`docs/configuration.md` for every config key.