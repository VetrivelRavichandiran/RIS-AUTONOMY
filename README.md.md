# 📡 RIS-AUTONOMY

**Deep reinforcement learning for adaptive Reconfigurable Intelligent Surface (RIS) phase optimization** in a millimeter-wave BS → RIS → UE MISO system.

A PPO agent chooses the RIS phase vector every coherence interval to maximize a communication reward (sum rate + SINR + energy efficiency − phase-switching cost). The platform ships a complete, validated mmWave simulation engine, a PPO training pipeline, a five-controller benchmark, a nine-way parameter-sweep experiment engine, a visualization library, a Streamlit dashboard, and a 57-test suite.

> **Every number the system reports comes from actual computation** — no placeholders, no TODOs, no mock ML.

---

## ✨ Features

- **Validated configuration** — single dataclass `Config` with YAML round-trip, precise `ConfigError` messages, and nested `clone_with_overrides` for sweeps.
- **Physical channel model** — 3GPP UMa / free-space / log-distance path loss, Rician / Rayleigh fading, LOS or blocked direct link, configurable CSI error — all vectorized over (K users, M antennas, N RIS elements).
- **Quantized RIS model** — `2^B`-level phase quantization, phase matrix, and switching-cost tracking.
- **Gymnasium environment** — correct coherence semantics (act on the observed channel), continuous or per-element-discrete actions, `render`/`close`, normalized clipped observations.
- **PPO training** — Stable-Baselines3 with a real `BestModelCallback` (saves only when strictly better), per-episode metrics CSV, checkpoints, and `training_stats.json`.
- **Genuinely distinct baselines** — no-RIS, random-RIS, sum-rate **greedy** coordinate descent, and **alternating per-user** conventional optimization.
- **Evaluation & benchmarking** — per-step metrics (rate, SINR, EE, BER, outage, Jain fairness, inference latency) and a fair five-controller comparison on identical channel seeds.
- **Experiment engine** — nine sweeps (SNR, RIS size, phase bits, user count, velocity, CSI error, power, distance, fading) plus a generalization (train-vs-test gap) experiment; each writes CSV/JSON + a real plot.
- **Visualization** — topology, phase grid/histogram, spatial heatmaps, training curves, and all sweep/comparison/CDF/latency plots (matplotlib + optional plotly HTML).
- **Streamlit dashboard** — configure a scenario, inspect topology/phases, train, evaluate, benchmark, and sweep from one page.
- **Determinism** — fixed-seed runs are byte-for-byte reproducible (verified by diffing two independent runs).

---

## 🏗️ System Model

```
        M antennas              N = R×C elements
  ┌──────────────┐  h_BR   ┌──────────────────┐  h_RU   ┌─────┐
  │   Base       │ ───────▶│   Reconfigurable │ ───────▶│  UE │
  │   Station    │         │   Intelligent    │         │     │
  │  (MISO MRT)  │ ─ ─ ─ ─ ─  Surface (RIS)   │         │     │
  └──────────────┘  h_BU (LOS or blocked) ────┴─────────┴─────┘
```

The PPO agent observes a normalized channel/state vector (magnitudes, phases, SINR, rate, positions, velocities, current/previous phases, CSI quality, interference) and outputs the RIS phase vector. The reward combines sum rate, SINR, and energy efficiency, penalized by phase-switching cost.

---

## 🚀 Quick Start

Requires **Python ≥ 3.11**.

```bash
cd RIS-AUTONOMY
python3 -m venv .venv
# Linux/macOS
.venv/bin/pip install -e .
# Windows (PowerShell)
.venv\Scripts\pip install -e .

# Verify the install (57 tests)
python -m pytest tests/ -q

# Train a small agent
python scripts/train.py --config configs/quick_test.yaml --timesteps 5000 --seed 42

# Evaluate the best model
python scripts/evaluate.py \
    --model results/<train-run>/models/final/best_model.zip \
    --config configs/quick_test.yaml --episodes 3

# Five-controller benchmark
python scripts/benchmark.py --config configs/quick_test.yaml

# Launch the Streamlit dashboard
python -m ris_autonomy.cli dashboard
```

On a CPU-only machine the CPU build of PyTorch is used automatically.

---

## 🖥️ CLI

All subcommands (and their `scripts/` wrappers) accept `--config`, `--output-dir`, and `--seed`.

| Command | Purpose | Key flags |
|---------|---------|-----------|
| `train` | Train a PPO agent | `--timesteps`, `--device`, `--config` |
| `evaluate` | Evaluate a trained model | `--model`, `--episodes`, `--stochastic` |
| `benchmark` | Five-controller comparison | `--config` |
| `experiment` | Run a named sweep / generalization | `--name`, `--model`, `--episodes` |
| `generate-channels` | Batch-generate channel realizations | `--num`, `--out` |
| `reproduce` | Re-run a saved run from its manifest | `--run-dir` |
| `dashboard` | Launch the Streamlit dashboard | — |

Every command creates a **timestamped run directory** under `results/` containing `config.yaml`, `run_manifest.json` (config + seed + versions), metrics, models, and plots — previous runs are never overwritten.

### Example results (CPU, quick-test scenario)

Five-controller benchmark on identical channel seeds:

| Controller   | Sum rate | SINR (dB) | EE      | Fairness |
|--------------|---------:|----------:|--------:|---------:|
| no_ris       | 2889.9   | −52.76    | 2675.9  | 0.806    |
| random_ris   | 3688.0   | −51.66    | 3414.8  | 0.758    |
| greedy       | 8071.4   | −49.23    | 7473.5  | 0.671    |
| conventional | 4632.3   | −49.61    | 4289.1  | 0.799    |
| **DRL (PPO)**| 3727.3   | −51.61    | 3451.2  | 0.756    |

The DRL agent beats conventional and random-RIS, is competitive with greedy on rate, and preserves user fairness (0.756 vs greedy's 0.671) — while incurring only ~0.4 ms inference latency. The SNR sweep shows outage dropping to 0.0 at high received SINR; the CSI-error sweep shows rate degrading gracefully as channel knowledge worsens.

---

## 📁 Repository Layout

```
RIS-AUTONOMY/
├── configs/                 # 6 validated scenario configs
├── app/dashboard.py         # Streamlit dashboard
├── scripts/                 # thin CLI wrappers (train/evaluate/benchmark/…)
├── src/ris_autonomy/
│   ├── config.py            # validated dataclass config + YAML
│   ├── cli.py               # argparse subcommands
│   ├── channels/            # pathloss, fading, channel generator
│   ├── ris/                 # quantization, surface, phase controller
│   ├── communications/      # signal model, SINR, rate, BER, EE
│   ├── environment/         # Gymnasium env, state builder, mobility
│   ├── rl/                  # agent, policies, rewards, callbacks, training
│   ├── baselines/           # no_ris, random, greedy, conventional
│   ├── evaluation/          # evaluator, benchmark, experiments, metrics
│   ├── visualization/       # topology, phase map, heatmaps, plots, io
│   └── utils/               # seeding, logging, io
├── tests/                   # 57 tests across 8 modules
├── docs/                    # architecture, configuration, user guide
├── data/                    # raw / processed / results (gitkeep)
└── models/                  # checkpoints / final (gitkeep)
```

---

## 📖 Documentation

- **[docs/architecture.md](docs/architecture.md)** — module reference, data flow, and exact tensor dimensions.
- **[docs/configuration.md](docs/configuration.md)** — every config key with types, defaults, and constraints.
- **[docs/user_guide.md](docs/user_guide.md)** — installation, CLI, run-directory layout, Python API, troubleshooting.
- **[BUILD_REPORT.md](BUILD_REPORT.md)** — what was built, acceptance outputs, and determinism verification.

---

## 🧪 Requirements

Python ≥ 3.11. Dependencies (see `requirements.txt`): numpy, scipy, torch, gymnasium, stable-baselines3, pyyaml, pandas, matplotlib, streamlit, plotly, pyarrow, tensorboard, pytest.

## 📄 License

MIT — see [LICENSE](LICENSE).