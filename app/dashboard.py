"""RIS-AUTONOMY interactive dashboard (Streamlit).

A single-page app to configure a scenario, inspect the topology and phase
configuration, run quick PPO training, evaluate a model, run the five-controller
benchmark, and run parameter sweeps — all wired to the real engine and the real
visualization functions.

Run with:
    .venv/bin/python -m ris_autonomy.cli dashboard
or:
    .venv/bin/python -m streamlit run app/dashboard.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from ris_autonomy.config import Config
from ris_autonomy.environment.wireless_env import RISAUTONOMYEnv

# Resolve the project root (parent of app/) so relative paths work from anywhere.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

CONFIGS = sorted(glob.glob(os.path.join("configs", "*.yaml")))
RESULTS_DIR = "results"

st.set_page_config(page_title="RIS-AUTONOMY", page_icon="📡", layout="wide")


# --------------------------------------------------------------------------- helpers
def _load_config(path: str) -> Config:
    return Config.from_yaml(path)


def _latest_run(name: str) -> str | None:
    """Most recent run directory of a given pipeline type, if any."""
    runs = sorted(glob.glob(os.path.join(RESULTS_DIR, f"*_{name}")))
    return runs[-1] if runs else None


def _find_model(path: str) -> str | None:
    """Return the model zip at ``path`` (accept a run dir or a direct zip)."""
    if not path:
        return None
    if path.endswith(".zip") and os.path.exists(path):
        return path
    for cand in ("best_model.zip", "final_model.zip"):
        full = os.path.join(path, "models", "final", cand)
        if os.path.exists(full):
            return full
    return None


def _episode_metrics(run_dir: str) -> str | None:
    p = os.path.join(run_dir, "metrics", "episode_metrics.csv")
    return p if os.path.exists(p) else None


# --------------------------------------------------------------------------- header
st.title("📡 RIS-AUTONOMY")
st.caption("Deep-RL platform for adaptive Reconfigurable Intelligent Surface phase optimization")

tab_cfg, tab_viz, tab_train, tab_eval, tab_bench, tab_sweep = st.tabs(
    ["⚙️ Configuration", "🗺️ Visualization", "🚀 Train", "📊 Evaluate", "🏁 Benchmark", "📈 Experiments"]
)

# =========================================================================== CONFIG
with tab_cfg:
    st.subheader("Scenario configuration")
    col1, col2 = st.columns([1, 1])
    with col1:
        cfg_choice = st.selectbox("Config file", CONFIGS, format_func=lambda p: os.path.basename(p))
        cfg = _load_config(cfg_choice)
    with col2:
        st.markdown("**Derived link parameters**")
        st.json(
            {
                "snr_db": round(cfg.snr_db, 2),
                "noise_power_w": float(f"{cfg.noise_power_w:.3e}"),
                "n_ris_elements": cfg.ris.rows * cfg.ris.columns,
                "n_users": cfg.users.count,
                "n_bs_antennas": cfg.bs.antennas,
                "phase_levels": 2 ** cfg.ris.phase_bits,
            }
        )

    st.markdown("**Full configuration**")
    st.json(cfg.to_dict())

    st.markdown("**Live overrides** (applied on top of the selected config)")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        o_users = st.number_input("Users (K)", 1, 8, cfg.users.count)
        o_bits = st.number_input("Phase bits (B)", 1, 6, cfg.ris.phase_bits)
    with oc2:
        o_csi = st.slider("CSI error", 0.0, 0.5, float(cfg.channel.csi_error), 0.05)
        o_mob = st.selectbox("Mobility", ["static", "linear", "random_waypoint"],
                             index=["static", "linear", "random_waypoint"].index(cfg.users.mobility))
    with oc3:
        o_tx = st.slider("Transmit power (dBm)", 0.0, 50.0, float(cfg.system.transmit_power_dbm), 1.0)
        o_ep = st.number_input("Steps / episode", 5, 500, cfg.rl.n_steps_per_episode, 5)

    overrides = {
        "users.count": int(o_users),
        "ris.phase_bits": int(o_bits),
        "channel.csi_error": float(o_csi),
        "users.mobility": o_mob,
        "system.transmit_power_dbm": float(o_tx),
        "rl.n_steps_per_episode": int(o_ep),
    }
    cfg_eff = cfg.clone_with_overrides(overrides)
    st.session_state["cfg_eff"] = cfg_eff
    st.success(f"Effective config: K={cfg_eff.users.count}, N={cfg_eff.ris.rows * cfg_eff.ris.columns}, "
               f"B={cfg_eff.ris.phase_bits}, SNR={cfg_eff.snr_db:.1f} dB, mobility={cfg_eff.users.mobility}")

# =========================================================================== VISUALIZATION
with tab_viz:
    st.subheader("Topology & phase configuration")
    cfg_eff = st.session_state.get("cfg_eff", _load_config(CONFIGS[0]))
    seed = st.number_input("Seed", 0, 100000, 42, key="viz_seed")
    if st.button("Generate topology & random phase"):
        env = RISAUTONOMYEnv(cfg_eff, int(seed))
        env.reset()
        st.session_state["topo"] = env.get_topology()
        st.session_state["theta"] = env.ris.theta.copy()
        st.session_state["rows"] = cfg_eff.ris.rows
        st.session_state["cols"] = cfg_eff.ris.columns
        env.close()

    if "topo" in st.session_state:
        from ris_autonomy.visualization import plot_phase_grid, plot_topology

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_topology(st.session_state["topo"], title="Wireless topology"))
        with c2:
            st.pyplot(plot_phase_grid(st.session_state["theta"], st.session_state["rows"],
                                      st.session_state["cols"], title="RIS phase configuration"))

# =========================================================================== TRAIN
with tab_train:
    st.subheader("Quick PPO training")
    cfg_eff = st.session_state.get("cfg_eff", _load_config(CONFIGS[0]))
    tc1, tc2, tc3 = st.columns(3)
    t_steps = tc1.number_input("Timesteps", 500, 200000, 5000, 500)
    t_seed = tc2.number_input("Seed", 0, 100000, 42, key="train_seed")
    t_dev = tc3.selectbox("Device", ["auto", "cpu"], index=0)

    if st.button("🚀 Run training", type="primary"):
        from ris_autonomy.rl.training import train

        progress = st.progress(0.0, text="Training…")
        try:
            result = train(cfg_eff, int(t_seed), int(t_steps),
                           os.path.join(RESULTS_DIR, "dashboard_train"), device=t_dev)
            st.session_state["last_train"] = result
            st.success(f"Training complete in {result['duration_s']:.1f}s → {result['output_dir']}")
        finally:
            progress.progress(1.0, text="Done")

    if "last_train" in st.session_state:
        run_dir = st.session_state["last_train"]["output_dir"]
        csv = _episode_metrics(run_dir)
        if csv:
            from ris_autonomy.visualization import plot_convergence, plot_training_curves

            st.markdown("**Training curves**")
            st.pyplot(plot_training_curves(csv))
            st.markdown("**Reward convergence**")
            st.pyplot(plot_convergence(csv))
            df = pd.read_csv(csv)
            st.dataframe(df.tail(20), use_container_width=True)

# =========================================================================== EVALUATE
with tab_eval:
    st.subheader("Evaluate a model")
    cfg_eff = st.session_state.get("cfg_eff", _load_config(CONFIGS[0]))
    e1, e2, e3 = st.columns([2, 1, 1])
    model_input = e1.text_input("Model path (run dir or .zip)", value="")
    e_eps = e2.number_input("Episodes", 1, 50, 3)
    e_det = e3.checkbox("Deterministic", value=True)
    e_seed = st.number_input("Seed", 0, 100000, 42, key="eval_seed")

    if st.button("Evaluate"):
        from ris_autonomy.evaluation.evaluator import evaluate_agent
        from ris_autonomy.rl.agent import load_agent

        model_path = _find_model(model_input)
        if not model_path:
            st.error("No model found at that path.")
        else:
            with st.spinner("Evaluating…"):
                model, _ = load_agent(model_path, cfg_eff)
                res = evaluate_agent(model, cfg_eff, int(e_eps), deterministic=bool(e_det),
                                     seed=int(e_seed))
            st.session_state["last_eval"] = res
            st.json(res["aggregates"])
            st.dataframe(pd.DataFrame(res["episodes"]), use_container_width=True)

    if "last_eval" in st.session_state:
        from ris_autonomy.visualization import plot_rate_cdf

        rates = [r["rate_sum"] for r in st.session_state["last_eval"]["episodes"]]
        st.pyplot(plot_rate_cdf(rates, title="CDF of per-episode sum rate"))

# =========================================================================== BENCHMARK
with tab_bench:
    st.subheader("Five-controller benchmark")
    cfg_eff = st.session_state.get("cfg_eff", _load_config(CONFIGS[0]))
    b1, b2 = st.columns(2)
    b_eps = b1.number_input("Episodes per controller", 1, 20, 3)
    b_seed = b2.number_input("Seed", 0, 100000, 42)

    if st.button("🏁 Run benchmark", type="primary"):
        from ris_autonomy.evaluation.benchmark import run_benchmark

        bench_cfg = cfg_eff.clone_with_overrides({"benchmark.episodes": int(b_eps)})
        with st.spinner("Running benchmark (trains a quick DRL model if needed)…"):
            df = run_benchmark(bench_cfg, os.path.join(RESULTS_DIR, "dashboard_bench"), int(b_seed))
        st.session_state["bench_df"] = df
        st.dataframe(df, use_container_width=True)

    if "bench_df" in st.session_state:
        from ris_autonomy.visualization import plot_baseline_comparison, plot_ee_comparison

        df = st.session_state["bench_df"]
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_baseline_comparison(df, "rate_sum", "Sum rate (bps)", "Achievable rate"))
        with c2:
            st.pyplot(plot_baseline_comparison(df, "sinr_db", "Mean SINR (dB)", "SINR"))
        st.pyplot(plot_ee_comparison(df))

# =========================================================================== EXPERIMENTS
with tab_sweep:
    st.subheader("Parameter sweeps & generalization")
    cfg_eff = st.session_state.get("cfg_eff", _load_config(CONFIGS[0]))
    sweeps = ["snr", "ris_size", "phase_bits", "user_count", "velocity",
              "csi_error", "power", "distance", "fading"]
    s1, s2, s3 = st.columns([2, 1, 1])
    s_name = s1.selectbox("Sweep", sweeps)
    s_model = s2.text_input("Model (run dir or .zip)", value="")
    s_eps = s3.number_input("Episodes", 1, 20, 3)

    if st.button("📈 Run sweep", type="primary"):
        from ris_autonomy.evaluation.experiments import _sweep_dispatch
        from ris_autonomy.rl.agent import load_agent

        model_path = _find_model(s_model)
        if not model_path:
            st.error("Provide a model path to sweep.")
        else:
            with st.spinner("Running sweep…"):
                model, _ = load_agent(model_path, cfg_eff)
                df = _sweep_dispatch(s_name, model, cfg_eff, int(s_eps))
            st.session_state["sweep_df"] = df
            st.session_state["sweep_name"] = s_name
            st.dataframe(df, use_container_width=True)

    if "sweep_df" in st.session_state:
        from ris_autonomy.visualization import (
            plot_fading_comparison, plot_rate_vs_ris_size, plot_rate_vs_snr,
            plot_vs_csi_error, plot_vs_distance, plot_vs_phase_bits, plot_vs_power,
            plot_vs_users, plot_vs_velocity,
        )

        df = st.session_state["sweep_df"]
        name = st.session_state["sweep_name"]
        x = df["value"].values
        plot = {
            "snr": lambda: plot_rate_vs_snr(df["snr_db"].values, df["rate_sum"].values),
            "ris_size": lambda: plot_rate_vs_ris_size(x, df["rate_sum"].values),
            "phase_bits": lambda: plot_vs_phase_bits(x, df["rate_sum"].values),
            "user_count": lambda: plot_vs_users(x, df["rate_sum"].values),
            "velocity": lambda: plot_vs_velocity(x, df["rate_sum"].values),
            "csi_error": lambda: plot_vs_csi_error(x, df["rate_sum"].values),
            "power": lambda: plot_vs_power(x, df["rate_sum"].values),
            "distance": lambda: plot_vs_distance(x, df["rate_sum"].values),
            "fading": lambda: plot_fading_comparison(x, df["rate_sum"].values),
        }.get(name)
        if plot:
            st.pyplot(plot())