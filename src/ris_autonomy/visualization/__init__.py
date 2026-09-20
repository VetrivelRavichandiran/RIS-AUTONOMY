"""Visualization for RIS-AUTONOMY.

Matplotlib is used for static figures (files, ``render``) and Plotly for
interactive HTML.  Every function takes plain data (arrays / dicts / DataFrames)
rather than environment internals, so plots are unit-testable and reusable by
both the CLI pipelines and the Streamlit dashboard.
"""
from .topology import plot_topology, plot_topology_interactive
from .phase_map import plot_phase_grid, plot_phase_histogram
from .heatmaps import plot_rate_heatmap, plot_sinr_heatmap
from .training_plots import plot_training_curves, plot_convergence
from .performance_plots import (
    plot_rate_vs_snr,
    plot_sinr_vs_snr,
    plot_rate_vs_ris_size,
    plot_vs_csi_error,
    plot_vs_velocity,
    plot_vs_phase_bits,
    plot_vs_users,
    plot_vs_power,
    plot_vs_distance,
    plot_fading_comparison,
    plot_baseline_comparison,
    plot_rate_cdf,
    plot_ee_comparison,
    plot_inference_latency,
)
from .io import save_figure

__all__ = [
    "plot_topology",
    "plot_topology_interactive",
    "plot_phase_grid",
    "plot_phase_histogram",
    "plot_rate_heatmap",
    "plot_sinr_heatmap",
    "plot_training_curves",
    "plot_convergence",
    "plot_rate_vs_snr",
    "plot_sinr_vs_snr",
    "plot_rate_vs_ris_size",
    "plot_vs_csi_error",
    "plot_vs_velocity",
    "plot_vs_phase_bits",
    "plot_vs_users",
    "plot_vs_power",
    "plot_vs_distance",
    "plot_fading_comparison",
    "plot_baseline_comparison",
    "plot_rate_cdf",
    "plot_ee_comparison",
    "plot_inference_latency",
    "save_figure",
]