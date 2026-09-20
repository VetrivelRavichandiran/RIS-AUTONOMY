"""Conventional (non-DRL) RIS phase controllers used as baselines.

Every baseline exposes the same interface::

    class Baseline:
        name: str
        def action(self, env) -> np.ndarray:
            # Return a continuous action in [-1, 1] of shape (N,) that the
            # environment maps to a quantized RIS phase vector.

The controllers are evaluated by the *same* harness (``evaluation.evaluator`` /
``evaluation.benchmark``) on the *same* channel realizations (same seed) so that
comparisons are fair.  The coordinate-descent core is vectorized: for each RIS
element every candidate level is scored with a single matrix multiply, and the
running effective channel is updated incrementally (no full recompute per level).
"""
from __future__ import annotations

import numpy as np

from ..communications.achievable_rate import rate_bps
from ..communications.signal_model import mrt_precoding, received_signal
from ..channels.channel_models import compute_effective_channel


def _sum_rate_from_heff(h_eff: np.ndarray, p_tx_w: float, noise_w: float, bandwidth_hz: float) -> float:
    """Sum achievable rate (bps) for an effective channel ``h_eff`` (K, M).

    Uses maximum-ratio transmit (MRT) precoding on the true effective channel,
    the same precoder the environment uses, so the rate is directly comparable
    to the DRL agent's.
    """
    w = mrt_precoding(h_eff)
    sinr, _ = received_signal(h_eff, w, p_tx_w, noise_w)
    return float(rate_bps(sinr, bandwidth_hz).sum())


def _user_rate_from_heff(h_eff: np.ndarray, k: int, p_tx_w: float, noise_w: float,
                         bandwidth_hz: float) -> float:
    """Achievable rate (bps) of user ``k`` for an effective channel (K, M)."""
    w = mrt_precoding(h_eff)
    sinr, _ = received_signal(h_eff, w, p_tx_w, noise_w)
    return float(rate_bps(sinr[k], bandwidth_hz))


def _coordinate_descent(
    h_direct: np.ndarray,
    H_br: np.ndarray,
    h_ru: np.ndarray,
    theta: np.ndarray,
    levels: np.ndarray,
    objective,
    passes: int = 2,
) -> np.ndarray:
    """Vectorized coordinate descent over RIS elements.

    The effective channel is ``H_eff = h_direct + sum_n exp(1j*theta_n) *
    outer(h_ru[:, n], H_br[n, :])``.  For element ``n`` the current contribution
    is removed, every candidate level is scored at once (one matrix multiply),
    and the best level is kept; the running ``H_eff`` is updated incrementally.

    ``objective`` is a callable ``h_eff (K, M) -> float``.
    """
    n = theta.shape[0]
    outer_all = h_ru[:, None, :] * H_br.T[None, :, :]  # (K, M, N) per-element outer products
    theta = theta.copy()
    h_eff = compute_effective_channel(h_direct, H_br, h_ru, theta)
    for _ in range(passes):
        for n_idx in range(n):
            outer_n = outer_all[:, :, n_idx]
            base_without_n = h_eff - np.exp(1j * theta[n_idx]) * outer_n
            cand = base_without_n[:, :, None] + outer_n[..., None] * np.exp(1j * levels)  # (K, M, L)
            scores = np.empty(levels.size)
            for l in range(levels.size):
                scores[l] = objective(cand[:, :, l])
            best = int(np.argmax(scores))
            h_eff = base_without_n + np.exp(1j * levels[best]) * outer_n
            theta[n_idx] = levels[best]
    return theta


class NoRIS:
    """Disable the RIS entirely so the effective channel is the direct link only.

    Sets ``env.ris_enabled = False`` so the environment computes
    ``H_eff = h_direct`` (the reflected path is removed).  The returned action
    is all zeros because the phases are ignored when the RIS is disabled.
    """

    name = "no_ris"

    def action(self, env) -> np.ndarray:
        env.ris_enabled = False
        return np.zeros(env.n, dtype=np.float32)


class RandomRIS:
    """Random continuous action, re-sampled every step."""

    name = "random_ris"

    def action(self, env) -> np.ndarray:
        return env.action_space.sample()


class GreedyRIS:
    """Local sum-rate maximization via coordinate descent over RIS elements."""

    name = "greedy"

    def action(self, env) -> np.ndarray:
        ch = env.last_channel
        p_tx_w = env.p_tx_w
        noise_w = env.noise_power_w
        bw = env.config.system.bandwidth_hz
        levels = env.ris.quantizer.levels_array

        def objective(h_eff: np.ndarray) -> float:
            return _sum_rate_from_heff(h_eff, p_tx_w, noise_w, bw)

        theta = _coordinate_descent(ch.h_direct, ch.H_br, ch.h_ru, env.ris.theta, levels, objective, passes=2)
        return ((theta / np.pi) - 1.0).clip(-1.0, 1.0).astype(np.float32)


class ConventionalRIS:
    """Conventional alternating per-user rate maximization.

    Distinct from :class:`GreedyRIS`: instead of maximizing the *sum* rate, it
    alternates over users and, for each user, runs coordinate descent maximizing
    that user's own rate (interference from the others treated as fixed).  This
    is a standard conventional (non-DRL) multi-user RIS optimization and yields
    different phase configurations than sum-rate greedy.
    """

    name = "conventional"

    def action(self, env) -> np.ndarray:
        ch = env.last_channel
        p_tx_w = env.p_tx_w
        noise_w = env.noise_power_w
        bw = env.config.system.bandwidth_hz
        levels = env.ris.quantizer.levels_array
        k = ch.h_direct.shape[0]
        theta = env.ris.theta.copy()

        for _ in range(2):  # 2 passes over users
            for k_idx in range(k):

                def objective(h_eff: np.ndarray, _k: int = k_idx) -> float:
                    return _user_rate_from_heff(h_eff, _k, p_tx_w, noise_w, bw)

                theta = _coordinate_descent(ch.h_direct, ch.H_br, ch.h_ru, theta, levels, objective, passes=1)
        return ((theta / np.pi) - 1.0).clip(-1.0, 1.0).astype(np.float32)


_REGISTRY = {
    "no_ris": NoRIS,
    "random_ris": RandomRIS,
    "greedy": GreedyRIS,
    "conventional": ConventionalRIS,
}


def make_baseline(name: str):
    """Instantiate a baseline controller by name."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name]()