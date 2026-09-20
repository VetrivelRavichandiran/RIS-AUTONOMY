"""Channel generation for the BS -> RIS -> UE system.

For K users, M BS antennas and N RIS elements the generator produces::

    h_direct : (K, M) complex   direct BS -> UE channels
    H_br     : (N, M) complex   BS -> RIS channels
    h_ru     : (K, N) complex   RIS -> UE channels

Each tap is ``10**(-PL/20) * fading`` where ``PL`` is the path loss for the
link's distance and ``fading`` is the configured small-scale model.

Direct-link condition (``channel.direct_link``):

* ``"los"``     -- distance-based LOS/NLOS using ``los_distance_m``.
* ``"blocked"`` -- the direct path is obstructed: NLOS plus an extra
  ``blockage_db`` of loss, while the two RIS hops stay LOS.  This is the
  canonical mmWave RIS setting in which the reflected path is the useful one
  and RIS phase control materially changes the received signal.
"""
from __future__ import annotations

import numpy as np

from .channel_models import ChannelRealization
from .fading import fading_coefficients
from .pathloss import pathloss_db


class ChannelGenerator:
    """Generates full channel realizations for the current UE positions."""

    def __init__(self, config, rng: np.random.Generator) -> None:
        self.c = config
        self.rng = rng

    # ------------------------------------------------------------------ helpers
    def _distances(self, positions: np.ndarray):
        """3-D distances: BS->UE (K,), BS->RIS (scalar), RIS->UE (K,)."""
        c = self.c
        p = np.asarray(positions)
        bs = np.asarray(c.scenario.bs_position)
        ris = np.asarray(c.scenario.ris_position)
        h = c.scenario.bs_height_m

        def d(a, b, dz=0.0):
            return np.sqrt(((a - b) ** 2).sum(axis=-1) + dz * dz)

        d_bs_ue = d(p, bs, h)
        d_bs_ris = float(d(bs, ris, h))
        d_ris_ue = d(p, ris)
        return d_bs_ue, d_bs_ris, d_ris_ue

    def _gain(self, d, shape: tuple, los: np.ndarray, extra_db: float = 0.0) -> np.ndarray:
        """Amplitude * fading for a link of distance ``d`` with shape ``shape``."""
        c = self.c
        d = np.atleast_1d(np.asarray(d, dtype=float))
        pl = pathloss_db(
            d,
            c.system.carrier_frequency_hz,
            c.channel.model,
            np.broadcast_to(np.asarray(los), d.shape),
            c.channel.pathloss_exponent,
        ) + extra_db
        amp = 10.0 ** (-np.atleast_1d(pl) / 20.0)
        amp = np.broadcast_to(amp, (shape[0],))
        fade = fading_coefficients(self.rng, shape, c.channel.fading, c.channel.rician_k_db)
        return amp[:, None] * fade

    # ------------------------------------------------------------------ generate
    def generate(self, positions: np.ndarray) -> ChannelRealization:
        """Generate a :class:`ChannelRealization` for the given UE positions."""
        c = self.c
        p = np.asarray(positions)
        k = len(p)
        n = c.ris.rows * c.ris.columns
        m = c.bs.antennas
        d_bs_ue, d_bs_ris, d_ris_ue = self._distances(p)

        if c.channel.direct_link == "blocked":
            # Direct link obstructed: NLOS + blockage loss; RIS hops stay LOS.
            h_direct = self._gain(d_bs_ue, (k, m), np.zeros(k, dtype=bool), extra_db=c.channel.blockage_db)
        else:
            h_direct = self._gain(d_bs_ue, (k, m), d_bs_ue < c.channel.los_distance_m)

        H_br = self._gain(np.full(n, d_bs_ris), (n, m), np.ones(n, dtype=bool))
        h_ru = self._gain(d_ris_ue, (k, n), np.ones(k, dtype=bool))

        return ChannelRealization(
            h_direct,
            H_br,
            h_ru,
            p.copy(),
            {
                "distances": d_bs_ue.tolist(),
                "d_bs_ris": d_bs_ris,
                "d_ris_ue": d_ris_ue.tolist(),
            },
        )


def generate_batch(config, rng, positions, num_realizations: int):
    """Generate ``num_realizations`` independent realizations at fixed positions."""
    return [ChannelGenerator(config, rng).generate(positions) for _ in range(num_realizations)]