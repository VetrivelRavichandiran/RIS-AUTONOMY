"""User-equipment mobility models.

Three movement models are supported, selected by ``users.mobility``:

* ``static``          -- positions never change.
* ``linear``          -- constant velocity, reflecting off the boundary.
* ``random_waypoint`` -- each user heads for a random target at its speed and
  picks a new target on arrival.

All models return positions clamped to the scenario boundary.
"""
from __future__ import annotations

import numpy as np


def random_initial_positions(
    rng: np.random.Generator,
    k_users: int,
    region: list[float],
) -> np.ndarray:
    """Sample K initial UE positions uniformly inside a rectangular region.

    Parameters
    ----------
    rng:
        NumPy random generator.
    k_users:
        Number of users.
    region:
        ``[x_min, y_min, x_max, y_max]`` in meters.

    Returns
    -------
    numpy.ndarray
        Positions, shape ``(K, 2)``.
    """
    x1, y1, x2, y2 = region
    return rng.uniform([x1, y1], [x2, y2], (k_users, 2))


class MobilityModel:
    """Advance UE positions according to the configured mobility model.

    Parameters
    ----------
    config:
        Validated configuration (provides the mobility model, boundary size,
        and per-user velocities).
    rng:
        NumPy random generator (used by ``random_waypoint``).
    """

    def __init__(self, config, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.targets: np.ndarray | None = None

    def _boundary(self) -> np.ndarray:
        """Return ``[width, height]`` of the scenario boundary (m)."""
        return np.array(
            [self.config.scenario.boundary_width_m, self.config.scenario.boundary_height_m]
        )

    def _random_targets(self, k_users: int) -> np.ndarray:
        """Sample K random waypoint targets inside the boundary."""
        width, height = self._boundary()
        return random_initial_positions(self.rng, k_users, [0.0, 0.0, width, height])

    def update(self, positions: np.ndarray, velocities: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Return the new positions after a time step ``dt``.

        Parameters
        ----------
        positions:
            Current positions, shape ``(K, 2)``.
        velocities:
            Current velocities, shape ``(K, 2)``.
        dt:
            Time step (s).

        Returns
        -------
        numpy.ndarray
            New positions, shape ``(K, 2)``, clamped to the boundary.
        """
        model = self.config.users.mobility
        if model == "static":
            return positions
        if model == "random_waypoint":
            return self._random_waypoint(positions, velocities, dt)
        return self._linear(positions, velocities, dt)

    # ------------------------------------------------------------- linear
    def _linear(self, positions: np.ndarray, velocities: np.ndarray, dt: float) -> np.ndarray:
        """Constant-velocity motion with reflection at the boundary."""
        limit = self._boundary()
        q = positions + velocities * dt
        # Reflect: fold the position back into [0, limit] using a triangle wave.
        return np.abs((q + limit) % (2.0 * limit) - limit)

    # ---------------------------------------------------- random_waypoint
    def _random_waypoint(self, positions: np.ndarray, velocities: np.ndarray, dt: float) -> np.ndarray:
        """Move toward a random waypoint, re-picking on arrival."""
        if self.targets is None:
            self.targets = self._random_targets(len(positions))
        delta = self.targets - positions
        dist = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        speed = np.linalg.norm(velocities, axis=1, keepdims=True)
        q = positions + (delta / dist) * speed * dt
        # Users that reached (or overshot) their target get a fresh target.
        arrived = np.linalg.norm(self.targets - q, axis=1) < speed[:, 0] * dt
        if arrived.any():
            self.targets[arrived] = self._random_targets(int(arrived.sum()))
        limit = self._boundary()
        return np.clip(q, [0.0, 0.0], limit)