"""
social/social_costmap.py
--------------------------
Derives a [0, 1] social cost map purely from static obstacle geometry.
No human labelling required.

Two heuristics are combined:
  1. Passage-width heuristic: cells in/near narrow passages get high cost
     (derived from ray-casting passage width in 4 cardinal directions).
  2. Open-space centre penalty: cells far from all obstacles get mild
     elevated cost (robots shouldn't idle in exposed areas).

Factory method `from_blockage_probability` (Phase 5) derives cost directly
from the UNet blockage probability map — passages the robots have learned
are risky automatically get high social cost without any labelling.

Paper reference: S-NAMO (Paper 2), Section III — Social Costmap.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class SocialCostmap:
    """
    Pre-computed social cost map for a given static obstacle layout.

    Parameters
    ----------
    grid            : 2-D numpy occupancy array (0=free, 1=wall/fixed, 2=box).
    narrow_threshold: passage width (cells) below which cost is maximised.
    open_weight     : scale for the open-space penalty (0 to disable).
    """

    def __init__(
        self,
        grid: np.ndarray,
        narrow_threshold: int = 3,
        open_weight: float = 0.3,
    ):
        self.grid = grid
        self.grid_size = grid.shape[0]
        self.narrow_threshold = narrow_threshold
        self.open_weight = open_weight
        self._map: np.ndarray = self._compute(grid)

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self, grid: np.ndarray) -> np.ndarray:
        gs = self.grid_size
        cost = np.zeros((gs, gs), dtype=float)

        # --- Heuristic 1: Passage-width cost ---
        for row in range(gs):
            for col in range(gs):
                if grid[row, col] != 0:
                    continue   # skip walls and boxes
                width = self._passage_width(grid, col, row)
                if width <= 0:
                    passage_cost = 1.0
                elif width < self.narrow_threshold:
                    passage_cost = 1.0 / max(1, width)
                else:
                    passage_cost = 0.0
                cost[row, col] = min(1.0, passage_cost)

        # --- Heuristic 2: Open-space centre penalty ---
        if self.open_weight > 0:
            # Distance from nearest wall/obstacle for each free cell
            dist_to_wall = self._dist_to_nearest_obstacle(grid)
            max_dist = dist_to_wall.max()
            if max_dist > 0:
                open_penalty = (dist_to_wall / max_dist) * self.open_weight
                cost = np.clip(cost + open_penalty, 0.0, 1.0)

        return cost

    def _passage_width(self, grid: np.ndarray, col: int, row: int) -> int:
        """Minimum ray-cast length in 4 cardinal directions (stops at wall)."""
        gs = self.grid_size
        widths = []
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            count = 0
            c, r = col + dc, row + dr
            while 0 <= c < gs and 0 <= r < gs and grid[r, c] == 0:
                count += 1
                c += dc
                r += dr
            widths.append(count)
        return min(widths)

    def _dist_to_nearest_obstacle(self, grid: np.ndarray) -> np.ndarray:
        """Euclidean distance from each free cell to the nearest wall/box."""
        gs = self.grid_size
        dist = np.full((gs, gs), float("inf"))
        obstacle_cells = np.argwhere(grid != 0)  # (row, col) pairs

        for row in range(gs):
            for col in range(gs):
                if grid[row, col] != 0:
                    dist[row, col] = 0.0
                    continue
                if len(obstacle_cells) == 0:
                    dist[row, col] = float(gs)
                    continue
                dists = np.hypot(
                    obstacle_cells[:, 1] - col,
                    obstacle_cells[:, 0] - row,
                )
                dist[row, col] = float(dists.min())

        return dist

    # ------------------------------------------------------------------
    # Factory: from blockage probability (Phase 5 — Novel Contribution)
    # ------------------------------------------------------------------

    @classmethod
    def from_blockage_probability(
        cls,
        blockage_map: np.ndarray,
        grid: np.ndarray,
        scale: float = 2.0,
    ) -> "SocialCostmap":
        """
        Derives social cost directly from UNet blockage probability output.

        Passages the fleet has learned to be risky (high blockage probability)
        automatically receive elevated social cost — no human annotation needed.

        Phase 5 — Novel Contribution.

        Parameters
        ----------
        blockage_map : (H, W) float array in [0, 1] from UNet's get_risk_map().
        grid         : underlying occupancy grid.
        scale        : multiplier on blockage probability before clamping to [0,1].
        """
        instance = cls.__new__(cls)
        instance.grid = grid
        instance.grid_size = grid.shape[0]
        instance.narrow_threshold = 3
        instance.open_weight = 0.0  # blockage map replaces open-space penalty
        instance._map = np.clip(blockage_map * scale, 0.0, 1.0).astype(float)

        # Zero out walls — they have no social cost (they're impassable anyway)
        instance._map[grid != 0] = 0.0
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cost(self, cell: tuple[int, int]) -> float:
        """Return the social cost [0, 1] for a given (col, row) cell."""
        col, row = cell
        if not (0 <= col < self.grid_size and 0 <= row < self.grid_size):
            return 1.0   # out-of-bounds: maximum cost
        return float(self._map[row, col])

    @property
    def map(self) -> np.ndarray:
        """The full (grid_size, grid_size) cost array."""
        return self._map.copy()
