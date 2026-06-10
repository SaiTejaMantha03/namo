"""
multi_robot/deadlock_resolution.py
------------------------------------
Deadlock resolution strategies from MR-NAMO (Paper 5) plus the novel
SR-width tie-breaking from Phase 5.

Strategies
----------
Repulsive DR  (Paper 5, Algorithm 2):
    Postponer = robot with smallest L2 norm of position vector.
    Evaders A* to the cell maximally far from all other robots.

Social DR     (Paper 5, Algorithm 3):
    Postponer = robot whose best evasion cell has the highest social cost.
    Evaders minimise w1 * dist_to_goal + w2 * social_cost.

SR-Width DR   (Phase 5 — Novel Contribution):
    Postponer = robot with the widest SR confidence interval.
    The most uncertain robot yields; the robot that knows it can succeed goes first.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, TYPE_CHECKING

from core.planner import a_star

if TYPE_CHECKING:
    from multi_robot.belief_broadcaster import BeliefBroadcaster
    from social.social_costmap import SocialCostmap


def _l2_norm(cell: tuple[int, int]) -> float:
    return math.sqrt(cell[0] ** 2 + cell[1] ** 2)


def _max_dist_cell(
    grid: np.ndarray,
    grid_size: int,
    other_cells: list[tuple],
) -> Optional[tuple[int, int]]:
    """Find the free cell that maximises the minimum distance to all other_cells."""
    best_cell = None
    best_dist = -1.0
    for row in range(grid_size):
        for col in range(grid_size):
            if grid[row, col] != 0:
                continue
            if (col, row) in other_cells:
                continue
            min_d = min(
                math.hypot(col - oc[0], row - oc[1]) for oc in other_cells
            )
            if min_d > best_dist:
                best_dist = min_d
                best_cell = (col, row)
    return best_cell


class DeadlockResolver:
    """
    Resolves multi-robot deadlocks using one of three strategies.

    Parameters
    ----------
    grid      : 2-D numpy occupancy array.
    grid_size : cells per side.
    w1        : weight for distance-to-goal in Social DR (default 1.0).
    w2        : weight for social cost in Social DR (default 1.5).
    """

    def __init__(
        self,
        grid: np.ndarray,
        grid_size: int,
        w1: float = 1.0,
        w2: float = 1.5,
    ):
        self.grid = grid
        self.grid_size = grid_size
        self.w1 = w1
        self.w2 = w2

    # ------------------------------------------------------------------
    # Repulsive DR (Paper 5, Algorithm 2)
    # ------------------------------------------------------------------

    def resolve_repulsive(
        self,
        conflicting_robots: list[int],
        robot_cells: dict[int, tuple],
        robot_goals: dict[int, tuple],
    ) -> dict[int, str]:
        """
        Postponer = robot with smallest |position| (L2 norm).
        Evaders get a new waypoint: the free cell furthest from all other robots.

        Returns dict: {robot_id: "WAIT" | "EVADE"}.
        """
        if not conflicting_robots:
            return {}

        # Postponer: smallest L2 norm
        postponer = min(conflicting_robots, key=lambda r: _l2_norm(robot_cells[r]))

        assignments = {r: "EVADE" for r in conflicting_robots}
        assignments[postponer] = "WAIT"

        # Compute evasion target for each evader
        evasion_targets: dict[int, Optional[tuple]] = {}
        for rid in conflicting_robots:
            if rid == postponer:
                continue
            other_cells = [robot_cells[r] for r in conflicting_robots if r != rid]
            target = _max_dist_cell(self.grid, self.grid_size, other_cells)
            evasion_targets[rid] = target

        return assignments, evasion_targets

    # ------------------------------------------------------------------
    # Social DR (Paper 5, Algorithm 3)
    # ------------------------------------------------------------------

    def resolve_social(
        self,
        conflicting_robots: list[int],
        robot_cells: dict[int, tuple],
        robot_goals: dict[int, tuple],
        social_map: "SocialCostmap",
    ) -> tuple[dict[int, str], dict[int, Optional[tuple]]]:
        """
        Postponer = robot whose best evasion cell has the highest social cost.
        Evaders minimise w1 * dist_to_goal + w2 * social_cost.

        Returns (assignments, evasion_targets).
        """
        if not conflicting_robots:
            return {}, {}

        best_evasion: dict[int, tuple] = {}
        best_evasion_cost: dict[int, float] = {}

        for rid in conflicting_robots:
            goal = robot_goals[rid]
            best_cell = None
            best_score = float("inf")

            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if self.grid[row, col] != 0:
                        continue
                    cell = (col, row)
                    if cell == robot_cells[rid]:
                        continue
                    dist = math.hypot(col - goal[0], row - goal[1])
                    sc = social_map.cost(cell)
                    score = self.w1 * dist + self.w2 * sc
                    if score < best_score:
                        best_score = score
                        best_cell = cell

            best_evasion[rid] = best_cell
            best_evasion_cost[rid] = social_map.cost(best_cell) if best_cell else float("inf")

        # Postponer: robot whose best evasion cell has the *highest* social cost
        postponer = max(conflicting_robots, key=lambda r: best_evasion_cost.get(r, 0.0))

        assignments = {r: "EVADE" for r in conflicting_robots}
        assignments[postponer] = "WAIT"

        evasion_targets = {r: (best_evasion.get(r) if r != postponer else None)
                           for r in conflicting_robots}

        return assignments, evasion_targets

    # ------------------------------------------------------------------
    # SR-Width DR (Phase 5 — Novel Contribution)
    # ------------------------------------------------------------------

    def resolve_sr_width(
        self,
        conflicting_robots: list[int],
        robot_cells: dict[int, tuple],
        robot_goals: dict[int, tuple],
        broadcaster: "BeliefBroadcaster",
    ) -> tuple[dict[int, str], dict[int, Optional[tuple]]]:
        """
        Postponer = robot with the WIDEST SR confidence interval.
        The robot most uncertain about manipulation success yields.
        The robot that knows it can succeed goes first.

        This replaces the arbitrary "smallest position vector" heuristic from
        Repulsive DR with a principled, uncertainty-grounded decision.

        Phase 5 — Novel Contribution.
        """
        if not conflicting_robots:
            return {}, {}

        widths = {
            rid: broadcaster.get_sr_interval_width(rid)
            for rid in conflicting_robots
        }

        # Postponer: most uncertain robot
        postponer = max(widths, key=widths.get)

        assignments = {r: "EVADE" for r in conflicting_robots}
        assignments[postponer] = "WAIT"

        # Evaders use the same repulsive evasion target
        evasion_targets: dict[int, Optional[tuple]] = {}
        for rid in conflicting_robots:
            if rid == postponer:
                evasion_targets[rid] = None
                continue
            other_cells = [robot_cells[r] for r in conflicting_robots if r != rid]
            target = _max_dist_cell(self.grid, self.grid_size, other_cells)
            evasion_targets[rid] = target

        return assignments, evasion_targets
