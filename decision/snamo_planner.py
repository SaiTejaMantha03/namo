"""
decision/snamo_planner.py
--------------------------
S-NAMO (Social NAMO) Planner.

Extends NAMOUnc with two social constraints:
  1. Taboo zones  — obstacles must NEVER be pushed into these cells
  2. Social costmap — when multiple push directions are valid, pick the
     one whose destination cell has the LOWEST social cost

The BYPASS vs REMOVE decision still uses the Laplace / NAMOUnc criterion.
The difference is in HOW removal is executed: socially-aware direction.

Paper reference: S-NAMO (Paper 2) — Section III (Costmap) + IV (Taboo zones)

Usage:
    from decision.snamo_planner import SNAMOPlanner
    planner = SNAMOPlanner(grid, grid_size)
    action, waypoints = planner.plan(start, goal, box_cells)
    planner.observe(success=True)   # after a real push
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.planner import a_star
from uncertainty.action_uncertainty import ManipulationBeliefModel
from uncertainty.bypass_model import TrajectoryRegressionModel
from uncertainty.interval_decision import choose_action
from social.social_costmap import SocialCostmap
from social.taboo_zones import TabooZoneManager


class SNAMOPlanner:
    """
    Social NAMO planner with uncertainty-aware BYPASS/REMOVE decisions.

    Parameters
    ----------
    grid         : 2-D occupancy array (0=free, 1=wall, 2=box).
    grid_size    : cells per side.
    taboo_zones  : list of polygon vertex lists for taboo regions, e.g.
                   [[(2,3),(4,3),(4,6),(2,6)]].  Empty list = no taboo zones.
    alpha, beta  : Beta prior for manipulation SR (default 90% prior).
    social_weight: how strongly social cost influences A* path planning.
    """

    def __init__(
        self,
        grid: np.ndarray,
        grid_size: int,
        taboo_zones: list = None,
        alpha: float = 9.0,
        beta: float = 1.0,
        social_weight: float = 5.0,
    ):
        self.grid = grid
        self.grid_size = grid_size
        self.social_weight = social_weight

        # Uncertainty model (NAMOUnc part)
        self.belief = ManipulationBeliefModel(alpha=alpha, beta=beta)
        self._traj_model = TrajectoryRegressionModel()

        # Social costmap — derived from grid geometry (no human labels)
        self.social_map = SocialCostmap(grid)

        # Taboo zone manager
        zones = taboo_zones if taboo_zones else []
        self.taboo = TabooZoneManager(zones, grid_size)

        # Cache social risk map as numpy array for A*
        self._social_risk: np.ndarray = self.social_map.map

    # ------------------------------------------------------------------
    # Belief update
    # ------------------------------------------------------------------

    def observe(self, success: bool) -> None:
        """Call after each real push attempt to update the SR belief."""
        self.belief.observe(success)

    def update_grid(self, grid: np.ndarray) -> None:
        """Recompute social map when the grid changes (box moved)."""
        self.grid = grid
        self.social_map = SocialCostmap(grid)
        self._social_risk = self.social_map.map

    # ------------------------------------------------------------------
    # Social-aware clearing direction
    # ------------------------------------------------------------------

    def _social_clearing_directions(
        self,
        box_cell: tuple,
        grid: np.ndarray,
        other_robots: list = (),
        locked_zones: set = frozenset(),
    ) -> list[tuple[tuple, tuple]]:
        """
        Find all valid push directions whose destinations (clear_cell) are not in a taboo zone,
        sorted by social cost ascending.

        Returns list of (clear_cell, approach_cell).
        """
        bx, by = box_cell
        robot_set = set(map(tuple, other_robots))
        blocked = set(map(tuple, locked_zones))

        candidates = [
            ((bx, by + 1), (bx, by - 1)),   # push North
            ((bx, by - 1), (bx, by + 1)),   # push South
            ((bx + 1, by), (bx - 1, by)),   # push East
            ((bx - 1, by), (bx + 1, by)),   # push West
        ]

        valid = []
        for clear_cell, approach_cell in candidates:
            cx, cy = clear_cell
            ax, ay = approach_cell

            # Bounds check
            if not (0 <= cx < self.grid_size and 0 <= cy < self.grid_size):
                continue
            if not (0 <= ax < self.grid_size and 0 <= ay < self.grid_size):
                continue

            # Clear cell must be free
            if grid[cy, cx] != 0:
                continue
            if clear_cell in robot_set or clear_cell in blocked:
                continue

            # Taboo zone check — never push into taboo
            if self.taboo.blocks(clear_cell):
                continue

            # Approach cell must be free (no wall AND no box)
            if grid[ay, ax] == 1 or grid[ay, ax] == 2:
                continue
            if approach_cell in robot_set or approach_cell in blocked:
                continue

            social_cost = self.social_map.cost(clear_cell)
            valid.append((social_cost, clear_cell, approach_cell))

        if not valid:
            return []

        # Sort by lowest social cost at destination
        valid.sort(key=lambda x: x[0])
        return [(clear_cell, approach_cell) for _, clear_cell, approach_cell in valid]

    # ------------------------------------------------------------------
    # Internal cost intervals (same as NAMOUnc, with social-weighted A*)
    # ------------------------------------------------------------------

    def _bypass_interval(self, grid, start, goal, obstacle_pos):
        g = grid.copy()
        g[obstacle_pos[1], obstacle_pos[0]] = 2
        path = a_star(start, goal, g, self.grid_size,
                      risk_map=self._social_risk,
                      risk_weight=self.social_weight)
        if not path:
            return (float("inf"), float("inf"))
        base = float(len(path) - 1)
        return self._traj_model.predict_interval(path, base)

    def _removal_interval(self, grid, start, goal, obstacle_pos):
        g_obs = grid.copy()
        g_obs[obstacle_pos[1], obstacle_pos[0]] = 2

        path_to = a_star(start, obstacle_pos, g_obs, self.grid_size,
                         risk_map=self._social_risk,
                         risk_weight=self.social_weight)
        if not path_to:
            return (float("inf"), float("inf"))
        reach_lo, reach_hi = self._traj_model.predict_interval(
            path_to, float(len(path_to) - 1))

        g_clear = grid.copy()
        g_clear[obstacle_pos[1], obstacle_pos[0]] = 0
        path_from = a_star(obstacle_pos, goal, g_clear, self.grid_size,
                           risk_map=self._social_risk,
                           risk_weight=self.social_weight)
        if not path_from:
            return (float("inf"), float("inf"))
        from_lo, from_hi = self._traj_model.predict_interval(
            path_from, float(len(path_from) - 1))

        sr_lo, sr_hi = self.belief.success_rate_interval()
        push_lo = 3.0 / max(sr_hi, 0.05)
        push_hi = 3.0 / max(sr_lo, 0.05)

        return (reach_lo + push_lo + from_lo, reach_hi + push_hi + from_hi)

    # ------------------------------------------------------------------
    # Main planning API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: tuple,
        goal: tuple,
        box_cells: list,
        other_robots: list = (),
        grid: np.ndarray = None,
    ) -> tuple[str, list]:
        """
        S-NAMO plan: decide BYPASS / REMOVE and return full waypoints.

        Parameters
        ----------
        start        : (col, row) current robot cell.
        goal         : (col, row) target cell.
        box_cells    : list of current movable obstacle positions.
        other_robots : cells occupied by other robots (treated as blocked).
        grid         : optional override grid (e.g. after a box moved).

        Returns
        -------
        (action, waypoints)
            action    : "BYPASS", "REMOVE", or "NAVIGATE"
            waypoints : [(col, row), ...] full path (start excluded).
        """
        if grid is not None:
            self.grid = grid
            self._social_risk = SocialCostmap(grid).map

        work_grid = self.grid.copy()
        gs = self.grid_size

        # --- Find first blocking box on direct path ---
        # Add a high penalty to box cells so A* prefers paths with fewer boxes
        box_risk_map = (work_grid == 2).astype(float)
        path_direct = a_star(start, goal, work_grid, gs,
                             ignore_boxes=True,
                             risk_map=box_risk_map,
                             risk_weight=50.0)
        blocking = None
        if path_direct:
            box_set = set(map(tuple, box_cells))
            for cell in path_direct[1:]:
                if cell in box_set:
                    blocking = cell
                    break

        # No obstacle blocking → social-cost-weighted navigation
        if blocking is None:
            path = a_star(start, goal, work_grid, gs,
                          other_robots=other_robots,
                          risk_map=self._social_risk,
                          risk_weight=self.social_weight)
            waypoints = path[1:] if path else []
            return "NAVIGATE", waypoints

        # --- Decide BYPASS or REMOVE via Laplace criterion ---
        bypass_iv  = self._bypass_interval(work_grid, start, goal, blocking)
        removal_iv = self._removal_interval(work_grid, start, goal, blocking)
        decision, U_bypass, U_removal = choose_action(bypass_iv, removal_iv)

        sr_lo, sr_hi = self.belief.success_rate_interval()
        print(f"[S-NAMO] {start}->{goal} | obs={blocking} | "
              f"SR=[{sr_lo:.2f},{sr_hi:.2f}] | "
              f"U_by={U_bypass:.1f} U_re={U_removal:.1f} | -> {decision}")

        # --- BYPASS ---
        if decision == "BYPASS":
            g_bypass = work_grid.copy()
            g_bypass[blocking[1], blocking[0]] = 2
            path = a_star(start, goal, g_bypass, gs,
                          other_robots=other_robots,
                          risk_map=self._social_risk,
                          risk_weight=self.social_weight)
            if path:
                return "BYPASS", path[1:]
            # fallthrough to REMOVE if bypass has no path

        # --- REMOVE: social-aware clearing direction ---
        results = self._social_clearing_directions(
            blocking, work_grid, other_robots=other_robots
        )

        best_plan = None
        fallback_plan = None

        for clear_cell, approach_cell in results:
            # Segment 1: start → approach cell (avoid the box)
            g_obs = work_grid.copy()
            g_obs[blocking[1], blocking[0]] = 2
            seg1 = a_star(start, approach_cell, g_obs, gs,
                          other_robots=other_robots,
                          risk_map=self._social_risk,
                          risk_weight=self.social_weight)

            # Segment 2: approach → through box cell → clear cell (push)
            seg2 = [blocking, clear_cell]

            # Segment 3: clear_cell → goal (box now at clear_cell)
            g_after = work_grid.copy()
            g_after[blocking[1], blocking[0]] = 0
            g_after[clear_cell[1], clear_cell[0]] = 2
            seg3 = a_star(clear_cell, goal, g_after, gs,
                          other_robots=other_robots,
                          risk_map=self._social_risk,
                          risk_weight=self.social_weight)

            if not seg1 or not seg3:
                # Social-cost path failed — retry without social cost
                seg1_ns = a_star(start, approach_cell, g_obs, gs, other_robots=other_robots)
                seg3_ns = a_star(clear_cell, goal, g_after, gs, other_robots=other_robots)
                if seg1_ns and seg3_ns:
                    seg1, seg3 = seg1_ns, seg3_ns

            if seg1 and seg3:
                best_plan = seg1[1:] + seg2 + seg3[1:]
                break

            # Fallback: approach is reachable, but post-push path is blocked by other boxes.
            # Plan post-push path ignoring other boxes.
            if seg1:
                seg3_ignore = a_star(clear_cell, goal, g_after, gs,
                                     other_robots=other_robots,
                                     ignore_boxes=True,
                                     risk_map=self._social_risk,
                                     risk_weight=self.social_weight)
                if seg3_ignore and fallback_plan is None:
                    fallback_plan = seg1[1:] + seg2 + seg3_ignore[1:]

        if best_plan is not None:
            return "REMOVE", best_plan
        if fallback_plan is not None:
            return "REMOVE", fallback_plan

        # Last resort fallback if no candidate was fully traversable
        from core.planner import a_star as _astar
        g_free = work_grid.copy()
        g_free[blocking[1], blocking[0]] = 0  # treat as passable
        path_direct = _astar(start, blocking, g_free, gs, other_robots=other_robots, ignore_boxes=True)
        return "REMOVE", path_direct[1:] if len(path_direct) > 1 else [blocking]
