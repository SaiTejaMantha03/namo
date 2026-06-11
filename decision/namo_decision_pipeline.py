"""
decision/namo_decision_pipeline.py
------------------------------------
NAMOUnc planner — uncertainty-aware BYPASS vs REMOVE decision making.

Upgrades the original deterministic NAMOPlanner to NAMOUnc (Paper 1) by:
  1. Replacing push_cost=3 scalar with a Beta-distribution SR model
  2. Replacing scalar bypass cost with a Gaussian trajectory cost interval
  3. Replacing scalar comparison with the Laplace criterion (U = (max+min)/2)
  4. Adding observe(success) to update the belief model after each push
  5. Returning actual waypoints (col, row) list the simulator can follow
  6. Using core/planner.a_star instead of its own inline copy

Public API (backward compatible with existing callers):
    planner = NAMOPlanner(grid_size=15)
    decision, cost = planner.make_decision(grid, start, goal, obstacle_pos)
    # decision: "BYPASS" or "REMOVE"
    # cost: scalar midpoint U for reporting

New API (full NAMOUnc):
    action, waypoints = planner.plan(grid, start, goal, box_cells)
    planner.observe(success=True)  # after a real push attempt
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.planner import a_star, find_clearing_direction
from uncertainty.action_uncertainty import ManipulationBeliefModel
from uncertainty.bypass_model import TrajectoryRegressionModel
from uncertainty.interval_decision import choose_action


class NAMOPlanner:
    """
    NAMOUnc-grade planner: uncertainty-aware BYPASS vs REMOVE decisions.

    Parameters
    ----------
    grid_size : int
        Number of cells per side for the occupancy grid.
    alpha, beta : float
        Initial Beta prior for manipulation success rate.
        Default Beta(9,1) encodes a 90% SR prior (NAMOUnc Table I).
    """

    def __init__(self, grid_size: int = 10, alpha: float = 9.0, beta: float = 1.0):
        self.grid_size = grid_size
        self.belief = ManipulationBeliefModel(alpha=alpha, beta=beta)
        self._trajectory_model = TrajectoryRegressionModel()

    # ------------------------------------------------------------------
    # Belief update (call after each real push attempt in simulation)
    # ------------------------------------------------------------------

    def observe(self, success: bool) -> None:
        """Update the Beta SR model after a manipulation attempt."""
        self.belief.observe(success)

    # ------------------------------------------------------------------
    # Internal cost computation
    # ------------------------------------------------------------------

    def _bypass_interval(
        self, grid: np.ndarray, start: tuple, goal: tuple, obstacle_pos: tuple
    ) -> tuple[float, float]:
        """
        Compute bypass cost interval [lo, hi] using Gaussian trajectory model.
        Obstacle is kept in place (treated as wall).
        """
        g = grid.copy()
        g[obstacle_pos[1], obstacle_pos[0]] = 2  # ensure obstacle is marked
        path = a_star(start, goal, g, self.grid_size)
        if not path:
            return (float("inf"), float("inf"))
        base_cost = float(len(path) - 1)
        return self._trajectory_model.predict_interval(path, base_cost)

    def _removal_interval(
        self, grid: np.ndarray, start: tuple, goal: tuple, obstacle_pos: tuple
    ) -> tuple[float, float]:
        """
        Compute removal cost interval [lo, hi] accounting for SR uncertainty.

        C_re = C_reach + C_push_interval + C_from_obs_to_goal
        C_push_interval = [base_push/S_max, base_push/S_min]
                         using the current Beta SR model.
        """
        g_obs = grid.copy()
        g_obs[obstacle_pos[1], obstacle_pos[0]] = 2

        # Path to reach the obstacle
        path_to_obs = a_star(start, obstacle_pos, g_obs, self.grid_size)
        if not path_to_obs:
            return (float("inf"), float("inf"))
        reach_cost = float(len(path_to_obs) - 1)
        reach_lo, reach_hi = self._trajectory_model.predict_interval(
            path_to_obs, reach_cost
        )

        # Path from obstacle to goal (obstacle cleared)
        g_clear = grid.copy()
        g_clear[obstacle_pos[1], obstacle_pos[0]] = 0
        path_from_obs = a_star(obstacle_pos, goal, g_clear, self.grid_size)
        if not path_from_obs:
            return (float("inf"), float("inf"))
        from_cost = float(len(path_from_obs) - 1)
        from_lo, from_hi = self._trajectory_model.predict_interval(
            path_from_obs, from_cost
        )

        # Push cost interval based on SR uncertainty
        # Cost to push = base_cost / SR (lower SR → higher expected push cost)
        base_push = 3.0
        sr_lo, sr_hi = self.belief.success_rate_interval()
        push_lo = base_push / max(sr_hi, 0.05)
        push_hi = base_push / max(sr_lo, 0.05)

        return (
            reach_lo + push_lo + from_lo,
            reach_hi + push_hi + from_hi,
        )

    # ------------------------------------------------------------------
    # Core decision (backward compatible)
    # ------------------------------------------------------------------

    def make_decision(
        self,
        grid: np.ndarray,
        start: tuple,
        goal: tuple,
        obstacle_pos: tuple,
        push_cost: float = 3.0,   # kept for API compat, overridden by Beta model
    ) -> tuple[str, float]:
        """
        NAMOUnc BYPASS vs REMOVE decision using Laplace criterion.

        Returns
        -------
        (decision, U_selected)
            decision : "BYPASS" or "REMOVE"
            U_selected : Laplace midpoint of the chosen option
        """
        self.grid_size = grid.shape[0]

        bypass_iv = self._bypass_interval(grid, start, goal, obstacle_pos)
        removal_iv = self._removal_interval(grid, start, goal, obstacle_pos)

        decision, U_bypass, U_removal = choose_action(bypass_iv, removal_iv)

        # SR model stats for logging
        sr_lo, sr_hi = self.belief.success_rate_interval()

        print(f"\nNAMOUnc Decision | start={start} goal={goal} obs={obstacle_pos}")
        print(f"  SR model : mean={self.belief.mean():.3f}  "
              f"interval=[{sr_lo:.3f}, {sr_hi:.3f}]  n_obs={self.belief._n_obs}")
        print(f"  Bypass   : [{bypass_iv[0]:.2f}, {bypass_iv[1]:.2f}]  "
              f"U={U_bypass:.2f}")
        print(f"  Removal  : [{removal_iv[0]:.2f}, {removal_iv[1]:.2f}]  "
              f"U={U_removal:.2f}")
        print(f"  >>> DECISION: {decision} <<<")

        U_selected = U_bypass if decision == "BYPASS" else U_removal
        return decision, U_selected

    # ------------------------------------------------------------------
    # Full planning API — returns waypoints for the simulator
    # ------------------------------------------------------------------

    def plan(
        self,
        grid: np.ndarray,
        start: tuple,
        goal: tuple,
        box_cells: list[tuple],
    ) -> tuple[str, list[tuple]]:
        """
        Full NAMOUnc planning: decide and return waypoints.

        Parameters
        ----------
        grid      : occupancy grid (0=free, 1=wall, 2=box).
        start     : robot current cell (col, row).
        goal      : robot target cell (col, row).
        box_cells : list of all current movable obstacle positions.

        Returns
        -------
        (action, waypoints)
            action    : "BYPASS" or "REMOVE"
            waypoints : [(col, row), ...] full path for the simulator to follow.
                        For REMOVE this includes the approach, push-through,
                        and continuation to goal.
        """
        self.grid_size = grid.shape[0]

        # Plan ignoring boxes to find direct path length
        path_direct = a_star(
            start, goal, grid, self.grid_size, ignore_boxes=True
        )

        # Find first blocking box on the direct path
        blocking = None
        if path_direct:
            box_set = set(map(tuple, box_cells))
            for cell in path_direct[1:]:
                if cell in box_set:
                    blocking = cell
                    break

        # No box blocking → plain navigation
        if blocking is None:
            path = a_star(start, goal, grid, self.grid_size)
            return "NAVIGATE", path if path else [start]

        # Decide BYPASS or REMOVE
        decision, _ = self.make_decision(grid, start, goal, blocking)

        if decision == "BYPASS":
            # Route around the box
            g_bypass = grid.copy()
            g_bypass[blocking[1], blocking[0]] = 2  # ensure blocked
            path = a_star(start, goal, g_bypass, self.grid_size)
            if path:
                return "BYPASS", path

        # REMOVE — build waypoint sequence: approach → push-through → goal
        result = find_clearing_direction(blocking, grid, self.grid_size)
        if result is None:
            # No valid clearing direction → forced bypass
            g_bypass = grid.copy()
            g_bypass[blocking[1], blocking[0]] = 2
            path = a_star(start, goal, g_bypass, self.grid_size)
            return "BYPASS", path if path else [start]

        clear_cell, approach_cell = result

        # Segment 1: start → approach cell
        g_obs = grid.copy()
        g_obs[blocking[1], blocking[0]] = 2
        seg1 = a_star(start, approach_cell, g_obs, self.grid_size)

        # Segment 2: approach → push through obstacle cell → clear cell
        # (robot physically walks through the box cell, pushing the box)
        seg2 = [blocking, clear_cell]

        # Segment 3: clear cell → goal (box now at clear_cell)
        g_after = grid.copy()
        g_after[blocking[1], blocking[0]] = 0
        g_after[clear_cell[1], clear_cell[0]] = 2
        seg3 = a_star(clear_cell, goal, g_after, self.grid_size)

        if not seg1 or not seg3:
            # Fall back to bypass if segments fail
            g_bypass = grid.copy()
            g_bypass[blocking[1], blocking[0]] = 2
            path = a_star(start, goal, g_bypass, self.grid_size)
            return "BYPASS", path if path else [start]

        # Stitch segments together (drop duplicate junction points)
        waypoints = seg1 + seg2 + (seg3[1:] if seg3 else [])
        return "REMOVE", waypoints


# ------------------------------------------------------------------
# Demo — runs standalone to verify on classic scenarios
# ------------------------------------------------------------------
def run_pipeline_demo():
    planner = NAMOPlanner(grid_size=10)

    # --- Scenario A: Only path blocked → REMOVE ---
    grid_a = np.zeros((10, 10), dtype=int)
    grid_a[0, :] = 1; grid_a[9, :] = 1
    grid_a[:, 0] = 1; grid_a[:, 9] = 1
    for col in range(1, 9):
        if col != 4:
            grid_a[5, col] = 1

    start, goal, obstacle = (4, 2), (4, 7), (4, 5)
    print("=" * 55)
    print("SCENARIO A: Only corridor blocked → expect REMOVE")
    print("=" * 55)
    action, waypoints = planner.plan(grid_a, start, goal, [obstacle])
    print(f"Action: {action}  |  Waypoints ({len(waypoints)}): {waypoints}")

    # Simulate push success, update belief
    planner.observe(success=True)
    print(f"After observe(True): SR mean={planner.belief.mean():.3f}")

    # --- Scenario B: Easy detour → BYPASS ---
    grid_b = np.zeros((10, 10), dtype=int)
    grid_b[0, :] = 1; grid_b[9, :] = 1
    grid_b[:, 0] = 1; grid_b[:, 9] = 1
    for col in range(1, 9):
        if col != 4 and col != 6:
            grid_b[5, col] = 1

    obstacle_b = (4, 5)
    print("\n" + "=" * 55)
    print("SCENARIO B: Easy detour available → expect BYPASS")
    print("=" * 55)
    action_b, waypoints_b = planner.plan(grid_b, start, goal, [obstacle_b])
    print(f"Action: {action_b}  |  Waypoints ({len(waypoints_b)}): {waypoints_b}")


if __name__ == "__main__":
    run_pipeline_demo()
