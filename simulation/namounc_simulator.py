"""
simulation/namounc_simulator.py
---------------------------------
NAMOUnc Simulator — uncertainty-aware NAMO on a 2-D grid.

This is a lightweight, PyBullet-free simulator that exercises:
  1. Interval-based BYPASS vs REMOVE decision (Laplace criterion)
  2. Beta-distribution manipulation success-rate (SR) model with
     Bayesian online updating after each push attempt
  3. Gaussian Linear Regressor trajectory cost intervals
  4. Cooperative belief sharing across a robot fleet via BeliefBroadcaster

It runs entirely as a grid-search simulation with no physics engine —
meaning it is fast, reproducible, and easy to reason about.

Paper reference: NAMOUnc (uncertainty model implemented across
    uncertainty/action_uncertainty.py, uncertainty/bypass_model.py,
    uncertainty/interval_decision.py)

Usage
-----
    # Single scenario
    python simulation/namounc_simulator.py --config configs/movable_obstacle_choke_namo.yaml

    # All configs (headless table)
    python simulation/namounc_simulator.py --config all

    # With cooperative belief sharing enabled
    python simulation/namounc_simulator.py --config configs/warehouse_3robots.yaml --belief-sharing

    # With deliberate push failure rate to show SR updating
    python simulation/namounc_simulator.py --config configs/movable_obstacle_choke_namo.yaml --push-fail-rate 0.3
"""

import sys
import argparse
import random
import math
import yaml
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.planner import a_star, find_clearing_direction
from uncertainty.action_uncertainty import ManipulationBeliefModel
from uncertainty.bypass_model import TrajectoryRegressionModel
from uncertainty.interval_decision import choose_action
from multi_robot.belief_broadcaster import BeliefBroadcaster


# ─────────────────────────────────────────────────────────────────────────────
# Grid builder from YAML
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(cfg: dict) -> np.ndarray:
    gs = cfg["world"]["grid_size"]
    grid = np.zeros((gs, gs), dtype=int)
    # Border
    grid[0, :] = 1; grid[gs-1, :] = 1
    grid[:, 0] = 1; grid[:, gs-1] = 1
    for wall in cfg.get("walls", []):
        wx, wy = wall["pos"]
        if 0 <= wx < gs and 0 <= wy < gs:
            grid[wy, wx] = 1
    for obs in cfg.get("obstacles", []):
        ox, oy = obs["pos"]
        if 0 <= ox < gs and 0 <= oy < gs:
            grid[oy, ox] = 2
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Per-robot NAMOUnc agent
# ─────────────────────────────────────────────────────────────────────────────

class NAMOUncAgent:
    """
    Single-robot uncertainty-aware NAMO agent.

    Decides BYPASS / REMOVE using cost intervals and the Laplace criterion.
    Updates its Beta-distribution SR belief after every push attempt.
    """

    def __init__(
        self,
        robot_id: int,
        start: tuple,
        goal: tuple,
        grid_size: int,
        alpha: float = 9.0,
        beta: float = 1.0,
        push_base_cost: float = 3.0,
    ):
        self.robot_id   = robot_id
        self.cell       = tuple(start)
        self.goal       = tuple(goal)
        self.grid_size  = grid_size
        self.push_base_cost = push_base_cost

        self.belief     = ManipulationBeliefModel(alpha=alpha, beta=beta,
                                                   obstacle_type="box")
        self._traj      = TrajectoryRegressionModel()
        self._plan: list = []
        self._action: str = "NAVIGATE"
        self.done       = False

        # Metrics
        self.steps_taken = 0
        self.pushes      = 0
        self.push_failures = 0
        self._stuck_ctr  = 0
        self._last_cell: Optional[tuple] = None

    # ── Interval helpers ──────────────────────────────────────────────────

    def _bypass_interval(self, grid: np.ndarray, box_pos: tuple) -> tuple:
        g = grid.copy()
        g[box_pos[1], box_pos[0]] = 2          # keep box in place
        path = a_star(self.cell, self.goal, g, self.grid_size)
        if not path:
            return (float("inf"), float("inf"))
        return self._traj.predict_interval(path, float(len(path) - 1))

    def _removal_interval(self, grid: np.ndarray, box_pos: tuple) -> tuple:
        g_obs = grid.copy()
        g_obs[box_pos[1], box_pos[0]] = 2

        path_to = a_star(self.cell, box_pos, g_obs, self.grid_size)
        if not path_to:
            return (float("inf"), float("inf"))
        reach_lo, reach_hi = self._traj.predict_interval(
            path_to, float(len(path_to) - 1))

        g_clear = grid.copy()
        g_clear[box_pos[1], box_pos[0]] = 0
        path_from = a_star(box_pos, self.goal, g_clear, self.grid_size)
        if not path_from:
            return (float("inf"), float("inf"))
        from_lo, from_hi = self._traj.predict_interval(
            path_from, float(len(path_from) - 1))

        sr_lo, sr_hi = self.belief.success_rate_interval()
        push_lo = self.push_base_cost / max(sr_hi, 0.05)
        push_hi = self.push_base_cost / max(sr_lo, 0.05)

        return (reach_lo + push_lo + from_lo,
                reach_hi + push_hi + from_hi)

    # ── Decide and plan ───────────────────────────────────────────────────

    def decide(self, grid: np.ndarray, other_robots: list = ()) -> tuple[str, list]:
        """
        Returns (action, waypoints) where action is BYPASS / REMOVE / NAVIGATE.
        """
        # Path ignoring boxes to find blocking obstacle
        ignore_path = a_star(self.cell, self.goal, grid, self.grid_size,
                             ignore_boxes=True)
        box_set = set(zip(*np.where(grid == 2)[::-1]))  # (col,row) of all boxes
        blocking: Optional[tuple] = None
        if ignore_path:
            for cell in ignore_path[1:]:
                if cell in box_set:
                    blocking = cell
                    break

        if blocking is None:
            # Free path — just navigate
            path = a_star(self.cell, self.goal, grid, self.grid_size,
                          other_robots=other_robots)
            return "NAVIGATE", (path[1:] if path else [])

        # ── Interval decision ──────────────────────────────────────────
        bypass_iv  = self._bypass_interval(grid, blocking)
        removal_iv = self._removal_interval(grid, blocking)
        decision, U_by, U_re = choose_action(bypass_iv, removal_iv)

        sr_lo, sr_hi = self.belief.success_rate_interval()
        print(f"  [R{self.robot_id}] cell={self.cell} obs={blocking} "
              f"SR=[{sr_lo:.2f},{sr_hi:.2f}] "
              f"U_by={'inf' if U_by==float('inf') else f'{U_by:.1f}'} "
              f"U_re={'inf' if U_re==float('inf') else f'{U_re:.1f}'} -> {decision}")

        if decision == "BYPASS":
            g_bypass = grid.copy()
            g_bypass[blocking[1], blocking[0]] = 2
            path = a_star(self.cell, self.goal, g_bypass, self.grid_size,
                          other_robots=other_robots)
            if path:
                return "BYPASS", path[1:]
            # No bypass path — fall through to REMOVE

        # ── Build REMOVE waypoints ─────────────────────────────────────
        result = find_clearing_direction(blocking, grid, self.grid_size,
                                         other_robots=other_robots)
        if result is None:
            # Last resort: push through
            g_free = grid.copy()
            g_free[blocking[1], blocking[0]] = 0
            path = a_star(self.cell, blocking, g_free, self.grid_size,
                          other_robots=other_robots)
            return "REMOVE", (path[1:] if len(path) > 1 else [blocking])

        clear_cell, approach_cell = result
        g_obs = grid.copy(); g_obs[blocking[1], blocking[0]] = 2
        seg1 = a_star(self.cell, approach_cell, g_obs, self.grid_size,
                      other_robots=other_robots)

        g_after = grid.copy()
        g_after[blocking[1], blocking[0]] = 0
        g_after[clear_cell[1], clear_cell[0]] = 2
        seg3 = a_star(clear_cell, self.goal, g_after, self.grid_size,
                      other_robots=other_robots)

        if not seg1:
            seg1 = a_star(self.cell, approach_cell, g_obs, self.grid_size)
        if not seg3:
            seg3 = a_star(clear_cell, self.goal, g_after, self.grid_size)

        if not seg1:
            g_free = grid.copy(); g_free[blocking[1], blocking[0]] = 0
            path = a_star(self.cell, blocking, g_free, self.grid_size)
            return "REMOVE", (path[1:] if len(path) > 1 else [blocking])

        waypoints = seg1[1:] + [blocking, clear_cell]
        if seg3:
            waypoints += seg3[1:]
        return "REMOVE", waypoints

    # ── Single step ───────────────────────────────────────────────────────

    def step(
        self,
        grid: np.ndarray,
        push_fail_rate: float = 0.0,
        other_robots: list = (),
        broadcaster: Optional[BeliefBroadcaster] = None,
    ) -> np.ndarray:
        """
        Advance the agent by one grid step.
        Returns the (possibly modified) grid after any push.
        """
        if self.done:
            return grid

        self.steps_taken += 1

        # Stuck detection
        if self._last_cell == self.cell:
            self._stuck_ctr += 1
        else:
            self._stuck_ctr = 0
        self._last_cell = self.cell

        # Force replan if stuck for 5+ steps
        if self._stuck_ctr >= 5 or not self._plan:
            self._action, self._plan = self.decide(grid, other_robots)

        if not self._plan:
            return grid

        # ── Execute next step ──────────────────────────────────────────
        next_cell = self._plan[0]

        # Check if next cell has a box (push event)
        box_here = (grid[next_cell[1], next_cell[0]] == 2)

        if box_here and self._action == "REMOVE":
            # Simulate push: stochastic success/failure
            success = (random.random() >= push_fail_rate)
            if broadcaster:
                broadcaster.broadcast_outcome(self.robot_id, success,
                                               obstacle_type="box")
            else:
                self.belief.observe(success)

            if success:
                # Move box: find where we're pushing it
                dx = next_cell[0] - self.cell[0]
                dy = next_cell[1] - self.cell[1]
                clear_to = (next_cell[0] + dx, next_cell[1] + dy)

                grid = grid.copy()
                grid[next_cell[1], next_cell[0]] = 0    # remove from old pos
                cx, cy = clear_to
                if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size and grid[cy, cx] == 0:
                    grid[cy, cx] = 2                    # place at new pos

                self.cell = next_cell
                self._plan.pop(0)
                self.pushes += 1
                # Replan immediately — grid changed
                self._plan = []
            else:
                # Push failed — update belief, replan around this box
                self.push_failures += 1
                self._plan = []           # replan next step
                self._stuck_ctr = 0
        else:
            # Normal move (or bypass step through box — treat as pass-through)
            if not box_here or self._action == "BYPASS":
                self.cell = next_cell
                self._plan.pop(0)

        # Goal check
        if self.cell == self.goal:
            self.done = True

        return grid

    def sr_summary(self) -> str:
        lo, hi = self.belief.success_rate_interval()
        return (f"R{self.robot_id}: SR={self.belief.mean():.3f} "
                f"[{lo:.3f},{hi:.3f}] "
                f"pushes={self.pushes} failures={self.push_failures} "
                f"steps={self.steps_taken}")


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    cfg: dict,
    grid: np.ndarray,
    push_fail_rate: float = 0.0,
    belief_sharing: bool = False,
    max_steps: int = 500,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    gs        = cfg["world"]["grid_size"]
    robots_cfg = cfg.get("robots", [])

    agents = [
        NAMOUncAgent(
            robot_id = r["id"],
            start    = tuple(r["start"]),
            goal     = tuple(r["goal"]),
            grid_size = gs,
        )
        for r in robots_cfg
    ]

    if not agents:
        return {"success": False, "steps": 0, "pushes": 0, "failures": 0}

    broadcaster: Optional[BeliefBroadcaster] = None
    if belief_sharing:
        broadcaster = BeliefBroadcaster([a.robot_id for a in agents])

    current_grid = grid.copy()
    name = cfg.get("meta", {}).get("name", "episode")
    if verbose:
        print(f"\n[NAMOUnc] '{name}' | robots={len(agents)} | "
              f"push_fail={push_fail_rate:.0%} | shared={belief_sharing}")

    for step in range(max_steps):
        if all(a.done for a in agents):
            break

        # ── Sync belief from broadcaster into each agent ──────────────
        if belief_sharing and broadcaster:
            for a in agents:
                a.belief = broadcaster.belief_models[a.robot_id]

        # ── Step each agent ───────────────────────────────────────────
        other_cells = [a.cell for a in agents]
        for a in agents:
            if a.done:
                continue
            others = [c for c2, c in zip(agents, other_cells) if c2 is not a]
            current_grid = a.step(
                current_grid,
                push_fail_rate = push_fail_rate,
                other_robots   = others,
                broadcaster    = broadcaster,
            )
            # Update list with latest position
            for idx, ag in enumerate(agents):
                other_cells[idx] = ag.cell

    success    = all(a.done for a in agents)
    total_steps = max(a.steps_taken for a in agents) if agents else 0
    total_pushes = sum(a.pushes for a in agents)
    total_fail   = sum(a.push_failures for a in agents)

    if verbose:
        status = "✅" if success else "❌"
        print(f"[NAMOUnc] {status}  steps={total_steps}  pushes={total_pushes}  "
              f"failures={total_fail}  success={success}")
        for a in agents:
            print(f"          {a.sr_summary()}")

    if belief_sharing and broadcaster and verbose:
        print(f"          {broadcaster.summary()}")

    return {
        "success":  success,
        "steps":    total_steps,
        "pushes":   total_pushes,
        "failures": total_fail,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

ALL_CONFIGS = [
    "configs/movable_obstacle_choke_namo.yaml",
    "configs/warehouse_small.yaml",
    "configs/warehouse_large.yaml",
    "configs/warehouse_3robots.yaml",
    "configs/single_corridor_yielding.yaml",
    "configs/symmetric_bottleneck_deadlock.yaml",
    "configs/narrow_doorway_congestion.yaml",
    "configs/cross_intersection_coordination.yaml",
]


def main():
    ap = argparse.ArgumentParser(description="NAMOUnc Grid Simulator")
    ap.add_argument("--config", default="configs/movable_obstacle_choke_namo.yaml",
                    help="YAML config path or 'all'")
    ap.add_argument("--push-fail-rate", type=float, default=0.0,
                    help="Probability each push attempt fails (0.0–1.0)")
    ap.add_argument("--belief-sharing", action="store_true",
                    help="Enable cooperative Beta-belief sharing across robots")
    ap.add_argument("--episodes", type=int, default=1,
                    help="Number of episodes to average over")
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    config_list = ALL_CONFIGS if args.config == "all" else [args.config]

    rows = []
    for cfg_path in config_list:
        full_path = Path(cfg_path)
        if not full_path.exists():
            full_path = Path(__file__).resolve().parent.parent / cfg_path
        if not full_path.exists():
            print(f"[SKIP] not found: {cfg_path}")
            continue

        with open(full_path) as f:
            cfg = yaml.safe_load(f)

        if not cfg.get("robots"):
            print(f"[SKIP] {cfg_path} — no robots")
            continue

        grid = build_grid(cfg)
        agg = {"success": 0, "steps": 0, "pushes": 0, "failures": 0}

        for ep in range(args.episodes):
            r = run_episode(
                cfg,
                grid.copy(),
                push_fail_rate  = args.push_fail_rate,
                belief_sharing  = args.belief_sharing,
                max_steps       = args.max_steps,
                seed            = ep,
                verbose         = (args.episodes == 1),
            )
            agg["success"]  += int(r["success"])
            agg["steps"]    += r["steps"]
            agg["pushes"]   += r["pushes"]
            agg["failures"] += r["failures"]

        n = args.episodes
        rows.append({
            "name":     cfg_path.split("/")[-1].replace(".yaml", ""),
            "sr":       f"{agg['success']/n*100:.0f}%",
            "steps":    f"{agg['steps']/n:.0f}",
            "pushes":   f"{agg['pushes']/n:.1f}",
            "failures": f"{agg['failures']/n:.1f}",
        })

    if len(rows) > 1:
        print(f"\n{'='*74}")
        print(f"{'Scenario':<40} {'SR':>4} {'Steps':>7} {'Pushes':>7} {'PushFail':>9}")
        print(f"{'='*74}")
        for row in rows:
            print(f"{row['name']:<40} {row['sr']:>4} {row['steps']:>7} "
                  f"{row['pushes']:>7} {row['failures']:>9}")
        print(f"{'='*74}\n")


if __name__ == "__main__":
    main()
