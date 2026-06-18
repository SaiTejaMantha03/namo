"""
simulation/snamo_simulator.py
-------------------------------
S-NAMO PyBullet simulator — fully wired MR-NAMO architecture.

Changes from previous version
------------------------------
1. Removed broken intersection token-ring that caused 0% pass rate on all
   no-box corridor/intersection configs (single_corridor_yielding etc.).

2. Wired the complete multi_robot/ MR-NAMO stack:
     RobotCoordinator  — Algorithm 1 (plan → detect → resolve → execute)
     ConflictDetector  — C1–C6 conflict types from Paper 5
     DeadlockResolver  — Repulsive / Social / SR-Width / SR-Social strategies
     BeliefBroadcaster — cooperative SR-belief sharing across fleet

3. For box-map configs, SNAMOPlanner still makes per-robot BYPASS/REMOVE
   decisions; the coordinator handles multi-robot spatial coordination.

4. Added --dr-strategy CLI argument.
   Default: "sr_social" (our novel Phase 2B contribution).

Usage:
    python simulation/snamo_simulator.py --config configs/single_corridor_yielding.yaml --gui
    python simulation/snamo_simulator.py --config all
    python simulation/snamo_simulator.py --config all --dr-strategy repulsive
"""

import sys
import math
import time
import argparse
import yaml
import numpy as np
from pathlib import Path

import pybullet as p
import pybullet_data

sys.path.append(str(Path(__file__).resolve().parent.parent))

from decision.snamo_planner import SNAMOPlanner
from multi_robot.coordinator import RobotCoordinator, RobotState
from multi_robot.deadlock_resolution import DeadlockResolver, _find_evasion_target
from multi_robot.belief_broadcaster import BeliefBroadcaster
from social.social_costmap import SocialCostmap
from core.planner import a_star



# ─────────────────────────────────────────────────────────────────────────────
# PyBullet helpers
# ─────────────────────────────────────────────────────────────────────────────

def cell_center(cx: int, cy: int, cs: float = 1.0):
    return ((cx + 0.5) * cs, (cy + 0.5) * cs)


def world_to_cell(xy, cs: float = 1.0):
    return (int(xy[0] // cs), int(xy[1] // cs))


def create_wall(pos, half_ext):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                              rgbaColor=[0.22, 0.22, 0.22, 1.0])
    return p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)


def create_box(pos, half_ext):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                              rgbaColor=[0.95, 0.60, 0.10, 1.0])
    bid = p.createMultiBody(baseMass=1.2,
                            baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=vis,
                            basePosition=pos)
    p.changeDynamics(bid, -1, lateralFriction=0.9, linearDamping=0.6,
                     angularDamping=0.9)
    return bid


def create_robot(pos, color):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.27, height=0.30)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.27, length=0.30,
                              rgbaColor=color)
    rid = p.createMultiBody(baseMass=2.0,
                            baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=vis,
                            basePosition=pos)
    p.changeDynamics(rid, -1, lateralFriction=1.0,
                     linearDamping=0.5, angularDamping=0.9)
    return rid


def create_goal_marker(pos, color):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.30, length=0.015,
                              rgbaColor=color)
    return p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)


def drive_robot(rid: int, target_xy: tuple[float, float], speed: float = 6.0) -> float:
    """Drive robot via kinematic velocity towards target."""
    pos, _ = p.getBasePositionAndOrientation(rid)
    dx = target_xy[0] - pos[0]
    dy = target_xy[1] - pos[1]
    dist = math.hypot(dx, dy)
    if dist < 0.12:
        p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
        return dist
    s = speed / max(dist, 1e-6)
    yaw = math.atan2(dy, dx)
    p.resetBasePositionAndOrientation(
        rid, [pos[0], pos[1], 0.15],
        p.getQuaternionFromEuler([0, 0, yaw]))
    p.resetBaseVelocity(rid, [dx * s, dy * s, 0], [0, 0, 0])
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Grid builder
# ─────────────────────────────────────────────────────────────────────────────

def build_clean_grid(cfg: dict) -> np.ndarray:
    """
    Build occupancy grid purely from YAML definitions.
    0=free, 1=wall, 2=movable box.
    Robot positions are NOT stamped — goals/starts remain free cells.
    """
    gs = cfg["world"]["grid_size"]
    grid = np.zeros((gs, gs), dtype=int)

    # Border walls (standard for all envs)
    for i in range(gs):
        grid[0, i] = 1
        grid[gs - 1, i] = 1
        grid[i, 0] = 1
        grid[i, gs - 1] = 1

    # YAML-defined static walls
    for wall in cfg.get("walls", []):
        wx, wy = wall["pos"]
        if 0 <= wx < gs and 0 <= wy < gs:
            grid[wy, wx] = 1

    # YAML-defined movable obstacles (boxes)
    for obs in cfg.get("obstacles", []):
        ox, oy = obs["pos"]
        if 0 <= ox < gs and 0 <= oy < gs:
            grid[oy, ox] = 2

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Robot colors
# ─────────────────────────────────────────────────────────────────────────────

COLORS = [
    [0.20, 0.55, 0.95, 1.0],   # blue
    [0.95, 0.25, 0.25, 1.0],   # red
    [0.15, 0.80, 0.30, 1.0],   # green
    [0.85, 0.20, 0.85, 1.0],   # purple
    [0.20, 0.85, 0.85, 1.0],   # cyan
]


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

class SNAMOSimulator:
    def __init__(self, config_path, gui=False, taboo_zones=None, dr_strategy="sr_social"):
        self.config_path = config_path
        self.gui = gui
        self.taboo_zones = taboo_zones or []
        self.dr_strategy = dr_strategy
        
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
            
        world = self.cfg["world"]
        self.gs = world["grid_size"]
        self.cs = world.get("cell_size", 1.0)
        self.max_steps = world.get("sim_steps", 1500)
        self.robots_cfg = self.cfg.get("robots", [])
        
        self.base_grid = build_clean_grid(self.cfg)
        self.physics_client = None
        self.step_count = 0
        self.robot_robot_collisions = 0
        self.success = False
        self.collision_pairs_seen = set()
        self.name = self.cfg.get("meta", {}).get("name", self.config_path)

    def reset(self):
        if not self.robots_cfg:
            print(f"[SKIP] {self.config_path} — no robots defined.")
            return

        if self.physics_client is not None:
            try:
                p.disconnect()
            except:
                pass
                
        mode = p.GUI if self.gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.loadURDF("plane.urdf")

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=self.gs * self.cs * 1.35,
                cameraYaw=0, cameraPitch=-70,
                cameraTargetPosition=[self.gs * self.cs * 0.5, self.gs * self.cs * 0.5, 0.0])

        wh = 0.8
        self.box_ids = []
        self.box_init_xy = {}

        for row in range(self.gs):
            for col in range(self.gs):
                x, y = cell_center(col, row, self.cs)
                v = self.base_grid[row, col]
                if v == 1:
                    create_wall([x, y, wh * 0.5], [0.5 * self.cs, 0.5 * self.cs, wh * 0.5])
                elif v == 2:
                    bid = create_box([x, y, 0.18], [0.34 * self.cs, 0.34 * self.cs, 0.18])
                    self.box_ids.append(bid)
                    raw, _ = p.getBasePositionAndOrientation(bid)
                    self.box_init_xy[bid] = raw[:2]

        self.robot_ids = []
        self.robot_goals = {}
        self.snamo_planners = {}

        for i, rcfg in enumerate(self.robots_cfg):
            sc = tuple(rcfg["start"])
            gc = tuple(rcfg["goal"])
            sx, sy = cell_center(sc[0], sc[1], self.cs)
            rid = create_robot([sx, sy, 0.15], COLORS[i % len(COLORS)])
            self.robot_ids.append(rid)
            self.robot_goals[rid] = gc
            self.snamo_planners[rid] = SNAMOPlanner(
                grid=self.base_grid.copy(),
                grid_size=self.gs,
                taboo_zones=self.taboo_zones,
            )
            gx, gy = cell_center(gc[0], gc[1], self.cs)
            c = COLORS[i % len(COLORS)]
            create_goal_marker([gx, gy, 0.01], [c[0], c[1], c[2], 0.30])

        self.social_map = SocialCostmap(self.base_grid)
        self.broadcaster = BeliefBroadcaster(robot_ids=self.robot_ids)
        self.resolver = DeadlockResolver(grid=self.base_grid, grid_size=self.gs)
        self.coordinator = RobotCoordinator(
            grid=self.base_grid,
            grid_size=self.gs,
            dr_strategy=self.dr_strategy,
            resolver=self.resolver,
            broadcaster=self.broadcaster,
            social_map=self.social_map,
            adaptive_h=True,
        )

        self.coord_states = {}
        for i, rid in enumerate(self.robot_ids):
            rcfg = self.robots_cfg[i]
            self.coord_states[rid] = RobotState(
                robot_id=rid,
                cell=tuple(rcfg["start"]),
                goal=tuple(rcfg["goal"]),
            )

        self.drive_target = {
            rid: cell_center(*tuple(self.robots_cfg[i]["start"]), self.cs)
            for i, rid in enumerate(self.robot_ids)
        }
        self.push_obs_pending = {r: False for r in self.robot_ids}
        self.last_snamo_action = {r: "NAVIGATE" for r in self.robot_ids}
        
        self.step_count = 0
        self.robot_robot_collisions = 0
        self.success = False
        self.collision_pairs_seen.clear()
        self.stall_counter = 0
        self.last_robot_positions = {}
        self.stalled = False

    def step(self, rl_actions=None):
        if getattr(self, "stalled", False):
            return

        robot_cells = {}
        any_moved = False
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            robot_cells[rid] = world_to_cell(pos[:2], self.cs)
            last_pos = self.last_robot_positions.get(rid, pos)
            if math.hypot(pos[0] - last_pos[0], pos[1] - last_pos[1]) > 0.02:
                any_moved = True
            self.last_robot_positions[rid] = pos

        if not any_moved and len(self.robot_ids) > 0:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        if self.stall_counter > 1500:
            self.stalled = True
            return

        box_cells_live = []
        current_grid = self.base_grid.copy()
        box_states_coord = {}

        for bid in self.box_ids:
            pos, _ = p.getBasePositionAndOrientation(bid)
            bc = world_to_cell(pos[:2], self.cs)
            box_cells_live.append(bc)
            box_states_coord[bid] = bc
            bx, by = bc
            if 0 <= bx < self.gs and 0 <= by < self.gs:
                current_grid[by, bx] = 2

        self.coordinator.grid = current_grid.copy()
        self.resolver.grid = current_grid.copy()

        all_done = True
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            gxc, gyc = self.robot_goals[rid]
            gxw, gyw = cell_center(gxc, gyc, self.cs)
            if math.hypot(pos[0] - gxw, pos[1] - gyw) > 0.5:
                all_done = False
                break
        if all_done:
            self.success = True
            return

        for rid in self.robot_ids:
            self.coord_states[rid].cell = robot_cells[rid]
            if self.coord_states[rid].status != "DONE" and robot_cells[rid] == self.robot_goals[rid]:
                self.coord_states[rid].status = "DONE"

        any_arrived = False
        for rid in self.robot_ids:
            if self.coord_states[rid].status not in ("DONE", "WAITING", "POCKET_WAITING") and rid in self.drive_target:
                pos, _ = p.getBasePositionAndOrientation(rid)
                tx, ty = self.drive_target[rid]
                if math.hypot(pos[0] - tx, pos[1] - ty) <= 0.15:
                    any_arrived = True
                    break

        if any_arrived or self.step_count % 5 == 0:
            next_cells = self.coordinator.step(self.coord_states, box_states_coord)

            # Map RL actions directly to coordinator status and next_cells if provided
            if rl_actions is not None:
                for rid, act in rl_actions.items():
                    if act == 0: # NAVIGATE
                        self.coord_states[rid].status = "NAVIGATING"
                        self.coord_states[rid].evasion_target = None
                    elif act == 1: # PUSH_BOX
                        self.coord_states[rid].status = "NAVIGATING"
                        self.coord_states[rid].evasion_target = None
                        self.last_snamo_action[rid] = "REMOVE"
                    elif act == 2: # YIELD
                        self.coord_states[rid].status = "EVADING"
                        if self.coord_states[rid].evasion_target is None:
                            other_cells = [robot_cells[r] for r in self.robot_ids if r != rid]
                            target = _find_evasion_target(
                                self.base_grid, self.gs, robot_cells[rid], other_cells
                            )
                            self.coord_states[rid].evasion_target = target
                            if target:
                                evade_plan = a_star(
                                    robot_cells[rid], target,
                                    self.base_grid, self.gs,
                                    other_robots=other_cells
                                )
                                self.coord_states[rid].plan = evade_plan if evade_plan else [robot_cells[rid]]
                    elif act == 3: # WAIT
                        self.coord_states[rid].status = "WAITING"
                        self.coord_states[rid].evasion_target = None
                        self.coord_states[rid].wait_ticks = 999999

            for rid in self.robot_ids:
                if self.coord_states[rid].status in ("DONE", "WAITING", "POCKET_WAITING"):
                    continue

                curr_cell = robot_cells[rid]
                goal_cell = self.robot_goals[rid]
                next_cell = next_cells.get(rid, curr_cell)

                if box_cells_live and self.coord_states[rid].status != "EVADING":
                    other_cells = [robot_cells[r] for r in self.robot_ids if r != rid]
                    plan_grid = current_grid.copy()
                    for oc in other_cells:
                        ox, oy = oc
                        if (ox, oy) != goal_cell:
                            if 0 <= ox < self.gs and 0 <= oy < self.gs and plan_grid[oy, ox] == 0:
                                plan_grid[oy, ox] = 1

                    self.snamo_planners[rid].update_grid(plan_grid)
                    action, waypoints = self.snamo_planners[rid].plan(
                        start=curr_cell, goal=goal_cell,
                        box_cells=box_cells_live, other_robots=other_cells, grid=plan_grid,
                    )
                    
                    if rl_actions is None:
                        self.last_snamo_action[rid] = action
                    action = self.last_snamo_action[rid]

                    if waypoints:
                        next_cell = waypoints[0]
                        # EVADING robots can push through WAITING robots
                        if self.coord_states[rid].status == "EVADING":
                            nxt_occupied = any(
                                robot_cells[r] == next_cell
                                for r in self.robot_ids
                                if r != rid and self.coord_states[r].status not in ("WAITING", "POCKET_WAITING")
                            )
                        else:
                            nxt_occupied = any(robot_cells[r] == next_cell for r in self.robot_ids if r != rid)
                        if nxt_occupied:
                            p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                            continue

                        if action == "REMOVE":
                            bx, by = next_cell
                            if 0 <= bx < self.gs and 0 <= by < self.gs:
                                if current_grid[by, bx] == 2:
                                    self.push_obs_pending[rid] = True
                                elif self.push_obs_pending[rid]:
                                    self.snamo_planners[rid].observe(success=True)
                                    self.broadcaster.broadcast_outcome(rid, success=True)
                                    self.push_obs_pending[rid] = False

                self.drive_target[rid] = cell_center(next_cell[0], next_cell[1], self.cs)

        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            gxc, gyc = self.robot_goals[rid]
            gxw, gyw = cell_center(gxc, gyc, self.cs)
            if math.hypot(pos[0] - gxw, pos[1] - gyw) <= 0.5:
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                continue

            if self.coord_states[rid].status in ("WAITING", "POCKET_WAITING"):
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                continue

            drive_robot(rid, self.drive_target[rid])

        if self.gui:
            time.sleep(0.001)

        p.stepSimulation()

        for i in range(len(self.robot_ids)):
            for j in range(i + 1, len(self.robot_ids)):
                key = ("rr", self.robot_ids[i], self.robot_ids[j])
                if key not in self.collision_pairs_seen and p.getContactPoints(self.robot_ids[i], self.robot_ids[j]):
                    self.robot_robot_collisions += 1
                    self.collision_pairs_seen.add(key)

        self.step_count += 1

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect()
            except:
                pass
            self.physics_client = None

def run_simulation(config_path: str, gui: bool = False, taboo_zones: list = None, dr_strategy: str = "sr_social") -> dict:
    sim = SNAMOSimulator(config_path, gui=gui, taboo_zones=taboo_zones, dr_strategy=dr_strategy)
    sim.reset()
    
    if not sim.robots_cfg:
        return {"success": False, "steps": 0, "pushes": 0, "collisions": 0, "deadlocks": 0, "avg_sr_width": 0.0}
        
    print(f"\n[SNAMOSim] '{sim.name}' | robots={len(sim.robot_ids)} boxes={len(sim.box_ids)} "
          f"max_steps={sim.max_steps} dr={dr_strategy}")
          
    while sim.step_count < sim.max_steps and not sim.success and not getattr(sim, "stalled", False):
        sim.step()
        
    push_count = 0
    for bid in sim.box_ids:
        pos, _ = p.getBasePositionAndOrientation(bid)
        ix, iy = sim.box_init_xy[bid]
        if math.hypot(pos[0] - ix, pos[1] - iy) > 0.4:
            push_count += 1

    # Physically verify success
    real_success = True
    for rid in sim.robot_ids:
        pos, _ = p.getBasePositionAndOrientation(rid)
        gxc, gyc = sim.robot_goals[rid]
        gxw, gyw = cell_center(gxc, gyc, sim.cs)
        if math.hypot(pos[0] - gxw, pos[1] - gyw) > 0.6:
            real_success = False
            break

    status_str = "SUCCESS" if real_success else ("STALLED" if getattr(sim, "stalled", False) else "FAILED")

    print(f"[SNAMOSim] Steps={sim.step_count}  Status={status_str}  "
          f"Pushes={push_count}  Collisions={sim.robot_robot_collisions}")
    sim.close()

    avg_w = float(np.mean(sim.coordinator.yield_sr_widths)) if sim.coordinator.yield_sr_widths else 0.0

    return {
        "success":    real_success,
        "stalled":    getattr(sim, "stalled", False),
        "steps":      sim.step_count,
        "pushes":     push_count,
        "collisions": sim.robot_robot_collisions,
        "deadlocks":  sim.coordinator.deadlock_count,
        "avg_sr_width": avg_w,
    }

ALL_CONFIGS = [
    "configs/movable_obstacle_choke_namo.yaml",
    "configs/warehouse_small.yaml",
    "configs/warehouse_large.yaml",
    "configs/warehouse_3robots.yaml",
    "configs/single_corridor_yielding.yaml",
    "configs/symmetric_bottleneck_deadlock.yaml",
    "configs/narrow_doorway_congestion.yaml",
    "configs/cross_intersection_coordination.yaml",
    "configs/custom_reconstructed_map_robots.yaml",
    "configs/symmetric_bottleneck_4robots.yaml",
]


def main():
    ap = argparse.ArgumentParser(description="S-NAMO Simulator")
    ap.add_argument("--config", default="configs/movable_obstacle_choke_namo.yaml",
                    help="YAML config path or 'all'")
    ap.add_argument("--gui",   action="store_true", help="Open PyBullet GUI")
    ap.add_argument("--dr-strategy", default="sr_social",
                    choices=["repulsive", "social", "sr_width", "sr_social"],
                    help="Deadlock resolution strategy (default: sr_social)")
    args = ap.parse_args()

    if args.config == "all":
        strategies = ["repulsive", "social", "sr_social"]
        all_results = {}

        for strat in strategies:
            print(f"\n{'#'*60}\n  STRATEGY: {strat.upper()}\n{'#'*60}")
            strat_results = []
            for cfg_path in ALL_CONFIGS:
                print(f"\n{'='*58}\nRunning: {cfg_path}\n{'='*58}")
                r = run_simulation(cfg_path, gui=False, dr_strategy=strat)
                strat_results.append((cfg_path.split("/")[-1], r))
            all_results[strat] = strat_results

        cfg_names = [name for name, _ in all_results[strategies[0]]]

        print(f"\n\n{'='*95}")
        print(f"  COMPARISON TABLE — DR Strategy Comparison")
        print(f"{'='*95}")
        header = f"{'Config':<38}"
        for strat in strategies:
            header += f" {strat.upper():^24}"
        print(header)
        sub = f"{'':38}"
        for _ in strategies:
            sub += f" {'OK':<5} {'Steps':<7} {'Push':<5} {'Col':<5}"
        print(sub)
        print(f"{'-'*95}")

        for i, name in enumerate(cfg_names):
            row = f"{name:<38}"
            for strat in strategies:
                r = all_results[strat][i][1]
                ok = "PASS" if r["success"] else "FAIL"
                row += f" {ok:<5} {r['steps']:<7} {r['pushes']:<5} {r['collisions']:<5}"
            print(row)

        print(f"{'-'*95}")
        summary = f"{'PASS RATE':<38}"
        for strat in strategies:
            passed = sum(1 for _, r in all_results[strat] if r["success"])
            total = len(all_results[strat])
            summary += f" {f'{passed}/{total}':<24}"
        print(summary)
        print(f"{'='*95}")
    else:
        try:
            run_simulation(args.config, gui=args.gui, dr_strategy=args.dr_strategy)
        except p.error:
            print("\n[SNAMOSim] GUI window closed. Terminating.")


if __name__ == "__main__":
    main()
