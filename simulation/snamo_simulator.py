"""
simulation/snamo_simulator.py
-------------------------------
S-NAMO PyBullet simulator.

Uses SNAMOPlanner (social costmap + taboo zones + NAMOUnc uncertainty)
to drive robots through any YAML config map.

Usage:
    python simulation/snamo_simulator.py --config configs/movable_obstacle_choke_namo.yaml --gui
    python simulation/snamo_simulator.py --config configs/warehouse_3robots.yaml --gui
    python simulation/snamo_simulator.py --config all
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

from maps.namo_environments import NAMOEnvironment, WarehouseEnvironment
from decision.snamo_planner import SNAMOPlanner


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


def drive_robot(rid, target_xy, speed=4.0):
    """Move robot toward target_xy. Returns remaining distance."""
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
        rid, [pos[0], pos[1], pos[2]],
        p.getQuaternionFromEuler([0, 0, yaw]))
    p.resetBaseVelocity(rid, [dx * s, dy * s, 0], [0, 0, 0])
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Grid builder — NO robot IDs stamped (critical fix)
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

def run_simulation(config_path: str, gui: bool = False,
                   taboo_zones: list = None) -> dict:
    """
    Run S-NAMO simulation on a YAML config.

    Parameters
    ----------
    config_path : path to YAML file.
    gui         : open PyBullet GUI window.
    taboo_zones : optional list of polygon vertex lists.

    Returns
    -------
    dict with keys: success, steps, pushes, collisions
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    world     = cfg["world"]
    gs        = world["grid_size"]
    cs        = world.get("cell_size", 1.0)
    max_steps = world.get("sim_steps", 1500)
    robots_cfg = cfg.get("robots", [])

    if not robots_cfg:
        print(f"[SKIP] {config_path} — no robots defined.")
        return {"success": False, "steps": 0, "pushes": 0, "collisions": 0}

    base_grid = build_clean_grid(cfg)

    # ── PyBullet setup ────────────────────────────────────────────────────
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)
    p.loadURDF("plane.urdf")

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=gs * cs * 1.35,
            cameraYaw=0, cameraPitch=-70,
            cameraTargetPosition=[gs * cs * 0.5, gs * cs * 0.5, 0.0])

    # ── Spawn walls and boxes ──────────────────────────────────────────────
    wh = 0.8
    box_ids: list[int] = []
    box_init_xy: dict[int, tuple] = {}

    for row in range(gs):
        for col in range(gs):
            x, y = cell_center(col, row, cs)
            v = base_grid[row, col]
            if v == 1:
                create_wall([x, y, wh * 0.5],
                            [0.5 * cs, 0.5 * cs, wh * 0.5])
            elif v == 2:
                bid = create_box([x, y, 0.18],
                                 [0.34 * cs, 0.34 * cs, 0.18])
                box_ids.append(bid)
                raw, _ = p.getBasePositionAndOrientation(bid)
                box_init_xy[bid] = raw[:2]

    # ── Spawn robots ───────────────────────────────────────────────────────
    robot_ids:    list[int]         = []
    robot_goals:  dict[int, tuple]  = {}
    planners:     dict[int, SNAMOPlanner] = {}

    for i, rcfg in enumerate(robots_cfg):
        sc = tuple(rcfg["start"])
        gc = tuple(rcfg["goal"])
        sx, sy = cell_center(sc[0], sc[1], cs)
        rid = create_robot([sx, sy, 0.15], COLORS[i % len(COLORS)])
        robot_ids.append(rid)
        robot_goals[rid]  = gc
        planners[rid]     = SNAMOPlanner(
            grid      = base_grid.copy(),
            grid_size = gs,
            taboo_zones = taboo_zones or [],
        )
        # Goal disc
        gx, gy = cell_center(gc[0], gc[1], cs)
        c = COLORS[i % len(COLORS)]
        create_goal_marker([gx, gy, 0.01], [c[0], c[1], c[2], 0.30])

    # ── Per-robot state ────────────────────────────────────────────────────
    wp:          dict[int, list]  = {r: []       for r in robot_ids}
    last_action: dict[int, str]   = {r: "NAVIGATE" for r in robot_ids}
    push_obs_pending: dict[int, bool] = {r: False for r in robot_ids}
    wait_ticks:  dict[int, int]   = {r: 0  for r in robot_ids}  # forced wait counter

    # Intersection token: only one robot may be in the central 1/3 zone at a time
    intersection_holder: int | None = None  # rid currently holding the token
    cx_lo = gs // 3;  cx_hi = (2 * gs) // 3   # central zone bounds

    collision_count = 0
    success         = False
    step            = 0

    name = cfg.get("meta", {}).get("name", config_path)
    print(f"\n[S-NAMO] '{name}' | robots={len(robot_ids)} boxes={len(box_ids)} max_steps={max_steps}")

    for step in range(max_steps):

        # ── Read physics state ──────────────────────────────────────────
        robot_cells: dict[int, tuple] = {}
        for rid in robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            robot_cells[rid] = world_to_cell(pos[:2], cs)

        box_cells_live: list[tuple] = []
        current_grid = base_grid.copy()
        for bid in box_ids:
            pos, _ = p.getBasePositionAndOrientation(bid)
            bc = world_to_cell(pos[:2], cs)
            box_cells_live.append(bc)
            bx, by = bc
            if 0 <= bx < gs and 0 <= by < gs:
                current_grid[by, bx] = 2

        # ── Success check ───────────────────────────────────────────────
        all_done = True
        for rid in robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            gxc, gyc = robot_goals[rid]
            gxw, gyw = cell_center(gxc, gyc, cs)
            if math.hypot(pos[0] - gxw, pos[1] - gyw) > 0.5:
                all_done = False
                break
        if all_done:
            success = True
            print(f"[S-NAMO] ✅  All robots at goals — step {step}")
            break

        # ── Per-robot: replan + drive ────────────────────────────────────
        # Intersection token management (no-obstacle maps only)
        if not box_cells_live:
            for rid in robot_ids:
                rc = robot_cells[rid]
                in_zone = (cx_lo <= rc[0] <= cx_hi and cx_lo <= rc[1] <= cx_hi)
                if in_zone:
                    if intersection_holder is None:
                        intersection_holder = rid
                    elif intersection_holder != rid:
                        # Another robot holds token — this one waits
                        wait_ticks[rid] = 15
                else:
                    if intersection_holder == rid:
                        intersection_holder = None  # released the zone

        for rid in robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            gxc, gyc = robot_goals[rid]
            gxw, gyw = cell_center(gxc, gyc, cs)

            # Already at goal
            if math.hypot(pos[0] - gxw, pos[1] - gyw) <= 0.5:
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                continue

            # Forced wait (intersection token or yield)
            if wait_ticks[rid] > 0:
                wait_ticks[rid] -= 1
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                wp[rid] = []  # clear plan so we replan fresh after waiting
                continue

            curr_cell = robot_cells[rid]
            goal_cell = robot_goals[rid]

            # ── Replan when waypoints empty ──────────────────────────────
            if not wp[rid]:
                other_cells = [robot_cells[r] for r in robot_ids if r != rid]

                if box_cells_live:
                    # Maps with obstacles: mark other robots as temporary walls,
                    # but NEVER block the planning robot's own goal cell.
                    plan_grid = current_grid.copy()
                    for oc in other_cells:
                        ox, oy = oc
                        if (ox, oy) == goal_cell:
                            continue   # don't block the goal even if someone is there
                        if 0 <= ox < gs and 0 <= oy < gs and plan_grid[oy, ox] == 0:
                            plan_grid[oy, ox] = 1
                else:
                    # Maps without obstacles: use live grid as-is
                    plan_grid = current_grid.copy()

                planners[rid].update_grid(plan_grid)
                action, waypoints = planners[rid].plan(
                    start        = curr_cell,
                    goal         = goal_cell,
                    box_cells    = box_cells_live,
                    other_robots = other_cells if box_cells_live else [],
                    grid         = plan_grid,
                )
                wp[rid]          = list(waypoints)
                last_action[rid] = action

            # ── Drive toward next waypoint ───────────────────────────────
            if wp[rid]:
                nxt = wp[rid][0]

                # Yield if another robot occupies the next cell
                # (lower robot_ids index yields to higher-index robots)
                nxt_occupied = any(
                    robot_cells[r] == nxt
                    for r in robot_ids if r != rid
                )
                if nxt_occupied:
                    occupier = next(
                        r for r in robot_ids
                        if r != rid and robot_cells[r] == nxt
                    )
                    lo_rid, hi_rid = planners[rid].belief.success_rate_interval()
                    width_rid = hi_rid - lo_rid
                    lo_occ, hi_occ = planners[occupier].belief.success_rate_interval()
                    width_occ = hi_occ - lo_occ
                    rid_idx  = robot_ids.index(rid)
                    occ_idx = robot_ids.index(occupier)
                    
                    if width_rid > width_occ or (abs(width_rid - width_occ) < 1e-5 and rid_idx < occ_idx):
                        # This robot has wider interval (more uncertain) or tie-breaks lower — wait and clear plan
                        p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                        wp[rid] = []   # force replan next tick around the blocker
                        continue

                nx_w, ny_w = cell_center(nxt[0], nxt[1], cs)
                dist_wp = drive_robot(rid, (nx_w, ny_w))

                if dist_wp < 0.20:
                    wp[rid].pop(0)

                    # Detect push: if this cell had a box, trigger observe()
                    if last_action[rid] == "REMOVE":
                        bx, by = nxt
                        if 0 <= bx < gs and 0 <= by < gs:
                            if current_grid[by, bx] == 2:
                                push_obs_pending[rid] = True
                            elif push_obs_pending[rid]:
                                planners[rid].observe(success=True)
                                push_obs_pending[rid] = False

        # ── Physics step ─────────────────────────────────────────────────
        p.stepSimulation()
        if gui:
            time.sleep(0.01)

        # ── Count collisions ─────────────────────────────────────────────
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                if p.getContactPoints(robot_ids[i], robot_ids[j]):
                    collision_count += 1
        for rid in robot_ids:
            for bid in box_ids:
                if p.getContactPoints(rid, bid):
                    collision_count += 1

    # ── Final push count ─────────────────────────────────────────────────
    push_count = 0
    for bid in box_ids:
        pos, _ = p.getBasePositionAndOrientation(bid)
        ix, iy = box_init_xy[bid]
        if math.hypot(pos[0] - ix, pos[1] - iy) > 0.4:
            push_count += 1

    print(f"[S-NAMO] steps={step+1}  success={success}  "
          f"pushes={push_count}  collisions={collision_count//2}")
    p.disconnect()

    return {
        "success":    success,
        "steps":      step + 1,
        "pushes":     push_count,
        "collisions": collision_count // 2,
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
    ap = argparse.ArgumentParser(description="S-NAMO Simulator")
    ap.add_argument("--config", default="configs/movable_obstacle_choke_namo.yaml",
                    help="YAML config path or 'all'")
    ap.add_argument("--gui",   action="store_true", help="Open PyBullet GUI")
    args = ap.parse_args()

    if args.config == "all":
        rows = []
        for cfg_path in ALL_CONFIGS:
            print(f"\n{'='*58}\nRunning: {cfg_path}\n{'='*58}")
            r = run_simulation(cfg_path, gui=False)
            rows.append((cfg_path.split("/")[-1], r))

        print(f"\n{'='*72}")
        print(f"{'Config':<44} {'OK':<5} {'Steps':<8} {'Pushes':<8} Collisions")
        print(f"{'='*72}")
        for name, r in rows:
            ok = "✅" if r["success"] else "❌"
            print(f"{name:<44} {ok:<5} {r['steps']:<8} {r['pushes']:<8} {r['collisions']}")
    else:
        run_simulation(args.config, gui=args.gui)


if __name__ == "__main__":
    main()
