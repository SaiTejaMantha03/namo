import sys
import yaml
import numpy as np
import pybullet as p
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.snamo_simulator import build_clean_grid, cell_center, world_to_cell, create_wall, create_box, create_robot, create_goal_marker, COLORS
from decision.snamo_planner import SNAMOPlanner
from multi_robot.coordinator import RobotCoordinator, RobotState
from multi_robot.deadlock_resolution import DeadlockResolver
from multi_robot.belief_broadcaster import BeliefBroadcaster
from social.social_costmap import SocialCostmap

# Run simulator step-by-step and print info
config_path = "configs/custom_reconstructed_map_robots.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

world = cfg["world"]
gs = world["grid_size"]
cs = world.get("cell_size", 1.0)
max_steps = 600
robots_cfg = cfg.get("robots", [])

base_grid = build_clean_grid(cfg)

import pybullet_data
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

wh = 0.8
box_ids = []
box_init_xy = {}

for row in range(gs):
    for col in range(gs):
        x, y = cell_center(col, row, cs)
        v = base_grid[row, col]
        if v == 1:
            create_wall([x, y, wh * 0.5], [0.5 * cs, 0.5 * cs, wh * 0.5])
        elif v == 2:
            bid = create_box([x, y, 0.18], [0.34 * cs, 0.34 * cs, 0.18])
            box_ids.append(bid)
            raw, _ = p.getBasePositionAndOrientation(bid)
            box_init_xy[bid] = raw[:2]

robot_ids = []
robot_goals = {}
snamo_planners = {}

for i, rcfg in enumerate(robots_cfg):
    sc = tuple(rcfg["start"])
    gc = tuple(rcfg["goal"])
    sx, sy = cell_center(sc[0], sc[1], cs)
    rid = create_robot([sx, sy, 0.15], COLORS[i % len(COLORS)])
    robot_ids.append(rid)
    robot_goals[rid] = gc
    snamo_planners[rid] = SNAMOPlanner(grid=base_grid.copy(), grid_size=gs)

social_map = SocialCostmap(base_grid)
broadcaster = BeliefBroadcaster(robot_ids=robot_ids)
resolver = DeadlockResolver(grid=base_grid, grid_size=gs)
coordinator = RobotCoordinator(
    grid=base_grid, grid_size=gs, dr_strategy="sr_social",
    resolver=resolver, broadcaster=broadcaster, social_map=social_map,
)

coord_states = {}
for i, rid in enumerate(robot_ids):
    rcfg = robots_cfg[i]
    coord_states[rid] = RobotState(robot_id=rid, cell=tuple(rcfg["start"]), goal=tuple(rcfg["goal"]))

drive_target = {rid: cell_center(*tuple(robots_cfg[i]["start"]), cs) for i, rid in enumerate(robot_ids)}
push_obs_pending = {r: False for r in robot_ids}
last_snamo_action = {r: "NAVIGATE" for r in robot_ids}

for step in range(max_steps):
    robot_cells = {}
    for rid in robot_ids:
        pos, _ = p.getBasePositionAndOrientation(rid)
        robot_cells[rid] = world_to_cell(pos[:2], cs)

    box_cells_live = []
    current_grid = base_grid.copy()
    box_states_coord = {}
    for i, bid in enumerate(box_ids):
        pos, _ = p.getBasePositionAndOrientation(bid)
        bc = world_to_cell(pos[:2], cs)
        box_cells_live.append(bc)
        box_states_coord[bid] = bc
        bx, by = bc
        if 0 <= bx < gs and 0 <= by < gs:
            current_grid[by, bx] = 2

    coordinator.grid = current_grid.copy()
    resolver.grid = current_grid.copy()

    any_arrived = False
    for rid in robot_ids:
        if coord_states[rid].status not in ("DONE", "WAITING", "POCKET_WAITING") and rid in drive_target:
            pos, _ = p.getBasePositionAndOrientation(rid)
            tx, ty = drive_target[rid]
            if math.hypot(pos[0] - tx, pos[1] - ty) <= 0.15:
                any_arrived = True
                break

    if any_arrived or step % 30 == 0:
        for rid in robot_ids:
            coord_states[rid].cell = robot_cells[rid]
        next_cells = coordinator.step(coord_states, box_states_coord)
        for rid in robot_ids:
            if coord_states[rid].status in ("DONE", "WAITING", "POCKET_WAITING"):
                continue
            curr_cell = robot_cells[rid]
            goal_cell = robot_goals[rid]
            next_cell = next_cells.get(rid, curr_cell)

            if box_cells_live and coord_states[rid].status != "EVADING":
                other_cells = [robot_cells[r] for r in robot_ids if r != rid]
                plan_grid = current_grid.copy()
                for oc in other_cells:
                    ox, oy = oc
                    if (ox, oy) != goal_cell:
                        if 0 <= ox < gs and 0 <= oy < gs and plan_grid[oy, ox] == 0:
                            plan_grid[oy, ox] = 1

                snamo_planners[rid].update_grid(plan_grid)
                action, waypoints = snamo_planners[rid].plan(
                    start=curr_cell, goal=goal_cell,
                    box_cells=box_cells_live, other_robots=other_cells, grid=plan_grid,
                )
                last_snamo_action[rid] = action
                if waypoints:
                    next_cell = waypoints[0]
            drive_target[rid] = cell_center(next_cell[0], next_cell[1], cs)

    # Print trace info around step 480 to 530
    if 480 <= step <= 530:
        r_pos, _ = p.getBasePositionAndOrientation(robot_ids[0])
        # Find box at 13, 5
        box_pos_13_5 = None
        for bid in box_ids:
            b_pos, _ = p.getBasePositionAndOrientation(bid)
            b_cell = world_to_cell(b_pos[:2], cs)
            if b_cell == (13, 5):
                box_pos_13_5 = b_pos[:2]
                break
        print(f"Step {step} | Robot pos: {r_pos[:2]} cell: {robot_cells[robot_ids[0]]} target: {drive_target[robot_ids[0]]} action: {last_snamo_action[robot_ids[0]]} | Box (13,5) pos: {box_pos_13_5}")

    for rid in robot_ids:
        pos, _ = p.getBasePositionAndOrientation(rid)
        if coord_states[rid].status in ("WAITING", "POCKET_WAITING"):
            p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
            continue
        
        # Drive logic
        tx, ty = drive_target[rid]
        dx = tx - pos[0]
        dy = ty - pos[1]
        dist = math.hypot(dx, dy)
        if dist < 0.12:
            p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
        else:
            s = 4.0 / max(dist, 1e-6)
            yaw = math.atan2(dy, dx)
            p.resetBasePositionAndOrientation(rid, [pos[0], pos[1], 0.15], p.getQuaternionFromEuler([0, 0, yaw]))
            p.resetBaseVelocity(rid, [dx * s, dy * s, 0], [0, 0, 0])

    p.stepSimulation()

p.disconnect()
