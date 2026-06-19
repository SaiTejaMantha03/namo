import sys
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import heapq
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from decision.unet_decision_pipeline import UNetDecisionPipeline
from maps.namo_environments import NAMOEnvironment

import yaml

def load_custom_map_yaml():
    project_dir = Path(__file__).resolve().parent
    yaml_path = project_dir / "configs" / "custom_reconstructed_map_robots.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        
    grid_size = config["world"]["grid_size"]
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Set custom walls
    for w in config.get("walls", []):
        grid[w["pos"][1], w["pos"][0]] = 1
        
    # Set obstacles
    for obs in config.get("obstacles", []):
        grid[obs["pos"][1], obs["pos"][0]] = 2
        
    return grid


GRID_DATA = load_custom_map_yaml()


def cell_center(cell_x, cell_y, cell_size=1.0):
    return (cell_x + 0.5) * cell_size, (cell_y + 0.5) * cell_size

def world_to_cell(position_xy, cell_size=1.0):
    return int(position_xy[0] // cell_size), int(position_xy[1] // cell_size)

def create_box(position, half_extents, color):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(
        baseMass=1.5,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )

def create_static_wall(position, half_extents):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.3, 0.3, 0.3, 1.0])
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )

def create_robot(position, color):
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.25, height=0.3)
    visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.25, length=0.3, rgbaColor=color)
    body_id = p.createMultiBody(
        baseMass=2.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )
    p.changeDynamics(body_id, -1, lateralFriction=1.0, linearDamping=0.2, angularDamping=0.9)
    return body_id

def drive_robot(robot_id, target_xy, speed=5.0):

    position, _ = p.getBasePositionAndOrientation(robot_id)
    dx = target_xy[0] - position[0]
    dy = target_xy[1] - position[1]
    distance = math.hypot(dx, dy)

    if distance < 0.22:
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        return distance

    scale = speed / max(distance, 1e-6)
    vx = dx * scale
    vy = dy * scale
    yaw = math.atan2(dy, dx)
    orientation = p.getQuaternionFromEuler([0, 0, yaw])
    p.resetBasePositionAndOrientation(robot_id, [position[0], position[1], 0.15], orientation)
    p.resetBaseVelocity(robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, 0])
    return distance

def a_star_internal(start, goal, other_robots, boxes, grid_size=20, impassable_boxes=None):
    if impassable_boxes is None:
        impassable_boxes = set()
    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    robot_set = set(other_robots)
    box_set = set(boxes)
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                if GRID_DATA[neighbor[1], neighbor[0]] == 1 or neighbor in impassable_boxes:
                    continue
                if neighbor in robot_set and neighbor != goal:
                    continue
                cost = 1.0
                if neighbor in box_set:
                    cost += 4.0
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
    return []

def find_clearing_direction_multi(box_cell, other_robots, boxes, grid_size=20, impassable_boxes=None):
    if impassable_boxes is None:
        impassable_boxes = set()
    bx, by = box_cell
    directions = [
        ((bx, by + 1), (bx, by - 1)),
        ((bx, by - 1), (bx, by + 1)),
        ((bx + 1, by), (bx - 1, by)),
        ((bx - 1, by), (bx + 1, by))
    ]
    robot_set = set(other_robots)
    box_set = set(boxes)
    for clear_cell, approach_cell in directions:
        cx, cy = clear_cell
        ax, ay = approach_cell
        if 0 <= cx < grid_size and 0 <= cy < grid_size:
            if 0 <= ax < grid_size and 0 <= ay < grid_size:
                if GRID_DATA[cy, cx] != 1 and clear_cell not in robot_set and clear_cell not in impassable_boxes:
                    if GRID_DATA[ay, ax] != 1 and approach_cell not in robot_set and approach_cell not in box_set and approach_cell not in impassable_boxes:
                        return clear_cell, approach_cell
    return None

def run_simulation(start, goal, pipeline, gui=False):
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)
    p.loadURDF("plane.urdf")
    
    cell_size = 1.0
    wall_height = 0.8
    grid_size = 20
    
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=25.0,
            cameraYaw=0,
            cameraPitch=-75,
            cameraTargetPosition=[10.0, 10.0, 0.0],
        )
        
    box_ids = []
    
    # Spawn layout
    for r in range(grid_size):
        for c in range(grid_size):
            x, y = cell_center(c, r, cell_size)
            val = GRID_DATA[r, c]
            if val == 1:
                create_static_wall([x, y, wall_height*0.5], [0.5, 0.5, wall_height*0.5])
            elif val == 2:
                bid = create_box([x, y, 0.2], [0.35, 0.35, 0.2], [0.95, 0.6, 0.1, 1.0])
                p.changeDynamics(bid, -1, lateralFriction=0.8)
                box_ids.append(bid)
                
    # Spawn robot
    s_x, s_y = cell_center(start[0], start[1])
    robot_id = create_robot([s_x, s_y, 0.15], [0.2, 0.6, 0.9, 1.0])
    
    initial_box_pos = {}
    for bid in box_ids:
        pos, _ = p.getBasePositionAndOrientation(bid)
        initial_box_pos[bid] = pos[:2]
        
    robot_state = {"state": "NAVIGATING", "waypoints": [], "target_box": None}
    impassable_boxes = set()
    pos_history = []
    success = False
    collision_count = 0
    sim_steps = 1500
    
    dummy_risk = np.zeros((grid_size, grid_size))
    _, init_path = pipeline.risk_aware_a_star(GRID_DATA, dummy_risk, start, goal)
    blocking_box = None
    if init_path:
        for cell in init_path:
            if GRID_DATA[cell[1], cell[0]] == 2:
                blocking_box = cell
                break
    if blocking_box is None:
        blocking_box = (10, 10)
    decision, _, _ = pipeline.evaluate_decision(GRID_DATA, start, goal, blocking_box, push_cost=4.0)
    
    for step in range(sim_steps):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        if step % 50 == 0:
            print(f"step={step:03d} pos=({pos[0]:.2f},{pos[1]:.2f}) state={robot_state['state']} wps={robot_state['waypoints']} impassable={impassable_boxes}")
        dist = math.hypot(pos[0] - (goal[0]+0.5)*cell_size, pos[1] - (goal[1]+0.5)*cell_size)
        if dist <= 0.4:
            success = True
            break
            
        curr_cell = world_to_cell(pos[:2], cell_size)
        
        current_box_positions = {}
        for bid in box_ids:
            bpos, _ = p.getBasePositionAndOrientation(bid)
            current_box_positions[bid] = world_to_cell(bpos[:2], cell_size)
            
        box_cells = list(current_box_positions.values())
        
        # Track position history for stuck detection during CLEARING state
        pos_history.append(pos[:2])
        if len(pos_history) > 25:
            pos_history.pop(0)
            
        if robot_state["state"] == "CLEARING":
            # Check if robot is stuck (barely moving)
            if len(pos_history) == 25:
                p_first = pos_history[0]
                p_last = pos_history[-1]
                movement = math.hypot(p_last[0] - p_first[0], p_last[1] - p_first[1])
                if movement < 0.05:
                    t_box = robot_state.get("target_box")
                    if t_box:
                        impassable_boxes.add(t_box)
                    print(f" -> [Stuck Detected] Aborting CLEARING of box {t_box} at step {step}. Adding to impassable_boxes.")
                    robot_state["state"] = "NAVIGATING"
                    robot_state["waypoints"] = []
                    robot_state["target_box"] = None
                    pos_history.clear()
            
            if robot_state["state"] == "CLEARING":
                if robot_state["waypoints"]:
                    target_cell = robot_state["waypoints"][0]
                    target_xy = cell_center(target_cell[0], target_cell[1], cell_size)
                    dist_to_wp = drive_robot(robot_id, target_xy)
                    if dist_to_wp < 0.22:
                        robot_state["waypoints"].pop(0)
                else:
                    robot_state["state"] = "NAVIGATING"
                    robot_state["target_box"] = None
                
        if robot_state["state"] == "NAVIGATING":
            path = a_star_internal(curr_cell, goal, [], box_cells, impassable_boxes=impassable_boxes)
            if len(path) > 1:
                blocking = None
                for cell in path[1:]:  # Skip start cell
                    if cell in box_cells:
                        blocking = cell
                        break
                if blocking:
                    # Find a clearing direction where the approach path is NOT blocked by box cells
                    clearing_res = None
                    path_to_approach = None
                    
                    bx, by = blocking
                    directions = [
                        ((bx, by + 1), (bx, by - 1)),
                        ((bx, by - 1), (bx, by + 1)),
                        ((bx + 1, by), (bx - 1, by)),
                        ((bx - 1, by), (bx + 1, by))
                    ]
                    for clear_cell, approach_cell in directions:
                        if 0 <= clear_cell[0] < grid_size and 0 <= clear_cell[1] < grid_size:
                            if 0 <= approach_cell[0] < grid_size and 0 <= approach_cell[1] < grid_size:
                                if GRID_DATA[clear_cell[1], clear_cell[0]] != 1 and clear_cell not in impassable_boxes:
                                    if GRID_DATA[approach_cell[1], approach_cell[0]] != 1 and approach_cell not in box_cells and approach_cell not in impassable_boxes:
                                        path_cand = a_star_internal(curr_cell, approach_cell, [], box_cells, impassable_boxes=impassable_boxes)
                                        if path_cand and not any(c in box_cells for c in path_cand[1:]):
                                            clearing_res = (clear_cell, approach_cell)
                                            path_to_approach = path_cand
                                            break
                                            
                    if clearing_res and path_to_approach:
                        clear_cell, approach_cell = clearing_res
                        robot_state["state"] = "CLEARING"
                        robot_state["waypoints"] = path_to_approach[1:] + [blocking, clear_cell]
                        robot_state["target_box"] = blocking
                        pos_history.clear()
                        target_xy = cell_center(robot_state["waypoints"][0][0], robot_state["waypoints"][0][1], cell_size)
                        drive_robot(robot_id, target_xy)
                    else:
                        impassable_boxes.add(blocking)
                        print(f" -> Cannot clear box {blocking} (approach path blocked or no clearing direction). Adding to impassable_boxes.")
                else:
                    target_xy = cell_center(path[1][0], path[1][1], cell_size)
                    drive_robot(robot_id, target_xy)
            else:
                drive_robot(robot_id, cell_center(goal[0], goal[1], cell_size), speed=0.5)
                
        p.stepSimulation()
        if gui:
            time.sleep(0.01)
        
        for bid in box_ids:
            if len(p.getContactPoints(robot_id, bid)) > 0:
                collision_count += 1
                
    push_count = 0
    for bid in box_ids:
        pos, _ = p.getBasePositionAndOrientation(bid)
        init = initial_box_pos[bid]
        dist_moved = math.hypot(pos[0] - init[0], pos[1] - init[1])
        if dist_moved > 0.5:
            push_count += 1
            
    p.disconnect()
    return decision, success, step, push_count, collision_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Launch in PyBullet 3D GUI mode")
    parser.add_argument("--scenario", type=int, default=0, help="Run a specific scenario ID (1-20)")
    args = parser.parse_args()
    
    project_dir = Path(__file__).resolve().parent
    weights_path = str(project_dir / "models" / "namo_unet.pth")
    pipeline = UNetDecisionPipeline(weights_path)
    
    scenarios = [
        ((17, 0), (15, 2)),
        ((17, 0), (18, 2)),
        ((17, 0), (15, 3)),
        ((17, 0), (17, 3)),
        ((17, 0), (19, 3)),
        ((18, 0), (15, 2)),
        ((18, 0), (18, 2)),
        ((18, 0), (15, 3)),
        ((18, 0), (17, 3)),
        ((18, 0), (19, 3)),
        ((17, 0), (0, 1)),
        ((17, 0), (1, 1)),
        ((17, 0), (4, 1)),
        ((17, 0), (5, 1)),
        ((17, 0), (6, 1)),
        ((18, 0), (0, 1)),
        ((18, 0), (1, 1)),
        ((18, 0), (4, 1)),
        ((18, 0), (5, 1)),
        ((18, 0), (6, 1)),
    ]
    
    if args.scenario > 0:
        if args.scenario <= len(scenarios):
            start, goal = scenarios[args.scenario - 1]
            print(f"Launching Scenario {args.scenario:02d} in GUI mode (Start={start}, Goal={goal})...")
            decision, success, steps, pushes, collisions = run_simulation(start, goal, pipeline, gui=args.gui)
            print(f"Result -> Success: {success} | Steps: {steps} | Pushes: {pushes} | Collisions: {collisions}")
        else:
            print(f"Invalid scenario ID {args.scenario}. Must be between 1 and {len(scenarios)}.")
    else:
        print("Running 20 simulations on the custom map...")
        results = []
        for idx, (start, goal) in enumerate(scenarios):
            decision, success, steps, pushes, collisions = run_simulation(start, goal, pipeline, gui=args.gui)
            results.append({
                "id": idx + 1,
                "start": start,
                "goal": goal,
                "decision": decision,
                "success": success,
                "steps": steps if success else 1500,
                "pushes": pushes,
                "collisions": collisions
            })
            print(f"Scenario {idx+1:02d}: Start={start} Goal={goal} | Decision={decision:7s} | Success={success} | Steps={steps} | Pushes={pushes} | Collisions={collisions}")
            
        sample_start = (17, 0)
        sample_goal = (5, 1)
        sample_obstacle = (16, 1)
        plot_filename = "custom_map_unet_evaluation.png"
        results_dir = project_dir / "results" / "custom_simulations"
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_save_path = str(results_dir / plot_filename)
        
        pipeline.evaluate_decision(GRID_DATA, sample_start, sample_goal, sample_obstacle, push_cost=4.0, save_plot_path=plot_save_path)

        
        table_lines = [
            "| Scenario | Start | Goal | Model Decision | Success | Steps | Pushes | Collisions |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        ]
        for r in results:
            success_str = "✅ Yes" if r["success"] else "❌ No"
            table_lines.append(f"| {r['id']} | {r['start']} | {r['goal']} | **{r['decision']}** | {success_str} | {r['steps']} | {r['pushes']} | {r['collisions']} |")
            
        results_dir = project_dir / "results" / "custom_simulations"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "custom_results_table.md").write_text("\n".join(table_lines))

if __name__ == '__main__':
    main()
