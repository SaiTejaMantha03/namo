import sys
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import heapq
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from decision.unet_decision_pipeline import UNetDecisionPipeline
from run_20_simulations import load_custom_map_yaml, cell_center, world_to_cell, create_box, create_static_wall, create_robot, drive_robot

GRID_DATA = load_custom_map_yaml()

def a_star_internal_baseline(start, goal, other_robots, boxes, grid_size=20, impassable_boxes=None,
                              treat_boxes_as_walls=False, use_risk_map=False, risk_map=None, risk_penalty_weight=10.0):
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
                if treat_boxes_as_walls and neighbor in box_set:
                    continue
                if neighbor in robot_set and neighbor != goal:
                    continue
                
                # Base step cost
                cost = 1.0
                if use_risk_map and risk_map is not None:
                    # Risk penalty weight
                    cost += risk_penalty_weight * risk_map[neighbor[1], neighbor[0]]
                
                if neighbor in box_set:
                    cost += 4.0
                    
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
    return []

def run_simulation_baseline(start, goal, pipeline, baseline_type="unet", risk_penalty_weight=10.0):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)
    p.loadURDF("plane.urdf")
    
    cell_size = 1.0
    wall_height = 0.8
    grid_size = 20
    
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
    
    # Compute risk map
    risk_map = None
    if baseline_type == "unet":
        risk_map = pipeline.get_risk_map(GRID_DATA, start, goal)
        
    treat_boxes_as_walls = (baseline_type == "a_star_only")
    use_risk_map = (baseline_type == "unet")
    
    for step in range(sim_steps):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
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
            
        if robot_state["state"] == "CLEARING" and not treat_boxes_as_walls:
            # Check if robot is stuck (barely moving)
            if len(pos_history) == 25:
                p_first = pos_history[0]
                p_last = pos_history[-1]
                movement = math.hypot(p_last[0] - p_first[0], p_last[1] - p_first[1])
                if movement < 0.05:
                    t_box = robot_state.get("target_box")
                    if t_box:
                        impassable_boxes.add(t_box)
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
            path = a_star_internal_baseline(curr_cell, goal, [], box_cells, impassable_boxes=impassable_boxes,
                                            treat_boxes_as_walls=treat_boxes_as_walls, use_risk_map=use_risk_map,
                                            risk_map=risk_map, risk_penalty_weight=risk_penalty_weight)
            if len(path) > 1:
                blocking = None
                for cell in path[1:]:  # Skip start cell
                    if cell in box_cells:
                        blocking = cell
                        break
                if blocking and not treat_boxes_as_walls:
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
                                        path_cand = a_star_internal_baseline(curr_cell, approach_cell, [], box_cells, impassable_boxes=impassable_boxes,
                                                                            treat_boxes_as_walls=treat_boxes_as_walls, use_risk_map=use_risk_map,
                                                                            risk_map=risk_map, risk_penalty_weight=risk_penalty_weight)
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
                else:
                    target_xy = cell_center(path[1][0], path[1][1], cell_size)
                    drive_robot(robot_id, target_xy)
            else:
                drive_robot(robot_id, cell_center(goal[0], goal[1], cell_size), speed=0.5)
                
        p.stepSimulation()
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
    return success, step if success else sim_steps, push_count, collision_count

def main():
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
    
    baselines = ["a_star_only", "reactive", "unet"]
    baseline_names = {
        "a_star_only": "A* Only (No Push)",
        "reactive": "Reactive NAMO (Plain A*)",
        "unet": "U-Net Risk-Guided NAMO (Ours)"
    }
    
    results = {b: [] for b in baselines}
    
    print("Running Baseline Evaluations on Custom Map Scenarios...")
    print("="*80)
    
    for b in baselines:
        print(f"\nRunning baseline: {baseline_names[b]}...")
        for idx, (start, goal) in enumerate(scenarios):
            success, steps, pushes, collisions = run_simulation_baseline(start, goal, pipeline, baseline_type=b)
            results[b].append({
                "success": success,
                "steps": steps,
                "pushes": pushes,
                "collisions": collisions
            })
            print(f"  Scen {idx+1:02d}: Success={str(success):5s} | Steps={steps:4d} | Pushes={pushes} | Collisions={collisions}")
            
    # Compile Summary Table
    summary_lines = [
        "# Ablation & Baseline Evaluation Results",
        "",
        "| Evaluation Method | Success Rate (%) | Avg Steps Taken | Avg Obstacle Pushes | Avg Contacts/Collisions |",
        "|:---|:---:|:---:|:---:|:---:|"
    ]
    
    for b in baselines:
        data = results[b]
        success_rate = sum(1 for r in data if r["success"]) / len(data) * 100.0
        avg_steps = sum(r["steps"] for r in data) / len(data)
        avg_pushes = sum(r["pushes"] for r in data) / len(data)
        avg_collisions = sum(r["collisions"] for r in data) / len(data)
        
        summary_lines.append(f"| {baseline_names[b]} | {success_rate:.1f}% | {avg_steps:.1f} | {avg_pushes:.1f} | {avg_collisions:.1f} |")
        
    summary_md = "\n".join(summary_lines)
    
    # Save results
    eval_dir = project_dir / "results" / "evaluation_tables"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "ablation_baselines_table.md").write_text(summary_md)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED. SUMMARY TABLE:")
    print("="*80)
    print(summary_md)
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
