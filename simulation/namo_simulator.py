import time
import math
import argparse
import sys
import subprocess
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from decision.unet_decision_pipeline import UNetDecisionPipeline
from maps.namo_environments import WarehouseEnvironment, NAMOEnvironment

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

def create_goal_marker(position, color):
    visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.25, length=0.02, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )

def drive_robot(robot_id, target_xy, speed=5.0):

    position, _ = p.getBasePositionAndOrientation(robot_id)
    dx = target_xy[0] - position[0]
    dy = target_xy[1] - position[1]
    distance = math.hypot(dx, dy)

    if distance < 0.2:
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        return distance

    scale = speed / max(distance, 1e-6)
    vx = dx * scale
    vy = dy * scale
    yaw = math.atan2(dy, dx)
    orientation = p.getQuaternionFromEuler([0, 0, yaw])
    p.resetBasePositionAndOrientation(robot_id, [position[0], position[1], position[2]], orientation)
    p.resetBaseVelocity(robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, 0])
    return distance

class NAMO3DSimulator:
    def __init__(self, size, grid, start, goal, obstacle, decision, gui=False):
        self.size = size
        self.grid = grid
        self.start = start
        self.goal = goal
        self.obstacle = obstacle
        self.decision = decision
        self.gui = gui
        
    def find_clearing_direction(self):
        ox, oy = self.obstacle
        directions = [
            ((ox, oy + 1), (ox, oy - 1)), # Push South, approach from North
            ((ox, oy - 1), (ox, oy + 1)), # Push North, approach from South
            ((ox + 1, oy), (ox - 1, oy)), # Push East, approach from West
            ((ox - 1, oy), (ox + 1, oy))  # Push West, approach from East
        ]
        
        for clear_cell, approach_cell in directions:
            cx, cy = clear_cell
            ax, ay = approach_cell
            if 0 <= cx < self.size and 0 <= cy < self.size:
                if 0 <= ax < self.size and 0 <= ay < self.size:
                    if self.grid[cy, cx] == 0 and self.grid[ay, ax] == 0:
                        return clear_cell, approach_cell
        return (ox, oy + 1), (ox, oy - 1)

    def run(self):
        mode = p.GUI if self.gui else p.DIRECT
        p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.loadURDF("plane.urdf")
        
        cell_size = 1.0
        wall_height = 0.8
        
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=self.size * 1.3,
                cameraYaw=0,
                cameraPitch=-75,
                cameraTargetPosition=[self.size * 0.5, self.size * 0.5, 0.0],
            )
            
        # Spawn layout
        box_id = None
        for r in range(self.size):
            for c in range(self.size):
                x, y = cell_center(c, r, cell_size)
                val = self.grid[r, c]
                if val == 1:
                    create_static_wall([x, y, wall_height*0.5], [0.5, 0.5, wall_height*0.5])
                elif val == 2:
                    box_id = create_box([x, y, 0.2], [0.35, 0.35, 0.2], [0.95, 0.6, 0.1, 1.0])
                    p.changeDynamics(box_id, -1, lateralFriction=0.8)
                    
        # Spawn robot
        s_x, s_y = cell_center(self.start[0], self.start[1])
        robot_id = create_robot([s_x, s_y, 0.15], [0.2, 0.6, 0.9, 1.0])
        
        # Spawn goal marker
        g_x, g_y = cell_center(self.goal[0], self.goal[1])
        create_goal_marker([g_x, g_y, 0.01], [0.2, 0.6, 0.9, 0.4])
        
        # Call UNet Decision Pipeline to generate and save the visual heatmap plot
        project_dir = Path(__file__).resolve().parent.parent
        weights_path = str(project_dir / "models" / "namo_unet.pth")
        pipeline = UNetDecisionPipeline(weights_path)
        
        plot_filename = f"sim_{self.size}x{self.size}_decision.png"
        results_dir = project_dir / "results" / "visualizations"
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_save_path = str(results_dir / plot_filename)
        
        pipeline.evaluate_decision(
            self.grid, self.start, self.goal, self.obstacle, 
            push_cost=4.0, save_plot_path=plot_save_path
        )
        print(f" -> Automatically generated and saved plot to: results/visualizations/{plot_filename}")

        print(f"\n[Simulation] Executing NAMO {self.decision} on {self.size}x{self.size} grid...")

        # Dynamic Waypoint Generation
        waypoints = []
        if self.decision == "REMOVE":
            clear_cell, approach_cell = self.find_clearing_direction()
            
            grid_bypass = self.grid.copy()
            grid_bypass[self.obstacle[1], self.obstacle[0]] = 2
            _, path_to_approach = pipeline.risk_aware_a_star(grid_bypass, np.zeros((self.size, self.size)), self.start, approach_cell)
            
            waypoints.extend([cell_center(wp[0], wp[1]) for wp in path_to_approach[1:]])
            waypoints.append(cell_center(self.obstacle[0], self.obstacle[1]))
            waypoints.append(cell_center(clear_cell[0], clear_cell[1]))
            
            grid_after_push = self.grid.copy()
            grid_after_push[self.obstacle[1], self.obstacle[0]] = 0
            grid_after_push[clear_cell[1], clear_cell[0]] = 2
            
            _, path_to_goal = pipeline.risk_aware_a_star(grid_after_push, np.zeros((self.size, self.size)), clear_cell, self.goal)
            waypoints.extend([cell_center(wp[0], wp[1]) for wp in path_to_goal[1:]])
            
        else:
            grid_bypass = self.grid.copy()
            grid_bypass[self.obstacle[1], self.obstacle[0]] = 2
            _, path = pipeline.risk_aware_a_star(grid_bypass, np.zeros((self.size, self.size)), self.start, self.goal)
            waypoints = [cell_center(wp[0], wp[1]) for wp in path[1:]]

        # Drive sequence
        current_wp_idx = 0
        for step in range(500):
            if current_wp_idx >= len(waypoints):
                print(f" -> Goal reached dynamically at step {step}!")
                break
                
            target = waypoints[current_wp_idx]
            dist = drive_robot(robot_id, target)
            p.stepSimulation()
            if self.gui: time.sleep(0.01)
            
            if dist < 0.18:
                current_wp_idx += 1
                
        p.disconnect()

def simulate_env(env_name, gui=False):
    project_dir = Path(__file__).resolve().parent.parent
    weights_path = str(project_dir / "models" / "namo_unet.pth")
    pipeline = UNetDecisionPipeline(weights_path)
    
    # Early returns for standard test setups
    if env_name == "3x3":
        g3 = np.zeros((3, 3), dtype=int)
        g3[0, :] = 1; g3[2, :] = 1; g3[1, 1] = 2
        decision_3, _, _ = pipeline.evaluate_decision(g3, (0, 1), (2, 1), (1, 1))
        sim = NAMO3DSimulator(3, g3, (0, 1), (2, 1), (1, 1), decision_3, gui)
        sim.run()
        return
        
    elif env_name == "5x5":
        g5 = np.zeros((5, 5), dtype=int)
        g5[0, :] = 1; g5[4, :] = 1; g5[:, 0] = 1; g5[:, 4] = 1
        g5[1, 1] = 1; g5[1, 2] = 1; g5[1, 3] = 1
        g5[3, 1] = 1; g5[3, 2] = 1; g5[3, 3] = 1
        g5[2, 2] = 2
        decision_5, _, _ = pipeline.evaluate_decision(g5, (1, 2), (3, 2), (2, 2))
        sim = NAMO3DSimulator(5, g5, (1, 2), (3, 2), (2, 2), decision_5, gui)
        sim.run()
        return

    # From here, load dynamic YAML scenarios
    import yaml
    with open(env_name, "r") as f:
        config = yaml.safe_load(f)
        
    grid_size = config["world"]["grid_size"]
    if "warehouse" in config["meta"]["name"]:
        env = WarehouseEnvironment(grid_size=grid_size)
    else:
        env = NAMOEnvironment(grid_size=grid_size)
        
    for obs in config.get("obstacles", []):
        env.add_obstacle(obs["pos"][0], obs["pos"][1])
        
    for wall in config.get("walls", []):
        env.add_wall(wall["pos"][0], wall["pos"][1])

    env.robots = []
    env.goals = {}
    env.robot_counter = 3
    for rob in config.get("robots", []):
        env.add_robot(tuple(rob["start"]), tuple(rob["goal"]))
        
    grid = env.generate_occupancy_grid()
    
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)
    p.loadURDF("plane.urdf")
    
    cell_size = config["world"]["cell_size"]
    wall_height = 0.8
    
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=grid_size * 1.3,
            cameraYaw=0,
            cameraPitch=-75,
            cameraTargetPosition=[grid_size * 0.5, grid_size * 0.5, 0.0],
        )
        
    box_ids = []
    box_positions = {}
    for r in range(grid_size):
        for c in range(grid_size):
            x, y = cell_center(c, r, cell_size)
            val = grid[r, c]
            if val == 1:
                create_static_wall([x, y, wall_height*0.5], [0.5*cell_size, 0.5*cell_size, wall_height*0.5])
            elif val == 2:
                bid = create_box([x, y, 0.2], [0.35*cell_size, 0.35*cell_size, 0.2], [0.95, 0.6, 0.1, 1.0])
                p.changeDynamics(bid, -1, lateralFriction=0.8)
                box_ids.append(bid)
                box_positions[bid] = (c, r)
                
    robot_ids = []
    robot_goals = {}
    colors = [
        [0.2, 0.6, 0.9, 1.0],
        [0.9, 0.2, 0.2, 1.0],
        [0.2, 0.8, 0.2, 1.0],
        [0.8, 0.2, 0.8, 1.0],
        [0.2, 0.8, 0.8, 1.0],
    ]
    for i, robot in enumerate(env.robots):
        s_x, s_y = cell_center(robot['pos'][0], robot['pos'][1], cell_size)
        rid = create_robot([s_x, s_y, 0.15], colors[i % len(colors)])
        robot_ids.append(rid)
        robot_goals[rid] = robot['goal']
        
        # Spawn a flat goal marker disc on the ground
        g_x, g_y = cell_center(robot['goal'][0], robot['goal'][1], cell_size)
        r_color = colors[i % len(colors)]
        g_color = [r_color[0], r_color[1], r_color[2], 0.4]  # Translucent
        create_goal_marker([g_x, g_y, 0.01], g_color)
        
    import heapq
    def a_star_internal(start, goal, other_robots, boxes, ignore_boxes=False, locked_boxes=set(), impassable_boxes=None):
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
                    if grid[neighbor[1], neighbor[0]] == 1 or neighbor in locked_boxes or neighbor in impassable_boxes:
                        continue
                    if neighbor in robot_set:
                        continue
                    cost = 1.0
                    if not ignore_boxes and neighbor in box_set:
                        cost += 4.0  # Push cost penalty
                    tentative_g = g_score[current] + cost
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
        return []
        
    def find_clearing_direction_multi(box_cell, other_robots, boxes, locked_zones=set(), impassable_boxes=None):
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
                    if grid[cy, cx] != 1 and clear_cell not in robot_set and clear_cell not in locked_zones and clear_cell not in impassable_boxes:
                        if grid[ay, ax] != 1 and approach_cell not in robot_set and approach_cell not in box_set and approach_cell not in locked_zones and approach_cell not in impassable_boxes:
                            return clear_cell, approach_cell
        return None
        
    sim_steps = config["world"]["sim_steps"]
    print(f"\n[Simulation] Running Yaml configuration: {config['meta']['name']} with {len(robot_ids)} robots...")
    
    collision_count = 0
    success = False
    initial_box_pos = {}
    for bid in box_ids:
        pos, _ = p.getBasePositionAndOrientation(bid)
        initial_box_pos[bid] = pos[:2]
        
    robot_states = {rid: {"state": "NAVIGATING", "waypoints": []} for rid in robot_ids}
    active_clearings = {}  # box_cell -> rid (mutex reservation)
    
    for step in range(sim_steps):
        current_robot_positions = {}
        for rid in robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            current_robot_positions[rid] = world_to_cell(pos[:2], cell_size)
            
        current_box_positions = {}
        for bid in box_ids:
            pos, _ = p.getBasePositionAndOrientation(bid)
            current_box_positions[bid] = world_to_cell(pos[:2], cell_size)
            
        all_reached = True
        for rid in robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = robot_goals[rid]
            dist = math.hypot(pos[0] - (g[0]+0.5)*cell_size, pos[1] - (g[1]+0.5)*cell_size)
            if dist > 0.4:
                all_reached = False
                break
                
        if all_reached:
            success = True
            break
            
        # Coordinated next-cell reservation map (rid -> cell)
        planned_destinations = {}
        for rid in robot_ids:
            state_info = robot_states[rid]
            if state_info["state"] == "CLEARING" and state_info["waypoints"]:
                planned_destinations[rid] = state_info["waypoints"][0]
                
        for rid in robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = robot_goals[rid]
            dist = math.hypot(pos[0] - (g[0]+0.5)*cell_size, pos[1] - (g[1]+0.5)*cell_size)
            if dist <= 0.4:
                p.resetBaseVelocity(rid, [0,0,0], [0,0,0])
                continue
                
            curr_cell = current_robot_positions[rid]
            other_rob_cells = [c for r, c in current_robot_positions.items() if r != rid]
            box_cells = list(current_box_positions.values())
            
            state_info = robot_states[rid]
            
            if state_info["state"] == "CLEARING":
                if state_info["waypoints"]:
                    target_cell = state_info["waypoints"][0]
                    target_xy = cell_center(target_cell[0], target_cell[1], cell_size)
                    dist_to_wp = drive_robot(rid, target_xy)
                    if dist_to_wp < 0.22:
                        state_info["waypoints"].pop(0)
                else:
                    # Finished clearing, release the lock on this box
                    state_info["state"] = "NAVIGATING"
                    for box, locker_id in list(active_clearings.items()):
                        if locker_id == rid:
                            active_clearings.pop(box)
                    
            if state_info["state"] == "NAVIGATING":
                # Plan path avoiding other locked boxes and other robots' planned destinations
                other_next_cells = [cell for other_rid, cell in planned_destinations.items() if other_rid != rid]
                locked_boxes = set(active_clearings.keys()) | set(other_next_cells)
                
                path = a_star_internal(curr_cell, g, other_rob_cells, box_cells, ignore_boxes=False, locked_boxes=locked_boxes)
                if len(path) > 1:
                    # Check if path contains an unlocked box
                    blocking_box = None
                    for cell in path:
                        if cell in box_cells:
                            blocking_box = cell
                            break
                            
                    if blocking_box:
                        # Request a mutex lock on this box to clear it
                        is_locked = blocking_box in active_clearings
                        if not is_locked:
                            # Also avoid locking coordinates being cleared by others
                            locked_zones = set()
                            for r_states in robot_states.values():
                                if r_states["state"] == "CLEARING" and r_states["waypoints"]:
                                    locked_zones.update(r_states["waypoints"])
                                    
                            clearing_res = find_clearing_direction_multi(blocking_box, other_rob_cells, box_cells, locked_zones=locked_zones)
                            if clearing_res:
                                active_clearings[blocking_box] = rid
                                clear_cell, approach_cell = clearing_res
                                # Plan to approach cell avoiding other boxes and locked boxes
                                path_to_approach = a_star_internal(curr_cell, approach_cell, other_rob_cells, box_cells, ignore_boxes=False, locked_boxes=set(active_clearings.keys()) | set(other_next_cells))
                                if path_to_approach and not any(c in box_cells for c in path_to_approach):
                                    state_info["state"] = "CLEARING"
                                    state_info["waypoints"] = path_to_approach[1:] + [blocking_box, clear_cell]
                                    planned_destinations[rid] = state_info["waypoints"][0]
                                    target_xy = cell_center(state_info["waypoints"][0][0], state_info["waypoints"][0][1], cell_size)
                                    drive_robot(rid, target_xy)
                                else:
                                    # Fallback: drive directly to approach cell
                                    state_info["state"] = "CLEARING"
                                    state_info["waypoints"] = [approach_cell, blocking_box, clear_cell]
                                    planned_destinations[rid] = approach_cell
                                    target_xy = cell_center(approach_cell[0], approach_cell[1], cell_size)
                                    drive_robot(rid, target_xy)
                            else:
                                # No clearing direction: drive along the planned path (pushing straight)
                                planned_destinations[rid] = path[1]
                                target_xy = cell_center(path[1][0], path[1][1], cell_size)
                                drive_robot(rid, target_xy)
                        else:
                            # Box is locked by another robot. Wait/replan.
                            p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                    else:
                        # Normal path, no boxes
                        planned_destinations[rid] = path[1]
                        target_xy = cell_center(path[1][0], path[1][1], cell_size)
                        drive_robot(rid, target_xy)
                else:
                    if curr_cell == g:
                        target_xy = cell_center(g[0], g[1], cell_size)
                        drive_robot(rid, target_xy)
                    else:
                        # No path available (fully blocked/waiting). Stay stationary.
                        p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])

                        
        p.stepSimulation()
        if gui:
            time.sleep(0.01)
            
        for i in range(len(robot_ids)):
            for j in range(i+1, len(robot_ids)):
                if len(p.getContactPoints(robot_ids[i], robot_ids[j])) > 0:
                     collision_count += 1
                     
        for rid in robot_ids:
            for bid in box_ids:
                if len(p.getContactPoints(rid, bid)) > 0:
                     collision_count += 1
                     
    push_count = 0
    for bid in box_ids:
        pos, _ = p.getBasePositionAndOrientation(bid)
        init = initial_box_pos[bid]
        dist_moved = math.hypot(pos[0] - init[0], pos[1] - init[1])
        if dist_moved > 0.5:
            push_count += 1
            
    # Final success check in case they reached at the very last step
    all_reached_final = True
    for rid in robot_ids:
        pos, _ = p.getBasePositionAndOrientation(rid)
        g = robot_goals[rid]
        dist = math.hypot(pos[0] - (g[0]+0.5)*cell_size, pos[1] - (g[1]+0.5)*cell_size)
        print(f"Robot {rid} final dist: {dist:.4f} (pos: {pos[:2]}, goal: {g})")
        if dist > 0.4:
            all_reached_final = False
    if all_reached_final:
        success = True
            
    print("\n" + "="*50)
    print("SIMULATION RESULTS:")
    print(f"Success: {success}")
    print(f"Steps: {step if success and (step < sim_steps - 1) else sim_steps}")
    print(f"Obstacle Pushes: {push_count}")
    print(f"Contacts/Collisions: {collision_count}")
    print("="*50)
    p.disconnect()
    return {
        "success": success,
        "steps": step if success and (step < sim_steps - 1) else sim_steps,
        "pushes": push_count,
        "collisions": collision_count
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="all")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    
    python_bin = sys.executable
    script_path = __file__
    
    if args.env == "all":
        # Launch 3x3 and 5x5 as separate subprocesses to prevent OpenGL context crashes on Mac
        print("Launching 3x3 Simulation Subprocess...")
        cmd3 = [python_bin, script_path, "--env", "3x3"]
        if args.gui: cmd3.append("--gui")
        subprocess.run(cmd3)
        
        print("\nLaunching 5x5 Simulation Subprocess...")
        cmd5 = [python_bin, script_path, "--env", "5x5"]
        if args.gui: cmd5.append("--gui")
        subprocess.run(cmd5)
    else:
        simulate_env(args.env, args.gui)

if __name__ == "__main__":
    main()
