import os
import sys
import math
import yaml
import numpy as np
import torch
import pybullet as p
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.namo_simulator import (
    cell_center, world_to_cell, create_box, create_static_wall,
    create_robot, create_goal_marker, drive_robot
)
from maps.namo_environments import WarehouseEnvironment, NAMOEnvironment
from decision.unet_decision_pipeline import UNetDecisionPipeline
from multi_robot.belief_broadcaster import BeliefBroadcaster

def a_star_internal(start, goal, other_robots, boxes, grid, grid_size, ignore_boxes=False, locked_boxes=set()):
    import heapq
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
                if grid[neighbor[1], neighbor[0]] == 1 or neighbor in locked_boxes:
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

def find_clearing_direction_multi(box_cell, other_robots, boxes, grid, grid_size, locked_zones=set()):
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
                if grid[cy, cx] != 1 and clear_cell not in robot_set and clear_cell not in box_set and clear_cell not in locked_zones:
                    if grid[ay, ax] != 1 and approach_cell not in robot_set and approach_cell not in box_set and approach_cell not in locked_zones:
                        return clear_cell, approach_cell
    return None

class NAMOmappoEnv:
    def __init__(self, config_path, gui=False, max_steps=100, control_interval=15):
        self.config_path = config_path
        self.gui = gui
        self.max_steps = max_steps
        self.control_interval = control_interval
        
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.grid_size = self.config["world"]["grid_size"]
        self.cell_size = self.config["world"]["cell_size"]
        
        # Load frozen UNet Pipeline
        project_dir = Path(__file__).resolve().parent.parent
        weights_path = str(project_dir / "models" / "namo_unet.pth")
        self.unet_pipeline = UNetDecisionPipeline(weights_path)
        
        self.physics_client = None
        self.robot_ids = []
        self.box_ids = []
        self.robot_goals = {}
        self.robot_states = {}
        self.active_clearings = {}
        self.current_step = 0
        self.sim_step_count = 0
        self.broadcaster = None
        self.push_obs_pending = {}
        
    def reset(self):
        if self.physics_client is not None:
            try:
                p.disconnect(physicsClientId=self.physics_client)
            except Exception:
                pass
                
        mode = p.GUI if self.gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        
        # Re-build floor
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        
        # Build environments
        if "warehouse" in self.config["meta"]["name"]:
            env = WarehouseEnvironment(grid_size=self.grid_size)
        else:
            env = NAMOEnvironment(grid_size=self.grid_size)
            
        for obs in self.config.get("obstacles", []):
            env.add_obstacle(obs["pos"][0], obs["pos"][1])
        for wall in self.config.get("walls", []):
            env.add_wall(wall["pos"][0], wall["pos"][1])
            
        # Get grids
        self.grid = env.generate_occupancy_grid()
        
        # Spawn layout walls and boxes
        self.box_ids = []
        self.box_id_map = {} # position -> body_id
        wall_height = 0.8
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x, y = cell_center(c, r, self.cell_size)
                val = self.grid[r, c]
                if val == 1:
                    create_static_wall([x, y, wall_height * 0.5], [0.5, 0.5, wall_height * 0.5])
                elif val == 2:
                    bid = create_box([x, y, 0.2], [0.35, 0.35, 0.2], [0.95, 0.6, 0.1, 1.0])
                    p.changeDynamics(bid, -1, lateralFriction=0.8)
                    self.box_ids.append(bid)
                    
        # Spawn robots
        self.robot_ids = []
        self.robot_goals = {}
        colors = [[0.2, 0.6, 0.9, 1.0], [0.8, 0.2, 0.2, 1.0], [0.2, 0.8, 0.2, 1.0], [0.8, 0.8, 0.2, 1.0]]
        for i, rob in enumerate(self.config["robots"]):
            s_x, s_y = cell_center(rob["start"][0], rob["start"][1], self.cell_size)
            rid = create_robot([s_x, s_y, 0.15], colors[i % len(colors)])
            self.robot_ids.append(rid)
            self.robot_goals[rid] = rob["goal"]
            
            # Spawn flat goal marker
            g_x, g_y = cell_center(rob["goal"][0], rob["goal"][1], self.cell_size)
            r_color = colors[i % len(colors)]
            g_color = [r_color[0], r_color[1], r_color[2], 0.4]
            create_goal_marker([g_x, g_y, 0.01], g_color)
            
        # Initialize robot states
        self.robot_states = {
            rid: {
                "state": "NAVIGATING",
                "waypoints": [],
                "target_box": None,
                "yield_cell": None
            } for rid in self.robot_ids
        }
        self.active_clearings = {}
        self.current_step = 0
        self.sim_step_count = 0
        self.agent_done_flags = {}
        
        # Initialize belief broadcaster and push pending tracking
        self.broadcaster = BeliefBroadcaster(self.robot_ids)
        self.push_obs_pending = {rid: False for rid in self.robot_ids}
        
        return self._get_observations()
        
    def _get_unet_local_crop(self, agent_rid):
        """Generates 5x5 local risk crop from the UNet risk map around the agent's current position."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == 1:
                    grid[r, c] = 1
        for bid in self.box_ids:
            pos, _ = p.getBasePositionAndOrientation(bid)
            bx, by = world_to_cell(pos[:2], self.cell_size)
            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                grid[by, bx] = 2
        pos, _ = p.getBasePositionAndOrientation(agent_rid)
        ax, ay = world_to_cell(pos[:2], self.cell_size)
        g = self.robot_goals[agent_rid]
        heatmap = self.unet_pipeline.get_risk_map(grid, (ax, ay), g)
        padded = np.pad(heatmap, pad_width=2, mode='constant', constant_values=1.0)
        crop = padded[ay : ay + 5, ax : ax + 5]
        return crop

    def _get_sr_width(self, rid):
        return self.broadcaster.get_sr_interval_width(rid)

    def _get_observations(self):
        obs = {}
        # Get positions
        current_robot_positions = {}
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            current_robot_positions[rid] = pos[:2]
            
        current_box_positions = []
        for bid in self.box_ids:
            pos, _ = p.getBasePositionAndOrientation(bid)
            current_box_positions.append(pos[:2])
            
        for rid in self.robot_ids:
            # 1. Local 5x5 crop from UNet risk heatmap
            crop = self._get_unet_local_crop(rid).flatten()
            
            # 2. Own relative goal info
            own_pos = current_robot_positions[rid]
            g = self.robot_goals[rid]
            goal_pos = np.array([(g[0]+0.5)*self.cell_size, (g[1]+0.5)*self.cell_size])
            
            # 3. Teammates
            other_rob_pos = []
            for other_rid in self.robot_ids:
                if other_rid != rid:
                    other_rob_pos.append(current_robot_positions[other_rid])
            # Pad teammate info to support up to 3 teammates (6 elements)
            while len(other_rob_pos) < 3:
                other_rob_pos.append([0.0, 0.0])
            other_rob_pos = np.array(other_rob_pos).flatten()
            
            # 4. Box positions (closest 3 boxes)
            box_dists = []
            for bp in current_box_positions:
                box_dists.append((math.hypot(own_pos[0]-bp[0], own_pos[1]-bp[1]), bp))
            box_dists.sort(key=lambda x: x[0])
            
            nearby_boxes = [bp for _, bp in box_dists[:3]]
            while len(nearby_boxes) < 3:
                nearby_boxes.append([0.0, 0.0])
            nearby_boxes = np.array(nearby_boxes).flatten()
            
            # 5. Uncertainty features
            sr_mean = self.broadcaster.belief_models[rid].mean()
            sr_width = self.broadcaster.get_sr_interval_width(rid)
            sr_obs = float(self.broadcaster.belief_models[rid]._n_obs)
            norm_sr_obs = min(1.0, sr_obs / 20.0)
            uncertainty_feats = np.array([sr_mean, sr_width, norm_sr_obs], dtype=np.float32)
            
            # Construct observation vector
            norm_own = np.array(own_pos) / (self.grid_size * self.cell_size)
            norm_goal = goal_pos / (self.grid_size * self.cell_size)
            norm_dir = (goal_pos - own_pos) / (self.grid_size * self.cell_size)
            norm_others = other_rob_pos / (self.grid_size * self.cell_size)
            norm_boxes = nearby_boxes / (self.grid_size * self.cell_size)
            
            obs_vec = np.concatenate([
                crop,             # 25
                norm_own,         # 2
                norm_goal,        # 2
                norm_dir,         # 2
                norm_others,      # 6
                norm_boxes,       # 6
                uncertainty_feats,# 3
            ])
            obs[rid] = obs_vec
            
        return obs
        
    def step(self, actions):
        """
        actions: dict mapping robot_id -> action index (0: NAVIGATE, 1: PUSH_BOX, 2: YIELD, 3: WAIT)
        """
        self.current_step += 1
        
        # Get starting goal distances for reward shaping
        starting_dists = {}
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = self.robot_goals[rid]
            starting_dists[rid] = math.hypot(pos[0] - (g[0]+0.5)*self.cell_size, pos[1] - (g[1]+0.5)*self.cell_size)
            
        # Apply strategic actions
        for rid, act in actions.items():
            state_info = self.robot_states[rid]
            pos, _ = p.getBasePositionAndOrientation(rid)
            curr_cell = world_to_cell(pos[:2], self.cell_size)
            
            # Read box cells
            box_cells = []
            box_map = {}
            for bid in self.box_ids:
                bpos, _ = p.getBasePositionAndOrientation(bid)
                bc = world_to_cell(bpos[:2], self.cell_size)
                box_cells.append(bc)
                box_map[bc] = bid
                
            other_rob_cells = []
            for other_rid in self.robot_ids:
                if other_rid != rid:
                    opos, _ = p.getBasePositionAndOrientation(other_rid)
                    other_rob_cells.append(world_to_cell(opos[:2], self.cell_size))
                    
            if act == 0: # NAVIGATE
                state_info["state"] = "NAVIGATING"
                state_info["target_box"] = None
                state_info["yield_cell"] = None
                
            elif act == 1: # PUSH_BOX
                # Locate closest box
                closest_box = None
                min_dist = float('inf')
                for bc in box_cells:
                    d = abs(curr_cell[0] - bc[0]) + abs(curr_cell[1] - bc[1])
                    if d < min_dist:
                        min_dist = d
                        closest_box = bc
                if closest_box and closest_box not in self.active_clearings:
                    self.active_clearings[closest_box] = rid
                    state_info["state"] = "CLEARING"
                    state_info["target_box"] = closest_box
                    
            elif act == 2: # YIELD
                # Find adjacent cell out of the main pathway/corridor
                yield_options = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    c_x, c_y = curr_cell[0] + dx, curr_cell[1] + dy
                    if 0 <= c_x < self.grid_size and 0 <= c_y < self.grid_size:
                        if self.grid[c_y, c_x] != 1 and (c_x, c_y) not in box_cells and (c_x, c_y) not in other_rob_cells:
                            yield_options.append((c_x, c_y))
                # Prefer cells that are off the main row/column of the corridor
                if yield_options:
                    # Pick option furthest from active robot goals or corridors
                    state_info["state"] = "YIELDING"
                    state_info["yield_cell"] = yield_options[0]
                    state_info["waypoints"] = [yield_options[0]]
                    
            elif act == 3: # WAIT
                state_info["state"] = "WAITING"
                state_info["waypoints"] = []
                
        # Run simulator for control_interval steps
        collisions_this_step = 0
        for _ in range(self.control_interval):
            self.sim_step_count += 1
            
            # Get current dynamic positions
            current_robot_positions = {}
            for rid in self.robot_ids:
                pos, _ = p.getBasePositionAndOrientation(rid)
                current_robot_positions[rid] = world_to_cell(pos[:2], self.cell_size)
                
            current_box_positions = {}
            for bid in self.box_ids:
                pos, _ = p.getBasePositionAndOrientation(bid)
                current_box_positions[bid] = world_to_cell(pos[:2], self.cell_size)
                
            # Internal navigation and path execution loop
            planned_destinations = {}
            for rid in self.robot_ids:
                state_info = self.robot_states[rid]
                if state_info["state"] == "CLEARING" and state_info["waypoints"]:
                    planned_destinations[rid] = state_info["waypoints"][0]
                    
            for rid in self.robot_ids:
                pos, _ = p.getBasePositionAndOrientation(rid)
                g = self.robot_goals[rid]
                dist = math.hypot(pos[0] - (g[0]+0.5)*self.cell_size, pos[1] - (g[1]+0.5)*self.cell_size)
                if dist <= 0.4:
                    p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                    continue
                    
                curr_cell = current_robot_positions[rid]
                other_rob_cells = [c for r, c in current_robot_positions.items() if r != rid]
                box_cells = list(current_box_positions.values())
                
                state_info = self.robot_states[rid]
                
                if state_info["state"] == "CLEARING":
                    # If waypoints are empty, plan them
                    if not state_info["waypoints"] and state_info["target_box"]:
                        # Setup approach and push sequence dynamically avoiding walls/robots
                        clearing_res = find_clearing_direction_multi(state_info["target_box"], other_rob_cells, box_cells, self.grid, self.grid_size)
                        if clearing_res:
                            clear_cell, approach_cell = clearing_res
                            path_to_approach = a_star_internal(curr_cell, approach_cell, other_rob_cells, box_cells, self.grid, self.grid_size)
                            if path_to_approach:
                                state_info["waypoints"] = path_to_approach[1:] + [state_info["target_box"], clear_cell]
                            else:
                                state_info["waypoints"] = [approach_cell, state_info["target_box"], clear_cell]
                        else:
                            # Cannot find a safe clearing direction, abort push
                            state_info["state"] = "NAVIGATING"
                            state_info["target_box"] = None
                        
                    if state_info["waypoints"]:
                        target_cell = state_info["waypoints"][0]
                        
                        # SR-width yielding: check if target_cell is occupied by another robot
                        nxt_occupied = any(
                            current_robot_positions[r] == target_cell
                            for r in self.robot_ids if r != rid
                        )
                        if nxt_occupied:
                            occupier = next(
                                r for r in self.robot_ids
                                if r != rid and current_robot_positions[r] == target_cell
                            )
                            width_rid = self._get_sr_width(rid)
                            width_occ = self._get_sr_width(occupier)
                            if width_rid > width_occ or (abs(width_rid - width_occ) < 1e-5 and rid < occupier):
                                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                                continue
                        
                        target_xy = cell_center(target_cell[0], target_cell[1], self.cell_size)
                        dist_to_wp = drive_robot(rid, target_xy)
                        if dist_to_wp < 0.22:
                            popped = state_info["waypoints"].pop(0)
                            if popped == state_info["target_box"]:
                                self.push_obs_pending[rid] = True
                            elif self.push_obs_pending[rid] and (len(state_info["waypoints"]) == 0):
                                self.broadcaster.broadcast_outcome(rid, success=True)
                                self.push_obs_pending[rid] = False
                                state_info["state"] = "NAVIGATING"
                                state_info["target_box"] = None
                            
                elif state_info["state"] == "YIELDING":
                    if state_info["waypoints"]:
                        target_cell = state_info["waypoints"][0]
                        target_xy = cell_center(target_cell[0], target_cell[1], self.cell_size)
                        dist_to_wp = drive_robot(rid, target_xy)
                        if dist_to_wp < 0.22:
                            state_info["waypoints"].pop(0)
                    else:
                        p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                        
                elif state_info["state"] == "WAITING":
                    p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                    
                else: # NAVIGATING
                    # Run standard A* internal
                    # Use sequential path planning avoiding teammate next cells
                    other_next_cells = [cell for other_rid, cell in planned_destinations.items() if other_rid != rid]
                    locked_boxes = set(self.active_clearings.keys()) | set(other_next_cells)
                    
                    path = a_star_internal(
                        curr_cell, g, other_rob_cells, box_cells, self.grid, self.grid_size,
                        ignore_boxes=False, locked_boxes=locked_boxes
                    )
                    
                    if len(path) > 1:
                        target_cell = path[1]
                        
                        # Check if target_cell is occupied by another robot
                        nxt_occupied = any(
                            current_robot_positions[r] == target_cell
                            for r in self.robot_ids if r != rid
                        )
                        if nxt_occupied:
                            occupier = next(
                                r for r in self.robot_ids
                                if r != rid and current_robot_positions[r] == target_cell
                            )
                            width_rid = self._get_sr_width(rid)
                            width_occ = self._get_sr_width(occupier)
                            if width_rid > width_occ or (abs(width_rid - width_occ) < 1e-5 and rid < occupier):
                                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                                continue
                                
                        planned_destinations[rid] = target_cell
                        target_xy = cell_center(target_cell[0], target_cell[1], self.cell_size)
                        drive_robot(rid, target_xy)
                    else:
                        # Drive directly to goal if inside goal cell
                        if curr_cell == g:
                            target_xy = cell_center(g[0], g[1], self.cell_size)
                            drive_robot(rid, target_xy)
                        else:
                            p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                            
            p.stepSimulation()
            if self.gui:
                import time
                time.sleep(0.01)
                
            # Accumulate collisions
            for i in range(len(self.robot_ids)):
                for j in range(i+1, len(self.robot_ids)):
                    if len(p.getContactPoints(self.robot_ids[i], self.robot_ids[j])) > 0:
                        collisions_this_step += 1
            for rid in self.robot_ids:
                for bid in self.box_ids:
                    if len(p.getContactPoints(rid, bid)) > 0:
                        collisions_this_step += 1
                        
        # End of control interval: compute reward and status
        obs = self._get_observations()
        
        rewards = {}
        dones = {}
        success = True
        
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = self.robot_goals[rid]
            dist = math.hypot(pos[0] - (g[0]+0.5)*self.cell_size, pos[1] - (g[1]+0.5)*self.cell_size)
            
            # Progress shaping: positive reward for moving closer, negative for moving further
            progress = starting_dists[rid] - dist
            
            # Step penalty + progress reward
            r = -0.1 + (1.5 * progress)
            
            # Collision penalty
            r -= 1.0 * collisions_this_step
            
            # Goal reward logic
            if dist <= 0.4:
                # If it wasn't done before, it just arrived
                if getattr(self, "agent_done_flags", {}).get(rid, False) == False:
                    r += 50.0
                    if not hasattr(self, "agent_done_flags"):
                        self.agent_done_flags = {}
                    self.agent_done_flags[rid] = True
                else:
                    # Already at goal, no more progress or step penalties, no goal farming
                    r = 0.0
                dones[rid] = True
            else:
                success = False
                dones[rid] = False
                if hasattr(self, "agent_done_flags"):
                    self.agent_done_flags[rid] = False
                
            rewards[rid] = r
            
        is_timeout = (self.current_step >= self.max_steps) and not success
        dones["__all__"] = success or is_timeout
        
        # Apply timeout penalty to agents that didn't reach the goal
        if is_timeout:
            for rid in self.robot_ids:
                if not dones[rid]:
                    rewards[rid] -= 50.0
        
        info = {"success": success, "collisions": collisions_this_step}
        
        return obs, rewards, dones, info
        
    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(physicsClientId=self.physics_client)
            except Exception:
                pass
            self.physics_client = None
