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

from simulation.snamo_simulator import SNAMOSimulator, world_to_cell
from decision.unet_decision_pipeline import UNetDecisionPipeline

class NAMOmappoEnv:
    def __init__(self, config_path, gui=False, max_steps=40, control_interval=30):
        self.config_path = config_path
        self.gui = gui
        self.max_steps = max_steps
        self.control_interval = control_interval
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.grid_size = self.config["world"]["grid_size"]
        self.cell_size = self.config["world"].get("cell_size", 1.0)
        
        # Load frozen UNet Pipeline
        project_dir = Path(__file__).resolve().parent.parent
        weights_path = str(project_dir / "models" / "namo_unet.pth")
        self.unet_pipeline = UNetDecisionPipeline(weights_path)
        
        self.sim = None
        self.current_step = 0
        self.agent_done_flags = {}
        
    def reset(self):
        if self.sim is not None:
            self.sim.close()
            
        self.sim = SNAMOSimulator(self.config_path, gui=self.gui, dr_strategy="sr_social")
        self.sim.reset()
        
        self.robot_ids = self.sim.robot_ids
        self.current_step = 0
        self.agent_done_flags = {rid: False for rid in self.robot_ids}
        
        return self._get_observations()
        
    def _get_unet_local_crop(self, agent_rid):
        """Generates 5x5 local risk crop from the UNet risk map around the agent's current position."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.sim.base_grid[r, c] == 1:
                    grid[r, c] = 1
        for bid in self.sim.box_ids:
            pos, _ = p.getBasePositionAndOrientation(bid)
            bx, by = world_to_cell(pos[:2], self.cell_size)
            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                grid[by, bx] = 2
        pos, _ = p.getBasePositionAndOrientation(agent_rid)
        ax, ay = world_to_cell(pos[:2], self.cell_size)
        g = self.sim.robot_goals[agent_rid]
        heatmap = self.unet_pipeline.get_risk_map(grid, (ax, ay), g)
        padded = np.pad(heatmap, pad_width=2, mode='constant', constant_values=1.0)
        crop = padded[ay : ay + 5, ax : ax + 5]
        return crop

    def _get_observations(self):
        obs = {}
        current_robot_positions = {}
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            current_robot_positions[rid] = pos[:2]
            
        current_box_positions = []
        for bid in self.sim.box_ids:
            pos, _ = p.getBasePositionAndOrientation(bid)
            current_box_positions.append(pos[:2])
            
        for rid in self.robot_ids:
            crop = self._get_unet_local_crop(rid).flatten()
            
            own_pos = current_robot_positions[rid]
            g = self.sim.robot_goals[rid]
            goal_pos = np.array([(g[0]+0.5)*self.cell_size, (g[1]+0.5)*self.cell_size])
            
            other_rob_pos = []
            for other_rid in self.robot_ids:
                if other_rid != rid:
                    other_rob_pos.append(current_robot_positions[other_rid])
            while len(other_rob_pos) < 3:
                other_rob_pos.append([0.0, 0.0])
            other_rob_pos = np.array(other_rob_pos).flatten()
            
            box_dists = []
            for bp in current_box_positions:
                box_dists.append((math.hypot(own_pos[0]-bp[0], own_pos[1]-bp[1]), bp))
            box_dists.sort(key=lambda x: x[0])
            
            nearby_boxes = [bp for _, bp in box_dists[:3]]
            while len(nearby_boxes) < 3:
                nearby_boxes.append([0.0, 0.0])
            nearby_boxes = np.array(nearby_boxes).flatten()
            
            sr_mean = self.sim.broadcaster.belief_models[rid].mean()
            sr_width = self.sim.broadcaster.get_sr_interval_width(rid)
            sr_obs = float(self.sim.broadcaster.belief_models[rid]._n_obs)
            norm_sr_obs = min(1.0, sr_obs / 20.0)
            uncertainty_feats = np.array([sr_mean, sr_width, norm_sr_obs], dtype=np.float32)
            
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
        self.current_step += 1
        
        starting_dists = {}
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = self.sim.robot_goals[rid]
            starting_dists[rid] = math.hypot(pos[0] - (g[0]+0.5)*self.cell_size, pos[1] - (g[1]+0.5)*self.cell_size)
            
        collisions_before = self.sim.collision_count
        
        # We step the simulator for `control_interval` physics steps
        # `control_interval` is 30, which exactly matches the 30-step decision period in `snamo_simulator.py`
        for _ in range(self.control_interval):
            self.sim.step(rl_actions=actions)
            
        collisions_this_step = self.sim.collision_count - collisions_before
        
        obs = self._get_observations()
        
        rewards = {}
        dones = {}
        
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = self.sim.robot_goals[rid]
            dist = math.hypot(pos[0] - (g[0]+0.5)*self.cell_size, pos[1] - (g[1]+0.5)*self.cell_size)
            
            progress = starting_dists[rid] - dist
            
            # Updated dense rewards per Phase B instructions
            # Step penalty: -0.01 (encourages speed)
            r = -0.01 + (1.5 * progress)
            
            if collisions_this_step == 0:
                # Safety bonus: +0.1 per step without collision
                r += 0.1
            else:
                r -= 1.0 * collisions_this_step
            
            if dist <= 0.4:
                if not self.agent_done_flags[rid]:
                    r += 1.0  # Goal arrival: +1.0
                    self.agent_done_flags[rid] = True
                else:
                    r = 0.0
                dones[rid] = True
            else:
                dones[rid] = False
                
            rewards[rid] = r
            
        is_timeout = (self.current_step >= self.max_steps) and not self.sim.success
        dones["__all__"] = self.sim.success or is_timeout
        
        if is_timeout:
            for rid in self.robot_ids:
                if not dones[rid]:
                    rewards[rid] -= 1.0 # scaled down penalty
        
        info = {"success": self.sim.success, "collisions": collisions_this_step}
        
        return obs, rewards, dones, info
        
    def close(self):
        if self.sim is not None:
            self.sim.close()
