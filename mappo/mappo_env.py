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

# ─── Reward shaping constants ─────────────────────────────────────────────────
STEP_PENALTY        = -0.01   # per-step cost to incentivize efficiency
COLLISION_PENALTY   = -0.10   # per robot-robot collision event
GOAL_BONUS          = 1.0     # reaching goal
TIMEOUT_PENALTY     = -1.0    # hitting max_steps without reaching goal
PROGRESS_REWARD     = 0.02    # per step moving closer to goal (dense)
PUSH_ATTEMPT_BONUS  = 0.05    # small bonus just for attempting PUSH (exploration)
PUSH_SUCCESS_BONUS  = 0.50    # bonus for making real progress during PUSH
PUSH_DIST_THRESHOLD = 0.10    # fraction of cell_size required for PUSH progress credit
# ──────────────────────────────────────────────────────────────────────────────


class NAMOmappoEnv:
    def __init__(self, config_path, gui=False, max_steps=40, control_interval=30):
        self.config_path = config_path
        self.gui = gui
        self.max_steps = max_steps
        self.control_interval = control_interval
        self.action_dim = 4  # NAVIGATE, PUSH, YIELD, WAIT

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.grid_size = self.config["world"]["grid_size"]
        self.cell_size = self.config["world"].get("cell_size", 1.0)

        # Detect whether this scenario has any movable obstacles
        self._has_boxes = len(self.config.get("obstacles", [])) > 0

        # Load frozen UNet Pipeline
        project_dir = Path(__file__).resolve().parent.parent
        weights_path = str(project_dir / "models" / "namo_unet.pth")
        self.unet_pipeline = UNetDecisionPipeline(weights_path)

        self.sim = None
        self.current_step = 0
        self.dr_strategy = "mappo"
        self.current_action_masks = {}
        self.current_costs = {}

    def reset(self):
        if self.sim is not None:
            self.sim.close()

        self.sim = SNAMOSimulator(self.config_path, gui=self.gui, dr_strategy="sr_social")
        self.sim.reset()

        self.robot_ids = self.sim.robot_ids
        self.current_step = 0
        self.agent_done_flags = {rid: False for rid in self.robot_ids}
        # Track previous distances for dense progress reward
        self._prev_dist = {}
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = self.sim.robot_goals[rid]
            self._prev_dist[rid] = math.hypot(
                pos[0] - (g[0] + 0.5) * self.cell_size,
                pos[1] - (g[1] + 0.5) * self.cell_size,
            )

        obs = self._get_observations()
        info = {"action_mask": self.current_action_masks}
        return obs, info

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
        crop = padded[ay: ay + 5, ax: ax + 5]
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
            goal_pos = np.array([(g[0] + 0.5) * self.cell_size, (g[1] + 0.5) * self.cell_size])

            other_rob_pos = []
            for other_rid in self.robot_ids:
                if other_rid != rid:
                    other_rob_pos.append(current_robot_positions[other_rid])
            while len(other_rob_pos) < 3:
                other_rob_pos.append([0.0, 0.0])
            other_rob_pos = np.array(other_rob_pos).flatten()

            box_dists = []
            for bp in current_box_positions:
                box_dists.append((math.hypot(own_pos[0] - bp[0], own_pos[1] - bp[1]), bp))
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
            norm_dir = (goal_pos - np.array(own_pos)) / (self.grid_size * self.cell_size)
            norm_others = other_rob_pos / (self.grid_size * self.cell_size)
            norm_boxes = nearby_boxes / (self.grid_size * self.cell_size)

            obs_vec = np.concatenate([
                crop,              # 25
                norm_own,          # 2
                norm_goal,         # 2
                norm_dir,          # 2
                norm_others,       # 6
                norm_boxes,        # 6
                uncertainty_feats, # 3
            ])
            obs[rid] = obs_vec.astype(np.float32)

            # ── Action masking ─────────────────────────────────────────────
            # Compute planner costs to assess feasibility
            planner = self.sim.snamo_planners[rid]
            total_cost, bypass_cost, removal_cost = planner.evaluate_actions(
                start=tuple(self.sim.coord_states[rid].cell),
                goal=tuple(self.sim.coord_states[rid].goal),
                box_cells=current_box_positions,
                other_robots=other_rob_pos.reshape(-1, 2).tolist() if len(other_rob_pos) > 0 else [],
                grid=self.sim.base_grid,
            )
            self.current_costs[rid] = total_cost

            # PUSH is valid if:
            #   (a) there are boxes in this scenario at all, AND
            #   (b) removal is physically possible (removal_cost != inf), OR
            #   (c) bypass is also inf (robot is trapped — PUSH is the only escape)
            if self._has_boxes:
                push_safe = (removal_cost != float('inf')) or (bypass_cost == float('inf'))
            else:
                push_safe = False   # no boxes in scenario — PUSH is meaningless

            yield_safe = bypass_cost != float('inf')
            self.current_action_masks[rid] = [True, push_safe, yield_safe, True]

        return obs

    def step(self, actions):
        self.current_step += 1

        # Save pre-step distances for dense progress reward
        starting_dists = {}
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = self.sim.robot_goals[rid]
            starting_dists[rid] = math.hypot(
                pos[0] - (g[0] + 0.5) * self.cell_size,
                pos[1] - (g[1] + 0.5) * self.cell_size,
            )

        collisions_before = self.sim.robot_robot_collisions

        # Step simulator for `control_interval` physics steps
        for _ in range(self.control_interval):
            self.sim.step(rl_actions=actions)

        collisions_this_step = self.sim.robot_robot_collisions - collisions_before

        obs = self._get_observations()
        rewards = {}
        dones = {}

        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = self.sim.robot_goals[rid]
            dist = math.hypot(
                pos[0] - (g[0] + 0.5) * self.cell_size,
                pos[1] - (g[1] + 0.5) * self.cell_size,
            )

            # ── Base step penalty ──────────────────────────────────────────
            r = STEP_PENALTY

            # ── Dense progress reward ──────────────────────────────────────
            # Reward proportional to how much closer we got to goal this step
            dist_delta = starting_dists[rid] - dist
            if dist_delta > 0:
                r += PROGRESS_REWARD * (dist_delta / self.cell_size)

            # ── Collision penalty ──────────────────────────────────────────
            if collisions_this_step > 0:
                r += COLLISION_PENALTY * collisions_this_step

            # ── PUSH-specific rewards ──────────────────────────────────────
            if actions.get(rid) == 1:  # PUSH action
                # Small bonus just for trying PUSH (exploration incentive)
                r += PUSH_ATTEMPT_BONUS
                # Larger bonus if we actually made meaningful progress
                threshold = PUSH_DIST_THRESHOLD * self.cell_size
                if dist_delta >= threshold:
                    r += PUSH_SUCCESS_BONUS

            # ── YIELD reward ───────────────────────────────────────────────
            elif actions.get(rid) == 2:  # YIELD
                # Yielding incurs the standard step penalty; no extra bonuses to avoid positive loops.
                pass

            # ── Goal bonus ─────────────────────────────────────────────────
            if dist <= 0.6:
                if not self.agent_done_flags[rid]:
                    r += GOAL_BONUS
                    self.agent_done_flags[rid] = True
                else:
                    r = 0.0
                dones[rid] = True
            else:
                dones[rid] = False

            rewards[rid] = r
            self._prev_dist[rid] = dist

        is_timeout = (self.current_step >= self.max_steps) and not self.sim.success
        dones["__all__"] = self.sim.success or is_timeout

        if is_timeout:
            for rid in self.robot_ids:
                if not dones[rid]:
                    rewards[rid] += TIMEOUT_PENALTY

        info = {
            "success": self.sim.success,
            "collisions": collisions_this_step,
            "action_mask": self.current_action_masks,
        }

        return obs, rewards, dones, info

    def close(self):
        if self.sim is not None:
            self.sim.close()
