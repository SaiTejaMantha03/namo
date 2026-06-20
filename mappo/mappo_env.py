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
STEP_PENALTY          = -0.01   # per-step cost to incentivize efficiency
COLLISION_PENALTY     = -0.10   # per robot-robot collision event
GOAL_BONUS            = 1.0     # reaching goal
TIMEOUT_PENALTY       = -1.0    # hitting max_steps without reaching goal
PROGRESS_REWARD       = 0.02    # per step moving closer to goal (dense)
PUSH_ATTEMPT_BONUS    = 0.05    # small bonus just for attempting PUSH (exploration)
PUSH_SUCCESS_BONUS    = 0.50    # bonus for making real progress during PUSH
PUSH_DIST_THRESHOLD   = 0.10    # fraction of cell_size required for PUSH progress credit

# ─── NEW: Congestion / deadlock reward signals ────────────────────────────────
DEADLOCK_PENALTY      = -0.05   # per step when robot is stuck AND nearby robots also stuck
DEADLOCK_RESOLVE_BONUS = 0.30   # when a robot breaks out of a mutual-wait state
YIELD_PENALTY         = -0.005  # slight nudge against indefinite yielding
WAIT_PENALTY          = -0.02   # v4: penalise WAIT to prevent collapse on hard stages
# ──────────────────────────────────────────────────────────────────────────────

# Deadlock detection threshold: if robot moved less than this (world units) in
# the last N steps we consider it "stuck"
STUCK_DIST_THRESHOLD = 0.15  # world units
STUCK_WINDOW         = 8     # number of control steps to look back


class NAMOmappoEnv:
    def __init__(self, config_path, gui=False, max_steps=40, control_interval=30, randomize_starts=False, include_congestion_feats=True):
        self.config_path = config_path
        self.gui = gui
        self.max_steps = max_steps
        self.control_interval = control_interval
        self.randomize_starts = randomize_starts
        self.include_congestion_feats = include_congestion_feats
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

        # ── NEW: per-robot state trackers for congestion features ─────────────
        # wait_time: how many consecutive control steps the robot has been stuck
        self._wait_time = {}
        # position history for stuck detection
        self._pos_history = {}
        # was the robot stuck last step (for resolve bonus)
        self._was_stuck = {}

    def reset(self):
        if self.sim is not None:
            self.sim.close()

        self.sim = SNAMOSimulator(self.config_path, gui=self.gui, dr_strategy="sr_social")

        # Randomize starts if requested to break symmetry
        if self.randomize_starts and hasattr(self.sim, "robots_cfg"):
            import random
            cfg_name = Path(self.config_path).name
            if "narrow_doorway_congestion" in cfg_name:
                for rcfg in self.sim.robots_cfg:
                    rcfg["start"] = [random.randint(1, 4), rcfg["start"][1]]
            elif "symmetric_bottleneck_4robots" in cfg_name or "cross_intersection_coordination" in cfg_name:
                for rcfg in self.sim.robots_cfg:
                    sx, sy = rcfg["start"]
                    if sx < 5 and sy == 7:
                        rcfg["start"] = [random.randint(1, 4), 7]
                    elif sx > 10 and sy == 7:
                        rcfg["start"] = [random.randint(11, 13), 7]
                    elif sx == 7 and sy < 5:
                        rcfg["start"] = [7, random.randint(1, 4)]
                    elif sx == 7 and sy > 10:
                        rcfg["start"] = [7, random.randint(11, 13)]

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

        # ── NEW: reset congestion trackers ────────────────────────────────────
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            self._wait_time[rid] = 0
            self._pos_history[rid] = [pos[:2]] * STUCK_WINDOW
            self._was_stuck[rid] = False

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

    def _is_stuck(self, rid) -> bool:
        """Return True if this robot has barely moved over the last STUCK_WINDOW steps."""
        hist = self._pos_history[rid]
        if len(hist) < 2:
            return False
        oldest = hist[0]
        newest = hist[-1]
        return math.hypot(newest[0] - oldest[0], newest[1] - oldest[1]) < STUCK_DIST_THRESHOLD

    def _nearby_robot_count(self, rid, current_positions, radius=5.0) -> int:
        """Number of other robots within `radius` world units."""
        own = current_positions[rid]
        count = 0
        for other_rid, pos in current_positions.items():
            if other_rid == rid:
                continue
            if math.hypot(pos[0] - own[0], pos[1] - own[1]) <= radius:
                count += 1
        return count

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

        max_wait = float(self.max_steps)   # normalisation denominator
        max_nearby = float(max(len(self.robot_ids) - 1, 1))

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

            norm_own   = np.array(own_pos) / (self.grid_size * self.cell_size)
            norm_goal  = goal_pos / (self.grid_size * self.cell_size)
            norm_dir   = (goal_pos - np.array(own_pos)) / (self.grid_size * self.cell_size)
            norm_others = other_rob_pos / (self.grid_size * self.cell_size)
            norm_boxes  = nearby_boxes / (self.grid_size * self.cell_size)

            # ── NEW: congestion features ──────────────────────────────────────
            wait_norm = min(1.0, self._wait_time.get(rid, 0) / max_wait)
            nearby_cnt = self._nearby_robot_count(rid, current_robot_positions)
            nearby_norm = float(nearby_cnt) / max_nearby
            stuck_flag = float(self._is_stuck(rid))
            congestion_feats = np.array([wait_norm, nearby_norm, stuck_flag], dtype=np.float32)
            # ─────────────────────────────────────────────────────────────────

            obs_parts = [
                crop,              # 25
                norm_own,          # 2
                norm_goal,         # 2
                norm_dir,          # 2
                norm_others,       # 6
                norm_boxes,        # 6
                uncertainty_feats, # 3
            ]
            if self.include_congestion_feats:
                obs_parts.append(congestion_feats)
            obs_vec = np.concatenate(obs_parts)
            obs[rid] = obs_vec.astype(np.float32)

            # ── Action masking ─────────────────────────────────────────────
            planner = self.sim.snamo_planners[rid]
            total_cost, bypass_cost, removal_cost = planner.evaluate_actions(
                start=tuple(self.sim.coord_states[rid].cell),
                goal=tuple(self.sim.coord_states[rid].goal),
                box_cells=current_box_positions,
                other_robots=other_rob_pos.reshape(-1, 2).tolist() if len(other_rob_pos) > 0 else [],
                grid=self.sim.base_grid,
            )
            self.current_costs[rid] = total_cost

            if self._has_boxes:
                push_safe = (removal_cost != float('inf')) or (bypass_cost == float('inf'))
            else:
                push_safe = False

            yield_safe = len(self.robot_ids) > 1
            self.current_action_masks[rid] = [True, push_safe, yield_safe, True]

        return obs

    def step(self, actions):
        self.current_step += 1

        # Save pre-step distances for dense progress reward
        starting_dists = {}
        starting_positions = {}
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            g = self.sim.robot_goals[rid]
            starting_dists[rid] = math.hypot(
                pos[0] - (g[0] + 0.5) * self.cell_size,
                pos[1] - (g[1] + 0.5) * self.cell_size,
            )
            starting_positions[rid] = pos[:2]

        collisions_before = self.sim.robot_robot_collisions

        # Step simulator for `control_interval` physics steps
        for _ in range(self.control_interval):
            self.sim.step(rl_actions=actions)

        collisions_this_step = self.sim.robot_robot_collisions - collisions_before

        # ── NEW: update congestion trackers ───────────────────────────────────
        current_positions_post = {}
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            current_positions_post[rid] = pos[:2]
            hist = self._pos_history.get(rid, [pos[:2]] * STUCK_WINDOW)
            hist.append(pos[:2])
            if len(hist) > STUCK_WINDOW:
                hist.pop(0)
            self._pos_history[rid] = hist

        # Update wait_time: increment if stuck, reset if moved meaningfully
        for rid in self.robot_ids:
            if self._is_stuck(rid):
                self._wait_time[rid] = self._wait_time.get(rid, 0) + 1
            else:
                self._wait_time[rid] = 0

        # Count how many robots are stuck right now (for deadlock detection)
        stuck_robots = {rid for rid in self.robot_ids if self._is_stuck(rid)}
        # ─────────────────────────────────────────────────────────────────────

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
            dist_delta = starting_dists[rid] - dist
            if dist_delta > 0:
                r += PROGRESS_REWARD * (dist_delta / self.cell_size)

            # ── Collision penalty ──────────────────────────────────────────
            if collisions_this_step > 0:
                r += COLLISION_PENALTY * collisions_this_step

            # ── PUSH-specific rewards ──────────────────────────────────────
            if actions.get(rid) == 1:  # PUSH action
                r += PUSH_ATTEMPT_BONUS
                threshold = PUSH_DIST_THRESHOLD * self.cell_size
                if dist_delta >= threshold:
                    r += PUSH_SUCCESS_BONUS

            # ── YIELD penalty (discourage permanent yielding) ──────────────
            elif actions.get(rid) == 2:  # YIELD
                r += YIELD_PENALTY

            # ── WAIT penalty (v4: prevent indefinite waiting on hard stages) ─
            elif actions.get(rid) == 3:  # WAIT
                r += WAIT_PENALTY

            # ── NEW: Deadlock / congestion reward signals ──────────────────
            is_stuck_now = rid in stuck_robots
            was_stuck    = self._was_stuck.get(rid, False)
            nearby_stuck = sum(1 for r2 in stuck_robots if r2 != rid)

            # Penalty: stuck and at least one other nearby robot is also stuck
            if is_stuck_now and nearby_stuck > 0:
                r += DEADLOCK_PENALTY

            # Bonus: robot just broke out of a deadlock (was stuck, now moving)
            if was_stuck and not is_stuck_now and dist_delta > 0:
                r += DEADLOCK_RESOLVE_BONUS

            self._was_stuck[rid] = is_stuck_now
            # ─────────────────────────────────────────────────────────────

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
            "stuck_count": len(stuck_robots),
        }

        return obs, rewards, dones, info

    def close(self):
        if self.sim is not None:
            self.sim.close()
