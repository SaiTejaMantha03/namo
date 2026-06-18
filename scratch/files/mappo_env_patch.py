"""
mappo_env_patch.py
==================
NOT a standalone file. This is a surgical patch showing exactly which
methods to change in your existing mappo/mappo_env.py.

Search for each section header and replace the corresponding method.
"""

# ════════════════════════════════════════════════════════════════════════════
# PATCH 1 — action masking via _bypass_interval / _removal_interval
# Add this as a NEW method inside your MANAMOEnv class.
# ════════════════════════════════════════════════════════════════════════════

def _compute_action_mask(self, robot_idx: int) -> list[bool]:
    """
    Returns a 4-element boolean mask [NAVIGATE, PUSH, YIELD, WAIT].

    PUSH is masked (False) when _removal_interval returns (inf, inf),
    meaning S-NAMO judged the removal geometrically impossible.

    YIELD is masked (False) when _bypass_interval returns (inf, inf),
    meaning the bypass path is blocked entirely.

    NAVIGATE and WAIT are always available.

    Access pattern: self.sim.coordinator.planner
    Adjust the attribute chain to match your actual object graph.
    """
    NAVIGATE = 0
    PUSH     = 1
    YIELD    = 2
    WAIT     = 3

    mask = [True, True, True, True]   # start permissive

    try:
        planner     = self.sim.coordinator.planner   # SNAMOPlanner instance
        robot_state = self.sim.robots[robot_idx]     # whatever state object planner expects
        grid        = self.sim.grid                  # current occupancy grid

        # --- PUSH mask ---
        # _removal_interval returns (lo, hi) path cost bounds.
        # If lo == inf the removal is impossible; mask it.
        removal_lo, removal_hi = planner._removal_interval(robot_state, grid)
        if removal_lo == float('inf'):
            mask[PUSH] = False

        # --- YIELD mask ---
        # _bypass_interval returns (lo, hi) bypass cost bounds.
        # If lo == inf the bypass path is completely blocked; mask it.
        bypass_lo, bypass_hi = planner._bypass_interval(robot_state, grid)
        if bypass_lo == float('inf'):
            mask[YIELD] = False

    except AttributeError as e:
        # Planner not accessible in this env configuration;
        # fall back to fully permissive mask and log a warning.
        import warnings
        warnings.warn(
            f"_compute_action_mask: could not reach planner ({e}). "
            "Returning fully permissive mask."
        )

    return mask   # e.g. [True, False, True, True]


# ════════════════════════════════════════════════════════════════════════════
# PATCH 2 — reward reshaping
# Replace the reward section inside your existing step() method.
# ════════════════════════════════════════════════════════════════════════════

def _compute_reward(
    self,
    robot_idx: int,
    action: int,
    prev_dist_to_goal: float,
    curr_dist_to_goal: float,
    goal_reached: bool,
    collision_this_step: bool,
) -> tuple[float, dict]:
    """
    Reward function.

    Returns (reward_scalar, info_dict).
    info_dict is logged so you can verify each component in training.

    Components
    ----------
    +1.00   goal reached
    -0.01   per-step living penalty (encourages speed)
    -0.05   collision this step (recoverable, NOT fatal)
    +0.50   productive PUSH or YIELD  ← only if distance decreased
    """
    PUSH  = 1
    YIELD = 2

    reward     = 0.0
    components = {}

    # --- goal bonus ---
    if goal_reached:
        reward += 1.0
        components["goal"] = 1.0

    # --- step penalty ---
    reward -= 0.01
    components["step_penalty"] = -0.01

    # --- collision penalty (reduced from -1.0 to -0.05) ---
    if collision_this_step:
        reward -= 0.05
        components["collision"] = -0.05

    # --- productive push/yield bonus ---
    # "Productive" = the robot is measurably closer to its goal.
    # We use Euclidean distance as a fast proxy for path cost.
    # The threshold (0.5 grid cells) is deliberately small so the robot
    # must make real progress, not just wiggle.
    PROGRESS_THRESHOLD = 0.5   # in grid-cell units; tune if needed

    if action in (PUSH, YIELD):
        dist_delta = prev_dist_to_goal - curr_dist_to_goal
        if dist_delta >= PROGRESS_THRESHOLD:
            reward += 0.5
            components["push_yield_bonus"] = 0.5
        else:
            components["push_yield_bonus"] = 0.0   # no bonus for idle pushes

    return reward, components


# ════════════════════════════════════════════════════════════════════════════
# PATCH 3 — step() changes (show what to add/change, not the full method)
# ════════════════════════════════════════════════════════════════════════════

# INSIDE step(), BEFORE calling _compute_reward:
# --------------------------------------------------
#   prev_dist = math.hypot(
#       robot.pos.x - robot.goal.x,
#       robot.pos.y - robot.goal.y
#   )
#
# AFTER the physics step runs:
#   curr_dist = math.hypot(
#       robot.pos.x - robot.goal.x,
#       robot.pos.y - robot.goal.y
#   )
#
#   collision_this_step = self.sim.robot_robot_collisions   # your existing variable
#
#   reward, reward_info = self._compute_reward(
#       robot_idx         = i,
#       action            = actions[i],
#       prev_dist_to_goal = prev_dist,
#       curr_dist_to_goal = curr_dist,
#       goal_reached      = done[i],
#       collision_this_step = collision_this_step,
#   )


# ════════════════════════════════════════════════════════════════════════════
# PATCH 4 — return action_mask in info dict from reset() and step()
# ════════════════════════════════════════════════════════════════════════════

# In reset():
# --------------------------------------------------
#   obs, info = ...existing reset logic...
#   info["action_mask"] = [
#       self._compute_action_mask(i) for i in range(self.n_robots)
#   ]
#   return obs, info

# In step():
# --------------------------------------------------
#   info["action_mask"] = [
#       self._compute_action_mask(i) for i in range(self.n_robots)
#   ]
#   return obs, rewards, dones, info
