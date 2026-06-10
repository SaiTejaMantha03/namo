"""
multi_robot/coordinator.py
----------------------------
Per-robot coordination loop: Algorithm 1 from MR-NAMO (Paper 5).

Each robot runs RobotCoordinator.step() independently at every timestep.
The coordinator:
  1. Checks if goal is reached.
  2. Replans if no current plan or plan head diverged.
  3. Detects conflicts in the next h steps.
  4. Resolves deadlocks using the chosen DR strategy.
  5. Otherwise, executes the next action on the plan.
"""

from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from core.planner import a_star, find_clearing_direction
from multi_robot.conflict_detection import ConflictDetector, ConflictType

if TYPE_CHECKING:
    from multi_robot.deadlock_resolution import DeadlockResolver
    from multi_robot.belief_broadcaster import BeliefBroadcaster
    from social.social_costmap import SocialCostmap


@dataclass
class RobotState:
    """All mutable state for a single robot across timesteps."""
    robot_id: int
    cell: tuple[int, int]           # current (col, row)
    goal: tuple[int, int]           # target (col, row)
    plan: list[tuple] = field(default_factory=list)
    status: str = "NAVIGATING"      # NAVIGATING | WAITING | EVADING | DONE
    active_box: Optional[int] = None
    planned_obstacle_cell: Optional[tuple] = None
    wait_ticks: int = 0
    evasion_target: Optional[tuple] = None
    _stuck_counter: int = 0         # steps without progress
    _last_cell: Optional[tuple] = None


class RobotCoordinator:
    """
    Implements Algorithm 1 from MR-NAMO (Paper 5).

    Parameters
    ----------
    grid           : 2-D numpy occupancy grid (0=free, 1=wall, 2=box).
    grid_size      : cells per side.
    h              : conflict lookahead horizon (steps).
    dr_strategy    : "repulsive" | "social" | "sr_width".
    resolver       : DeadlockResolver instance.
    broadcaster    : BeliefBroadcaster (needed for sr_width strategy).
    social_map     : SocialCostmap (needed for social strategy).
    stuck_limit    : steps without progress before forced replan (default 8).
    """

    def __init__(
        self,
        grid: np.ndarray,
        grid_size: int,
        h: int = 10,
        dr_strategy: str = "repulsive",
        resolver: Optional["DeadlockResolver"] = None,
        broadcaster: Optional["BeliefBroadcaster"] = None,
        social_map: Optional["SocialCostmap"] = None,
        stuck_limit: int = 8,
    ):
        self.grid = grid
        self.grid_size = grid_size
        self.h = h
        self.dr_strategy = dr_strategy
        self.resolver = resolver
        self.broadcaster = broadcaster
        self.social_map = social_map
        self.stuck_limit = stuck_limit
        self._detector = ConflictDetector()

    # ------------------------------------------------------------------
    # Main per-timestep step
    # ------------------------------------------------------------------

    def step(
        self,
        states: dict[int, RobotState],
        box_states: dict[int, tuple],
    ) -> dict[int, tuple]:
        """
        Run one timestep for ALL robots simultaneously.
        Returns next_cells: {robot_id: (col, row)}.
        """
        next_cells: dict[int, tuple] = {}

        # ---- 1. DONE and WAITING robots --------------------------------
        for rid, state in states.items():
            if state.status == "DONE":
                next_cells[rid] = state.cell
                continue
            if state.status == "WAITING":
                state.wait_ticks -= 1
                if state.wait_ticks <= 0:
                    state.status = "NAVIGATING"
                    state.plan = []   # force replan after wait
                next_cells[rid] = state.cell

        active_ids = [rid for rid in states
                      if states[rid].status not in ("DONE", "WAITING")]

        if not active_ids:
            return next_cells

        # ---- 2. Stuck detection + forced replan -----------------------
        for rid in active_ids:
            state = states[rid]
            if state._last_cell == state.cell:
                state._stuck_counter += 1
            else:
                state._stuck_counter = 0
            state._last_cell = state.cell

            # Force replan if stuck too long; also clear EVADING so robot
            # re-enters normal navigation after evasion completes.
            needs_replan = (
                not state.plan
                or (state.plan and state.plan[0] != state.cell)
                or state._stuck_counter >= self.stuck_limit
                or (state.status == "EVADING" and state.cell == state.evasion_target)
            )

            if needs_replan:
                if state._stuck_counter >= self.stuck_limit:
                    state._stuck_counter = 0
                if state.status == "EVADING":
                    state.status = "NAVIGATING"
                    state.evasion_target = None

                other_rob_cells = [states[r].cell for r in states if r != rid]
                box_cells_list = list(box_states.values())
                plan = a_star(
                    state.cell, state.goal,
                    self.grid, self.grid_size,
                    other_robots=other_rob_cells,
                    blocked_cells=box_cells_list,
                )
                state.plan = plan if plan else [state.cell]

        # ---- 3. Build plans for conflict detector ----------------------
        robot_plans = {rid: states[rid].plan[:self.h] for rid in active_ids}

        robot_states_cd = {
            rid: {
                "cell": states[rid].cell,
                "active_box": states[rid].active_box,
                "planned_obstacle_cell": states[rid].planned_obstacle_cell,
            }
            for rid in active_ids
        }

        # ---- 4. Detect conflicts ----------------------------------------
        conflicts = self._detector.detect(
            robot_plans, box_states, robot_states_cd, h=self.h
        )

        # ---- 4b. Head-on corridor deadlock: back-up maneuver -----------
        # When two robots are heading directly toward each other in a narrow
        # corridor (no side escape), the one with higher robot_id backs up
        # a few steps to let the other pass through.
        headon_pairs: set[tuple] = set()
        for c in conflicts:
            from multi_robot.conflict_detection import ConflictType as CT
            if c.conflict_type == CT.ROBOT_ROBOT and len(c.robots_involved) == 2:
                r1, r2 = c.robots_involved
                if r1 in active_ids and r2 in active_ids:
                    s1, s2 = states[r1], states[r2]
                    if len(s1.plan) > 1 and len(s2.plan) > 1:
                        # Check they are heading toward each other
                        dir1 = (s1.plan[1][0]-s1.cell[0], s1.plan[1][1]-s1.cell[1])
                        dir2 = (s2.plan[1][0]-s2.cell[0], s2.plan[1][1]-s2.cell[1])
                        dot = dir1[0]*dir2[0] + dir1[1]*dir2[1]
                        if dot < 0:   # opposite directions → head-on
                            headon_pairs.add((min(r1,r2), max(r1,r2)))

        for r1, r2 in headon_pairs:
            # Robot with higher id backs up to let the other pass
            backer = max(r1, r2)
            state  = states[backer]
            # Back-up: move opposite to our current plan direction for a few steps
            if len(state.plan) >= 2:
                curr = state.cell
                nxt = state.plan[1]
                dx = curr[0] - nxt[0]
                dy = curr[1] - nxt[1]
                
                # Generate a short backtrack path
                backtrack = [curr]
                c = curr
                for _ in range(3):
                    c = (c[0] + dx, c[1] + dy)
                    if 0 <= c[0] < self.grid_size and 0 <= c[1] < self.grid_size and self.grid[c[1], c[0]] == 0:
                        backtrack.append(c)
                    else:
                        break
                
                # Stay at the backed-up position for a few ticks to let the other pass
                if len(backtrack) > 1:
                    last = backtrack[-1]
                    backtrack.extend([last] * 5)
                    state.plan = backtrack
                    state.status = "EVADING"

        # ---- 5. Resolve deadlocks ---------------------------------------
        resolved_rids: set[int] = set()

        if conflicts and self.resolver:
            conflict_robot_sets = [set(c.robots_involved) for c in conflicts]
            merged: set = set()
            for s in conflict_robot_sets:
                merged |= s
            involved = [r for r in merged if r in active_ids]

            if len(involved) >= 2:
                robot_cells_d = {rid: states[rid].cell for rid in involved}
                robot_goals_d = {rid: states[rid].goal for rid in involved}

                if self.dr_strategy == "social" and self.social_map:
                    assignments, evasion_targets = self.resolver.resolve_social(
                        involved, robot_cells_d, robot_goals_d, self.social_map
                    )
                elif self.dr_strategy == "sr_width" and self.broadcaster:
                    assignments, evasion_targets = self.resolver.resolve_sr_width(
                        involved, robot_cells_d, robot_goals_d, self.broadcaster
                    )
                else:
                    assignments, evasion_targets = self.resolver.resolve_repulsive(
                        involved, robot_cells_d, robot_goals_d
                    )

                for rid, action in assignments.items():
                    state = states[rid]
                    if action == "WAIT":
                        state.status = "WAITING"
                        state.wait_ticks = random.randint(1, 3)
                        next_cells[rid] = state.cell
                        resolved_rids.add(rid)
                    elif action == "EVADE":
                        state.status = "EVADING"
                        target = evasion_targets.get(rid)
                        state.evasion_target = target
                        if target:
                            other_rob_cells = [states[r].cell for r in states if r != rid]
                            evade_plan = a_star(
                                state.cell, target,
                                self.grid, self.grid_size,
                                other_robots=other_rob_cells,
                            )
                            state.plan = evade_plan if evade_plan else [state.cell]
                        # Fall through — let step 6 advance the plan this tick

        # ---- 6. Execute next step for all active robots ----------------
        occupied_next: dict[int, tuple] = {}   # tracks where robots are moving

        for rid in active_ids:
            if rid in resolved_rids:
                continue   # WAIT robots already assigned above

            state = states[rid]

            # Goal check
            if state.cell == state.goal:
                state.status = "DONE"
                next_cells[rid] = state.cell
                continue

            if len(state.plan) > 1:
                next_cell = state.plan[1]
                # Collision avoidance: don't step into another robot's current cell
                currently_occupied = {states[r].cell for r in states if r != rid}
                if next_cell not in currently_occupied and next_cell not in occupied_next.values():
                    state.plan.pop(0)
                    state.cell = next_cell
            # else: at end of plan, stay put (will replan next tick)

            next_cells[rid] = state.cell
            occupied_next[rid] = state.cell

        return next_cells
