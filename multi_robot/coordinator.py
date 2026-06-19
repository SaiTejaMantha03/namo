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
import math
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
    priority_score: float = 1.0     # Novel Phase 2A: inherited priority
    _stuck_counter: int = 0         # steps without progress
    _last_cell: Optional[tuple] = None
    waiting_for_robot: Optional[int] = None
    _post_evade_grace: int = 0


class RobotCoordinator:
    """
    Implements Algorithm 1 from MR-NAMO (Paper 5) with three novel extensions:

      Phase 2A — Priority Inheritance:
        When robot A waits for robot B, B inherits A's priority_score so
        lower-priority robots cannot sneak ahead during convoy scenarios.

      Phase 2B — resolve_sr_social():
        A new DR strategy combining SR-Width priority with social-cost-aware
        evasion target selection. Exposed as dr_strategy="sr_social".

      Phase 2C — Adaptive Conflict Horizon:
        h = clamp(ceil(mean_pairwise_dist / 2), 5, 20), recomputed each tick.

    Parameters
    ----------
    grid           : 2-D numpy occupancy grid (0=free, 1=wall, 2=box).
    grid_size      : cells per side.
    h              : base conflict lookahead horizon (overridden by adaptive h).
    dr_strategy    : "repulsive" | "social" | "sr_width" | "sr_social".
    resolver       : DeadlockResolver instance.
    broadcaster    : BeliefBroadcaster (needed for sr_width / sr_social).
    social_map     : SocialCostmap (needed for social / sr_social strategy).
    stuck_limit    : steps without progress before forced replan (default 8).
    adaptive_h     : if True, recompute h each tick from pairwise distances.
    priority_decay : fraction by which inherited priority decays per tick.
    """

    def __init__(
        self,
        grid: np.ndarray,
        grid_size: int,
        h: int = 10,
        dr_strategy: str = "sr_social",
        resolver: Optional["DeadlockResolver"] = None,
        broadcaster: Optional["BeliefBroadcaster"] = None,
        social_map: Optional["SocialCostmap"] = None,
        stuck_limit: int = 8,
        adaptive_h: bool = True,
        priority_decay: float = 0.95,
    ):
        self.grid = grid
        self.grid_size = grid_size
        self.base_h = h
        self.dr_strategy = dr_strategy
        self.resolver = resolver
        self.broadcaster = broadcaster
        self.social_map = social_map
        self.stuck_limit = stuck_limit
        self.adaptive_h = adaptive_h
        self.priority_decay = priority_decay
        self._detector = ConflictDetector()
        self.deadlock_count = 0
        self.yield_sr_widths = []

    # ------------------------------------------------------------------
    # Phase 2C: Adaptive conflict horizon
    # ------------------------------------------------------------------

    def _compute_adaptive_h(self, states: dict[int, RobotState]) -> int:
        """h = clamp(ceil(mean_pairwise_Manhattan / 2), 5, 20)."""
        active = [s for s in states.values() if s.status != "DONE"]
        if len(active) < 2:
            return self.base_h
        dists = []
        cells = [s.cell for s in active]
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                dists.append(
                    abs(cells[i][0] - cells[j][0]) + abs(cells[i][1] - cells[j][1])
                )
        mean_dist = sum(dists) / len(dists)
        return max(5, min(20, math.ceil(mean_dist / 2)))

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

        # Decrement post-evasion grace counters
        for state in states.values():
            if state._post_evade_grace > 0:
                state._post_evade_grace -= 1

        # Phase 2C: adaptive horizon
        h = self._compute_adaptive_h(states) if self.adaptive_h else self.base_h

        # Phase 2A: decay inherited priorities each tick
        for state in states.values():
            if state.priority_score > 1.0:
                state.priority_score = max(1.0, state.priority_score * self.priority_decay)

        # ---- 1. DONE and WAITING robots --------------------------------
        for rid, state in states.items():
            if state.status == "DONE":
                next_cells[rid] = state.cell
                continue
            if state.status in ("WAITING", "POCKET_WAITING"):
                other_evading = False
                if getattr(state, "waiting_for_robot", None) in states:
                    other_state = states[state.waiting_for_robot]
                    
                    if state.status == "POCKET_WAITING":
                        if other_state.status == "DONE":
                            # The other robot is done! We can resume immediately.
                            state.status = "NAVIGATING"
                            state.plan = []
                            state.waiting_for_robot = None
                            next_cells[rid] = state.cell
                            continue
                        else:
                            # Check if other robot has passed the pocket cell using projection
                            my_cell = state.cell
                            other_cell = other_state.cell
                            other_goal = other_state.goal
                            
                            dist = abs(other_cell[0] - my_cell[0]) + abs(other_cell[1] - my_cell[1])
                            if dist <= 1:
                                other_evading = True
                            else:
                                dir_goal = (other_goal[0] - other_cell[0], other_goal[1] - other_cell[1])
                                dir_pocket = (my_cell[0] - other_cell[0], my_cell[1] - other_cell[1])
                                dot = dir_goal[0] * dir_pocket[0] + dir_goal[1] * dir_pocket[1]
                                if dot > 0:
                                    other_evading = True
                                else:
                                    # The other robot has passed us! We can resume immediately.
                                    state.status = "NAVIGATING"
                                    state.plan = []
                                    state.waiting_for_robot = None
                                    next_cells[rid] = state.cell
                                    continue
                    else:
                        # Normal WAITING robot: wait until the evader is far from intersection
                        if other_state.status == "EVADING":
                            other_evading = True
                        elif other_state.status == "NAVIGATING":
                            my_cell = state.cell
                            other_cell = other_state.cell
                            dist = abs(other_cell[0] - my_cell[0]) + abs(other_cell[1] - my_cell[1])
                            if dist <= 8:
                                other_evading = True  # still in intersection zone
                
                if not other_evading:
                    state.wait_ticks -= 1
                    if state.wait_ticks <= 0:
                        state.status = "NAVIGATING"
                        state.plan = []   # force replan after wait
                        state.waiting_for_robot = None
                next_cells[rid] = state.cell

        active_ids = [rid for rid in states
                      if states[rid].status not in ("DONE", "WAITING", "POCKET_WAITING")]

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
            diverged = False
            if state.plan:
                dx = abs(state.plan[0][0] - state.cell[0])
                dy = abs(state.plan[0][1] - state.cell[1])
                if dx + dy > 1:
                    diverged = True

            needs_replan = (
                not state.plan
                or diverged
                or state._stuck_counter >= self.stuck_limit
                or (state.status == "EVADING" and state.cell == state.evasion_target)
                or (state.status == "EVADING" and state.evasion_target
                    and (not state.plan or state.plan[-1] != state.evasion_target))
            )

            if needs_replan:
                if state._stuck_counter >= self.stuck_limit:
                    state._stuck_counter = 0
                if state.status == "EVADING" and state.cell == state.evasion_target:
                    # Transition to WAITING in the pocket, waiting for the waiter to pass
                    waiter_id = None
                    for other_rid, other_state in states.items():
                        if other_rid != rid and getattr(other_state, "waiting_for_robot", None) == rid:
                            waiter_id = other_rid
                            break
                    if waiter_id is not None:
                        state.status = "POCKET_WAITING"
                        state.waiting_for_robot = waiter_id
                        state.wait_ticks = 100
                        state.evasion_target = None
                    else:
                        state.status = "NAVIGATING"
                        state.evasion_target = None
                        state._post_evade_grace = 50
                elif state.status == "EVADING":
                    if state.evasion_target:
                        other_rob_cells = [
                            states[r].cell for r in states
                            if r != rid and states[r].status not in ("WAITING", "POCKET_WAITING")
                        ]
                        evade_plan = a_star(
                            state.cell, state.evasion_target,
                            self.grid, self.grid_size,
                            other_robots=other_rob_cells,
                        )
                        state.plan = evade_plan if evade_plan else [state.cell]
                    else:
                        state.status = "NAVIGATING"
                        state._post_evade_grace = 50

                if state.status not in ("EVADING", "POCKET_WAITING", "WAITING"):
                    done_cells = [
                        states[r].cell for r in states
                        if r != rid and states[r].status == "DONE"
                    ]
                    other_active_cells = [
                        states[r].cell for r in states
                        if r != rid and states[r].status != "DONE"
                    ]
                    box_cells_list = list(box_states.values())

                    # Key insight: in tight single-width corridors with no boxes,
                    # blocking other robots as hard walls leaves A* with NO PATH
                    # (the only route goes through the other robot's cell).
                    # The conflict detector + DR handle actual avoidance — the
                    # planner just needs a valid route for the detector to inspect.
                    # With boxes present, we still block others to avoid pushing them.
                    plan = a_star(
                        state.cell, state.goal,
                        self.grid, self.grid_size,
                        other_robots=[],
                        blocked_cells=done_cells,
                        ignore_boxes=True,
                    )
                    state.plan = plan if plan else [state.cell]

        # ---- 3. Build plans for conflict detector ----------------------
        robot_plans = {rid: states[rid].plan[:h] for rid in active_ids}

        robot_states_cd = {
            rid: {
                "cell": states[rid].cell,
                "active_box": states[rid].active_box,
                "planned_obstacle_cell": states[rid].planned_obstacle_cell,
            }
            for rid in active_ids
        }

        # ---- 4. Detect conflicts ----------------------------------------
        # Skip detection for robot pairs where one is already EVADING —
        # they are in the process of resolving and re-detecting creates
        # an infinite-WAIT loop (WAITER never resumes).
        evading_ids = {rid for rid in active_ids if states[rid].status == "EVADING"}
        conflicts = self._detector.detect(
            robot_plans, box_states, robot_states_cd, h=h
        )
        # Filter out C1 conflicts where one robot is already evading away
        conflicts = [
            c for c in conflicts
            if not (c.conflict_type == ConflictType.ROBOT_ROBOT
                    and any(rid in evading_ids for rid in c.robots_involved))
        ]
        # Skip conflicts involving robots in post-evasion grace period
        grace_ids = {rid for rid, s in states.items() if s._post_evade_grace > 0}
        conflicts = [
            c for c in conflicts
            if not any(rid in grace_ids for rid in c.robots_involved)
        ]

        # ---- 4b. Head-on corridor deadlock is handled dynamically by resolve_sr_social ---
        # (Heuristic back-up loop removed to allow pocket-based evasion to take precedence)

        # ---- 5. Resolve deadlocks ---------------------------------------
        resolved_rids: set[int] = set()

        multi_robot_conflicts = [c for c in conflicts if len(c.robots_involved) >= 2]
        if multi_robot_conflicts and self.resolver:
            conflict_robot_sets = [set(c.robots_involved) for c in multi_robot_conflicts]
            merged: set = set()
            for s in conflict_robot_sets:
                merged |= s
            involved = [r for r in merged if r in active_ids]

            if len(involved) >= 2:
                # Don't re-run DR if any involved robot is already EVADING.
                # Re-assigning WAIT against an evader creates an infinite loop.
                if any(states[r].status == "EVADING" for r in involved):
                    pass  # Resolution already in progress — don't re-assign WAIT
                else:
                    self.deadlock_count += 1
                    robot_cells_d = {rid: states[rid].cell for rid in involved}
                    robot_goals_d = {rid: states[rid].goal for rid in involved}

                    if self.dr_strategy == "sr_social" and self.broadcaster and self.social_map:
                        assignments, evasion_targets = self.resolver.resolve_sr_social(
                            involved, robot_cells_d, robot_goals_d,
                            self.broadcaster, self.social_map
                        )
                    elif self.dr_strategy == "social" and self.social_map:
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

                    if self.broadcaster:
                        for rid, action in assignments.items():
                            if action == "WAIT":
                                w = self.broadcaster.get_sr_interval_width(rid)
                                self.yield_sr_widths.append(w)

                    for rid, action in assignments.items():
                        state = states[rid]
                        if action == "WAIT":
                            # Phase 2A: transfer priority to all evaders
                            evaders = []
                            for other_rid, other_action in assignments.items():
                                if other_action == "EVADE":
                                    states[other_rid].priority_score += state.priority_score
                                    evaders.append(other_rid)
                            state.status = "WAITING"
                            state.wait_ticks = random.randint(3, 8)
                            state.waiting_for_robot = evaders[0] if evaders else None
                            next_cells[rid] = state.cell
                            resolved_rids.add(rid)
                        elif action == "EVADE":
                            state.status = "EVADING"
                            target = evasion_targets.get(rid)
                            state.evasion_target = target
                            if target:
                                # When evading, don't treat WAITING robots as path obstacles
                                # — they will yield. Only avoid DONE and EVADING robots.
                                other_rob_cells = [
                                    states[r].cell for r in states
                                    if r != rid and states[r].status not in ("WAITING", "POCKET_WAITING")
                                ]
                                evade_plan = a_star(
                                    state.cell, target,
                                    self.grid, self.grid_size,
                                    other_robots=other_rob_cells,
                                )
                                state.plan = evade_plan if evade_plan else [state.cell]

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

            next_cell = None
            should_pop = False

            if state.plan:
                if state.cell == state.plan[0]:
                    if len(state.plan) > 1:
                        next_cell = state.plan[1]
                        should_pop = True
                else:
                    next_cell = state.plan[0]
                    should_pop = False

            if next_cell is not None:
                # Collision avoidance: don't step into another robot's cell.
                # EXCEPTION: EVADING robots can push through WAITING robots.
                currently_occupied = set()
                for r in states:
                    if r == rid:
                        continue
                    r_state = states[r]
                    if state.status == "EVADING" and r_state.status in ("WAITING", "POCKET_WAITING"):
                        continue  # evader can push through waiters
                    currently_occupied.add(r_state.cell)
                # Phase 2A: respect priority — don't displace a higher-priority robot
                priority_blocker = any(
                    states[r].cell == next_cell
                    and states[r].priority_score > state.priority_score
                    for r in states if r != rid
                )
                if (next_cell not in currently_occupied
                        and next_cell not in occupied_next.values()
                        and not priority_blocker):
                    if should_pop:
                        state.plan.pop(0)
                        state.cell = next_cell
                    next_cells[rid] = next_cell
                    occupied_next[rid] = next_cell
                    continue

            # Default: stay at current cell
            next_cells[rid] = state.cell
            occupied_next[rid] = state.cell

        return next_cells
