"""
multi_robot/conflict_detection.py
-----------------------------------
Detects the 6 conflict types from MR-NAMO (Paper 5, Section III-B).

Each robot's plan is a list of (col, row) grid cells representing its intended
path over the next h timesteps. The detector inspects these plans and the
current box/robot states to identify all active conflicts.

Conflict Types
--------------
C1  Robot-Robot              : two robots' paths intersect within h steps.
C2  Object-in-Path           : a movable obstacle sits on a robot's route.
C3  Simultaneous Space Access: two robots' *next* cells are identical.
C4  Stealing Object          : another robot is actively moving an obstacle
                               that this robot planned to move.
C5  Stolen Object            : the planned obstacle has already been displaced.
C6  Simultaneous Grab        : two robots reach for the same obstacle in the same step.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import math


class ConflictType(Enum):
    ROBOT_ROBOT              = auto()   # C1
    OBJECT_IN_PATH           = auto()   # C2
    SIMULTANEOUS_SPACE       = auto()   # C3
    STEALING_OBJECT          = auto()   # C4
    STOLEN_OBJECT            = auto()   # C5
    SIMULTANEOUS_GRAB        = auto()   # C6


@dataclass
class Conflict:
    conflict_type: ConflictType
    robots_involved: list[int]
    timestep: Optional[int] = None      # which step (0=immediate, None=general)
    obstacle_cell: Optional[tuple] = None
    details: str = ""

    def __repr__(self):
        return (
            f"Conflict({self.conflict_type.name}, "
            f"robots={self.robots_involved}, "
            f"t={self.timestep}, obs={self.obstacle_cell})"
        )


class ConflictDetector:
    """
    Inspect all robot plans simultaneously to find conflicts.

    Usage
    -----
    detector = ConflictDetector()
    conflicts = detector.detect(
        robot_plans  = {rid: [(col,row), ...], ...},
        box_states   = {box_id: (col,row), ...},
        robot_states = {rid: {"cell": (col,row), "active_box": box_id_or_None}, ...},
        h = 10,
    )
    """

    def detect(
        self,
        robot_plans: dict[int, list[tuple]],
        box_states: dict[int, tuple],
        robot_states: dict[int, dict],
        h: int = 10,
    ) -> list[Conflict]:
        conflicts: list[Conflict] = []

        robot_ids = list(robot_plans.keys())

        # Clip plans to horizon h
        plans_h = {rid: robot_plans[rid][:h] for rid in robot_ids}

        # Build plan sets for fast lookup
        plan_sets = {rid: set(map(tuple, plans_h[rid])) for rid in robot_ids}

        # Current box positions
        box_cells = set(map(tuple, box_states.values()))
        box_cell_to_id = {tuple(v): k for k, v in box_states.items()}

        for i, rid_a in enumerate(robot_ids):
            plan_a = plans_h[rid_a]
            plan_set_a = plan_sets[rid_a]

            # ----- C2: Object-in-Path ----------------------------------------
            for step, cell in enumerate(plan_a):
                if cell in box_cells:
                    bid = box_cell_to_id.get(cell)
                    # Check it's not being actively moved by *this* robot
                    active = robot_states[rid_a].get("active_box")
                    if active is None or active != bid:
                        conflicts.append(Conflict(
                            ConflictType.OBJECT_IN_PATH,
                            robots_involved=[rid_a],
                            timestep=step,
                            obstacle_cell=cell,
                            details=f"Box {bid} on path",
                        ))
                        break  # report first occurrence only

            for j, rid_b in enumerate(robot_ids):
                if j <= i:
                    continue  # avoid duplicate pairs

                plan_b = plans_h[rid_b]
                plan_set_b = plan_sets[rid_b]

                # ----- C1: Robot-Robot path intersection ---------------------
                overlap = plan_set_a & plan_set_b
                if overlap:
                    # find earliest overlapping timestep
                    min_t = h
                    for t, cell in enumerate(plan_a):
                        if cell in plan_set_b:
                            min_t = t
                            break
                    conflicts.append(Conflict(
                        ConflictType.ROBOT_ROBOT,
                        robots_involved=[rid_a, rid_b],
                        timestep=min_t,
                        details=f"Paths intersect at {len(overlap)} cells",
                    ))

                # ----- C3: Simultaneous Space Access (next cell only) --------
                next_a = plan_a[1] if len(plan_a) > 1 else None
                next_b = plan_b[1] if len(plan_b) > 1 else None
                if next_a is not None and next_b is not None and next_a == next_b:
                    conflicts.append(Conflict(
                        ConflictType.SIMULTANEOUS_SPACE,
                        robots_involved=[rid_a, rid_b],
                        timestep=1,
                        obstacle_cell=next_a,
                        details="Both robots moving into same cell next step",
                    ))

                # ----- C4: Stealing Object -----------------------------------
                # rid_a planned to move a box that rid_b is *actively* moving
                active_b = robot_states[rid_b].get("active_box")
                if active_b is not None:
                    active_b_cell = box_states.get(active_b)
                    if active_b_cell and tuple(active_b_cell) in plan_set_a:
                        conflicts.append(Conflict(
                            ConflictType.STEALING_OBJECT,
                            robots_involved=[rid_a, rid_b],
                            obstacle_cell=tuple(active_b_cell),
                            details=f"Robot {rid_a} plans to use box that {rid_b} is moving",
                        ))

                # Symmetric: rid_b plans to use box that rid_a is moving
                active_a = robot_states[rid_a].get("active_box")
                if active_a is not None:
                    active_a_cell = box_states.get(active_a)
                    if active_a_cell and tuple(active_a_cell) in plan_set_b:
                        conflicts.append(Conflict(
                            ConflictType.STEALING_OBJECT,
                            robots_involved=[rid_b, rid_a],
                            obstacle_cell=tuple(active_a_cell),
                            details=f"Robot {rid_b} plans to use box that {rid_a} is moving",
                        ))

                # ----- C6: Simultaneous Grab ---------------------------------
                # Both robots plan to interact with the same box in the same step
                for t in range(min(len(plan_a), len(plan_b))):
                    cell_at_t_a = plan_a[t]
                    cell_at_t_b = plan_b[t]
                    # "interacting" = reaching a box cell
                    if cell_at_t_a in box_cells and cell_at_t_b in box_cells:
                        bid_a = box_cell_to_id.get(cell_at_t_a)
                        bid_b = box_cell_to_id.get(cell_at_t_b)
                        if bid_a is not None and bid_a == bid_b:
                            conflicts.append(Conflict(
                                ConflictType.SIMULTANEOUS_GRAB,
                                robots_involved=[rid_a, rid_b],
                                timestep=t,
                                obstacle_cell=cell_at_t_a,
                                details=f"Both robots grab box {bid_a} at step {t}",
                            ))
                            break

        # ----- C5: Stolen Object (compare planned vs actual box positions) ---
        for rid in robot_ids:
            planned_box_cell = robot_states[rid].get("planned_obstacle_cell")
            if planned_box_cell is not None:
                planned_box_cell = tuple(planned_box_cell)
                # Find the actual box at that location
                if planned_box_cell not in box_cells:
                    # Box has moved — stolen object
                    conflicts.append(Conflict(
                        ConflictType.STOLEN_OBJECT,
                        robots_involved=[rid],
                        obstacle_cell=planned_box_cell,
                        details=f"Robot {rid}'s planned obstacle no longer at {planned_box_cell}",
                    ))

        return conflicts
