"""
core/planner.py
---------------
Canonical A* planner and clearing-direction finder for the NAMO project.

All other files should import from here instead of copy-pasting:
    from core.planner import a_star, find_clearing_direction
"""

import heapq
import numpy as np
from typing import Optional


def a_star(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray,
    grid_size: int,
    other_robots: tuple | list = (),
    blocked_cells: tuple | list | set = (),
    ignore_boxes: bool = False,
    locked_boxes: set = frozenset(),
    risk_map: Optional[np.ndarray] = None,
    risk_weight: float = 10.0,
) -> list[tuple[int, int]]:
    """
    Risk-aware A* planner on a 2-D integer occupancy grid.

    Grid values:
        0  = free space
        1  = fixed wall
        2  = movable obstacle (box)
        3+ = robot IDs

    Parameters
    ----------
    start, goal       : (col, row) grid cells.
    grid              : 2-D numpy array of shape (grid_size, grid_size).
    grid_size         : number of cells per side.
    other_robots      : cells occupied by other robots (treated as temporary walls).
    blocked_cells     : additional cells that are impassable (e.g. failed-push boxes).
    ignore_boxes      : if True, boxes are treated as free space (for BYPASS planning).
    locked_boxes      : boxes currently being handled by another robot (impassable).
    risk_map          : optional (grid_size, grid_size) float array in [0, 1].
                        Step cost = 1 + risk_weight * risk_map[row, col].
    risk_weight       : scalar multiplier on the risk map.

    Returns
    -------
    List of (col, row) cells from start to goal (inclusive), or [] if no path.
    """
    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    open_set: list = []
    heapq.heappush(open_set, (0.0, start))
    came_from: dict = {}
    g_score: dict = {start: 0.0}

    robot_set = set(map(tuple, other_robots))
    blocked_set = set(map(tuple, blocked_cells)) | set(map(tuple, locked_boxes))

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
            nb = (current[0] + dx, current[1] + dy)
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
                continue

            cell_val = grid[nb[1], nb[0]]

            # Fixed walls are always impassable
            if cell_val == 1:
                continue
            # Boxes: impassable unless explicitly treating as free
            if cell_val == 2 and not ignore_boxes and nb != goal:
                continue
            # Extra blocked cells
            if nb in blocked_set and nb != goal:
                continue
            # Other robots (temporary walls), except at goal
            if nb in robot_set and nb != goal:
                continue

            step_cost = 1.0
            if risk_map is not None:
                step_cost += risk_weight * float(risk_map[nb[1], nb[0]])

            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative_g
                heapq.heappush(open_set, (tentative_g + h(nb), nb))

    return []  # no path found


def find_clearing_direction(
    box_cell: tuple[int, int],
    grid: np.ndarray,
    grid_size: int,
    other_robots: tuple | list = (),
    locked_zones: set = frozenset(),
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Given a movable box, find a (clear_cell, approach_cell) pair such that:
      - clear_cell  : where the box can be pushed to (must be free).
      - approach_cell: where the robot should stand to push it there (must be free).

    Returns (clear_cell, approach_cell) or None if no valid direction exists.
    """
    bx, by = box_cell
    robot_set = set(map(tuple, other_robots))
    blocked = set(map(tuple, locked_zones))

    candidates = [
        ((bx, by + 1), (bx, by - 1)),   # push North → approach from South
        ((bx, by - 1), (bx, by + 1)),   # push South → approach from North
        ((bx + 1, by), (bx - 1, by)),   # push East  → approach from West
        ((bx - 1, by), (bx + 1, by)),   # push West  → approach from East
    ]

    for clear_cell, approach_cell in candidates:
        cx, cy = clear_cell
        ax, ay = approach_cell
        if not (0 <= cx < grid_size and 0 <= cy < grid_size):
            continue
        if not (0 <= ax < grid_size and 0 <= ay < grid_size):
            continue
        # Clear cell must be free space (not wall, not blocked, not another robot)
        if grid[cy, cx] == 1 or clear_cell in blocked or clear_cell in robot_set:
            continue
        # Approach cell must be free too
        if grid[ay, ax] == 1 or approach_cell in blocked or approach_cell in robot_set:
            continue
        return clear_cell, approach_cell

    return None
