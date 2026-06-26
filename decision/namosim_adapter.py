"""
decision/namosim_adapter.py
----------------------------
Adapter to use namosim's A* algorithm with the existing grid-based world.

This extracts the core graph search from namosim without ROS2 dependencies.
"""

import heapq
import math
import typing as t


# Constants from namosim
SQRT_OF_2 = math.sqrt(2.0)
TAXI_NEIGHBORHOOD = ((0, 1), (0, -1), (1, 0), (-1, 0))
CHESSBOARD_NEIGHBORHOOD_EXTRAS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def is_in_matrix(pos, width, height):
    return 0 <= pos[0] < width and 0 <= pos[1] < height


class PriorityQueue:
    """Priority queue for A* from namosim."""
    def __init__(self):
        self.heap = []
        self.elements_to_heap_nodes_uids = {}
        self.next_uid = 1

    def push(self, cost, element):
        new_heap_node = HeapNode(cost, element, self.next_uid)
        self.next_uid += 1
        if element in self.elements_to_heap_nodes_uids:
            self.elements_to_heap_nodes_uids[element].append(new_heap_node.uid)
        else:
            self.elements_to_heap_nodes_uids[element] = [new_heap_node.uid]
        heapq.heappush(self.heap, new_heap_node)

    def pop(self):
        while self:
            candidate_heap_node = heapq.heappop(self.heap)
            corresponding_element = candidate_heap_node.element
            corresponding_uids = self.elements_to_heap_nodes_uids[corresponding_element]
            if corresponding_uids[-1] == candidate_heap_node.uid:
                corresponding_uids.pop()
                if not corresponding_uids:
                    del self.elements_to_heap_nodes_uids[corresponding_element]
                return corresponding_element
            corresponding_uids.remove(candidate_heap_node.uid)
        return None

    def __bool__(self):
        return bool(self.heap)


class HeapNode:
    def __init__(self, cost, element, uid):
        self.cost = cost
        self.element = element
        self.uid = uid

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.element == other
        return self.element == other.element


def reconstruct_path(came_from: t.Dict, end, reverse: bool = True) -> t.List:
    path = [end]
    current = end
    while current in came_from:
        current = came_from[current]
        path.append(current)
    if reverse:
        path.reverse()
    return path


def namosim_a_star(
    start: tuple,
    goal: tuple,
    grid: list,
    width: int,
    height: int,
    check: t.Callable = lambda x: x == 0,
    risk_map=None,
    risk_weight: float = 0.0,
    chessboard: bool = False,
) -> list:
    """
    A* pathfinding from namosim, adapted for grid-based worlds.
    
    Parameters
    ----------
    start, goal : (col, row) tuples
    grid : 2D list where 0=free, 1=wall, 2=box
    width, height : grid dimensions
    check : function to determine if a cell is traversable
    risk_map : optional 2D array of risk costs
    risk_weight : multiplier for risk costs
    chessboard : if True, allow diagonal movement
    
    Returns
    -------
    List of (col, row) tuples from start to goal, or empty list if no path
    """
    came_from = {}
    open_queue = PriorityQueue()
    close_set = set()
    gscore = {start: 0.0}
    
    def heuristic(a, b):
        if chessboard:
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return max(dx, dy) + (SQRT_OF_2 - 1) * min(dx, dy)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_queue.push(heuristic(start, goal), start)
    
    while open_queue:
        current = open_queue.pop()
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        if current in close_set:
            continue
        close_set.add(current)
        
        # Get neighbors
        neighbors = []
        tentative_gscores = []
        current_gscore = gscore[current]
        
        # Taxi (4-directional) neighbors
        for i, j in TAXI_NEIGHBORHOOD:
            neighbor = (current[0] + i, current[1] + j)
            if (neighbor not in close_set 
                and is_in_matrix(neighbor, width, height)
                and check(grid[neighbor[1]][neighbor[0]])):
                neighbors.append(neighbor)
                step_cost = 1.0
                if risk_map is not None:
                    step_cost += risk_weight * float(risk_map[neighbor[1]][neighbor[0]])
                tentative_gscores.append(current_gscore + step_cost)
        
        # Chessboard (diagonal) neighbors
        if chessboard:
            for i, j in CHESSBOARD_NEIGHBORHOOD_EXTRAS:
                neighbor = (current[0] + i, current[1] + j)
                if (neighbor not in close_set 
                    and is_in_matrix(neighbor, width, height)
                    and check(grid[neighbor[1]][neighbor[0]])
                    and check(grid[current[1]][neighbor[0]])
                    and check(grid[neighbor[1]][current[0]])):
                    neighbors.append(neighbor)
                    step_cost = SQRT_OF_2
                    if risk_map is not None:
                        step_cost += risk_weight * float(risk_map[neighbor[1]][neighbor[0]])
                    tentative_gscores.append(current_gscore + step_cost)
        
        for neighbor, tentative_g_score in zip(neighbors, tentative_gscores):
            if neighbor not in gscore or tentative_g_score < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore = tentative_g_score + heuristic(neighbor, goal)
                open_queue.push(fscore, neighbor)
    
    return []  # no path found


def namosim_a_star_grid(
    start: tuple,
    goal: tuple,
    grid,
    grid_size: int,
    other_robots: tuple = (),
    blocked_cells: tuple = (),
    ignore_boxes: bool = False,
    locked_boxes: set = frozenset(),
    risk_map=None,
    risk_weight: float = 0.0,
) -> list:
    """
    Drop-in replacement for core.planner.a_star using namosim's algorithm.
    
    Grid values:
        0  = free space
        1  = fixed wall
        2  = movable obstacle (box)
        3+ = robot IDs
    """
    robot_set = set(map(tuple, other_robots))
    blocked_set = set(map(tuple, blocked_cells)) | set(map(tuple, locked_boxes))
    
    def check(cell_val):
        if cell_val == 1:
            return False
        if cell_val == 2 and not ignore_boxes:
            return False
        return True
    
    def check_with_robots(cell_val):
        return check(cell_val)
    
    # For cells occupied by other robots, we need special handling
    path = namosim_a_star(
        start=start,
        goal=goal,
        grid=grid.tolist() if hasattr(grid, 'tolist') else grid,
        width=grid_size,
        height=grid_size,
        check=check_with_robots,
        risk_map=risk_map.tolist() if risk_map is not None and hasattr(risk_map, 'tolist') else risk_map,
        risk_weight=risk_weight,
        chessboard=False,
    )
    
    # Filter out cells occupied by other robots (except goal)
    if path:
        filtered = [path[0]]
        for cell in path[1:]:
            if cell == goal or (cell not in robot_set and cell not in blocked_set):
                filtered.append(cell)
        return filtered
    
    return path
