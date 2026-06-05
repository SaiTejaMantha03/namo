import numpy as np
import heapq

class NAMOPlanner:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        
    def a_star(self, grid, start, goal):
        """
        Standard A* pathfinder.
        Returns the path length (cost) and the path coordinates.
        If no path exists, returns infinity.
        """
        h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: h(start)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return len(path) - 1, path[::-1]
                
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                    # Treat walls (1) and unremoved obstacles (2) as blocked, unless it is the target destination
                    if grid[neighbor[1], neighbor[0]] in [1, 2] and neighbor != goal:
                        continue
                        
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + h(neighbor)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        return float('inf'), []

    def make_decision(self, grid, start, goal, obstacle_pos, push_cost=3):
        """
        Calculates bypass cost C_by and removal cost C_re, then chooses the optimal strategy.
        """
        print(f"\nEvaluating NAMO Decision from Start {start} to Goal {goal}:")
        
        # 1. Calculate Bypass Cost C_by
        # The obstacle remains in place (treated as blocked / value 2)
        grid_with_obstacle = grid.copy()
        grid_with_obstacle[obstacle_pos[1], obstacle_pos[0]] = 2
        bypass_cost, bypass_path = self.a_star(grid_with_obstacle, start, goal)
        
        # 2. Calculate Removal Cost C_re
        # Step A: Path cost to reach the obstacle
        path_to_obs_cost, path_to_obs = self.a_star(grid_with_obstacle, start, obstacle_pos)
        
        # Step B: Path cost from cleared obstacle position to goal
        grid_cleared = grid.copy()
        # The obstacle is removed/cleared (value 0)
        grid_cleared[obstacle_pos[1], obstacle_pos[0]] = 0 
        path_from_obs_cost, path_from_obs = self.a_star(grid_cleared, obstacle_pos, goal)
        
        # C_re = cost to reach obstacle + cost to manipulate/push it + cost from obstacle to goal
        removal_cost = path_to_obs_cost + push_cost + path_from_obs_cost
        
        print(f" -> Bypass Cost (C_by): {bypass_cost} steps")
        print(f" -> Removal Cost (C_re): {removal_cost} steps (Reaching: {path_to_obs_cost} + Push: {push_cost} + Goal: {path_from_obs_cost})")
        
        # 3. Decision Making
        if removal_cost < bypass_cost:
            print(">>> DECISION: REMOVE OBSTACLE (C_re < C_by) <<<")
            return "REMOVE", removal_cost
        else:
            print(">>> DECISION: BYPASS OBSTACLE (C_by <= C_re) <<<")
            return "BYPASS", bypass_cost

# Define Scenarios to demonstrate the decision making pipeline
def run_pipeline_demo():
    planner = NAMOPlanner(grid_size=10)
    
    # -------------------------------------------------------------
    # Scenario A: Detour is extremely long (Remove Obstacle is best)
    # -------------------------------------------------------------
    grid_a = np.zeros((10, 10), dtype=int)
    grid_a[0, :] = 1; grid_a[9, :] = 1; grid_a[:, 0] = 1; grid_a[:, 9] = 1
    # Create horizontal dividing wall with only a narrow corridor at col 4
    for col in range(1, 9):
        if col != 4:
            grid_a[5, col] = 1
            
    start = (4, 2)
    goal = (4, 7)
    obstacle = (4, 5) # Obstacle blocks the only corridor opening
    
    print("=================== SCENARIO A: LONG DETOUR ===================")
    print("The only corridor is blocked. Detouring requires going all the way around.")
    planner.make_decision(grid_a, start, goal, obstacle, push_cost=3)
    
    # -------------------------------------------------------------
    # Scenario B: Detour is short/easy (Bypass Obstacle is best)
    # -------------------------------------------------------------
    grid_b = np.zeros((10, 10), dtype=int)
    grid_b[0, :] = 1; grid_b[9, :] = 1; grid_b[:, 0] = 1; grid_b[:, 9] = 1
    # Corridor has two openings (col 4 and col 6)
    for col in range(1, 9):
        if col != 4 and col != 6:
            grid_b[5, col] = 1
            
    obstacle_b = (4, 5) # Blocks only the left corridor opening
    
    print("\n=================== SCENARIO B: SHORT DETOUR ===================")
    print("Corridor opening is blocked, but there is a clear adjacent bypass opening.")
    planner.make_decision(grid_b, start, goal, obstacle_b, push_cost=5)

if __name__ == "__main__":
    run_pipeline_demo()
