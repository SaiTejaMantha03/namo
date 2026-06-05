import numpy as np
import heapq

class DatasetGenerator:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size

    def a_star(self, grid, start, goal):
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
                return path[::-1]
                
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                    if grid[neighbor[1], neighbor[0]] in [1, 2]:
                        continue
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + h(neighbor)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def generate_single_pair(self, force_safe=False):
        """
        Generates one balanced 20x20 corridor training sample with variations:
        - Aisle widths: 1, 2, or 3 cells
        - Number of robots: 2, 3, or 4
        - Box obstacles: dynamic positions
        - Directional conflict detection (opposite vs same direction)
        - Partial-risk labeling: 0.9 (width 1), 0.4 (width 2), 0.1 (width 3)
        - Negative examples (force_safe=True)
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Border walls
        grid[0, :] = 1; grid[-1, :] = 1; grid[:, 0] = 1; grid[:, -1] = 1
        
        # Aisle Width (1, 2, or 3 cells)
        aisle_width = np.random.choice([1, 2, 3])
        corridor_row = np.random.randint(4, self.grid_size - 6)
        
        # Openings
        left_open = np.random.randint(3, 6)
        right_open = np.random.randint(self.grid_size - 7, self.grid_size - 3)
        
        # Build corridor boundary walls
        for col in range(1, self.grid_size - 1):
            grid[corridor_row - 1, col] = 1
            grid[corridor_row + aisle_width, col] = 1
                
        # Random boxes/obstacles (0 to 2 obstacles)
        num_obstacles = np.random.randint(0, 3)
        obstacle_cells = []
        for _ in range(num_obstacles):
            obs_x = np.random.randint(left_open, right_open + 1)
            obs_y = np.random.randint(corridor_row, corridor_row + aisle_width)
            grid[obs_y, obs_x] = 2
            obstacle_cells.append((obs_x, obs_y))
            
        # Spawn robots (2, 3, or 4 robots)
        num_robots = np.random.choice([2, 3, 4])
        robots_data = []
        
        # Clear positions to spawn
        left_spawn_zone = [(x, y) for x in range(1, left_open) for y in range(corridor_row, corridor_row + aisle_width)]
        right_spawn_zone = [(x, y) for x in range(right_open + 1, self.grid_size - 1) for y in range(corridor_row, corridor_row + aisle_width)]
        
        np.random.shuffle(left_spawn_zone)
        np.random.shuffle(right_spawn_zone)
        
        is_deadlock = not force_safe
        
        for i in range(num_robots):
            if force_safe or not is_deadlock:
                # Safe case: Robots travel in the SAME direction (e.g. all Left to Right)
                if len(left_spawn_zone) > 0 and len(right_spawn_zone) > 0:
                    start = left_spawn_zone.pop()
                    goal = right_spawn_zone.pop()
                    robots_data.append((start, goal, "LR"))
            else:
                # Deadlock case: Asymmetric spawn, opposing directions
                if i % 2 == 0 and len(left_spawn_zone) > 0 and len(right_spawn_zone) > 0:
                    start = left_spawn_zone.pop()
                    goal = right_spawn_zone.pop()
                    robots_data.append((start, goal, "LR"))
                elif len(right_spawn_zone) > 0 and len(left_spawn_zone) > 0:
                    start = right_spawn_zone.pop()
                    goal = left_spawn_zone.pop()
                    robots_data.append((start, goal, "RL"))
                    
        # Occupancy Grid Input
        occupancy = np.zeros((4, self.grid_size, self.grid_size), dtype=float)
        occupancy[0] = (grid == 1).astype(float)
        occupancy[1] = (grid == 2).astype(float)
        
        for start, goal, _ in robots_data:
            occupancy[2, start[1], start[0]] = 1.0
            occupancy[3, goal[1], goal[0]] = 1.0
            
        # Label generation
        label = np.zeros((self.grid_size, self.grid_size), dtype=float)
        
        # Path analysis to find conflict zones
        if is_deadlock and not force_safe:
            # Check opposite paths
            lr_paths = []
            rl_paths = []
            grid_temp = grid.copy()
            # Clear boxes temporarily to calculate potential path intersection
            for ox, oy in obstacle_cells:
                grid_temp[oy, ox] = 0
                
            for start, goal, direction in robots_data:
                path = self.a_star(grid_temp, start, goal)
                if direction == "LR":
                    lr_paths.append(path)
                else:
                    rl_paths.append(path)
                    
            # Identify overlap between opposing flows
            overlap_cells = set()
            for p_lr in lr_paths:
                for p_rl in rl_paths:
                    if p_lr and p_rl:
                        overlap_cells.update(set(p_lr).intersection(set(p_rl)))
                        
            # Apply severity-based partial risk
            # Width 1 = 0.9, Width 2 = 0.4, Width 3 = 0.1
            risk_val = 0.9 if aisle_width == 1 else (0.4 if aisle_width == 2 else 0.1)
            for (ox, oy) in overlap_cells:
                label[oy, ox] = risk_val
                
        return occupancy, label

    def generate_dataset(self, num_samples=1500):
        inputs = []
        labels = []
        
        # Maintain a 50/50 balance of Deadlock (Positive) and Safe (Negative) cases
        half_samples = num_samples // 2
        
        # Generate positive/deadlock samples
        for _ in range(half_samples):
            occ, lbl = self.generate_single_pair(force_safe=False)
            inputs.append(occ)
            labels.append(lbl)
            
        # Generate negative/safe samples
        for _ in range(num_samples - half_samples):
            occ, lbl = self.generate_single_pair(force_safe=True)
            inputs.append(occ)
            labels.append(lbl)
            
        # Shuffle inputs and labels together
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        
        return np.array(inputs)[indices], np.array(labels)[indices]

if __name__ == "__main__":
    generator = DatasetGenerator(grid_size=20)
    print("Generating 1500 balanced NAMO training pairs...")
    inputs, labels = generator.generate_dataset(num_samples=1500)
    
    # Check dataset balance
    positive_count = sum(1 for lbl in labels if np.max(lbl) > 0)
    negative_count = len(labels) - positive_count
    print(f" -> Positive (Deadlock) cases: {positive_count} ({positive_count/len(labels)*100:.1f}%)")
    print(f" -> Negative (Safe) cases:    {negative_count} ({negative_count/len(labels)*100:.1f}%)")
    
    import os
    from pathlib import Path
    npz_path = str(Path(__file__).resolve().parent.parent / "namo_dataset.npz")
    np.savez(npz_path, inputs=inputs, labels=labels)
    print(f"Dataset successfully saved to {npz_path}")
