import numpy as np
import matplotlib.pyplot as plt
import heapq

class Visualizer:
    def __init__(self, grid_size=10):
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
                    if grid[neighbor[1], neighbor[0]] in [1, 2] and neighbor != goal:
                        continue
                        
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + h(neighbor)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def plot_scenario(self, grid, start, goal, obstacle, path, title, save_path):
        fig, ax = plt.subplots(figsize=(5, 5))
        
        display_grid = np.zeros((self.grid_size, self.grid_size, 3))
        display_grid[:, :] = [0.95, 0.95, 0.95] # Light grey free space
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r, c] == 1:
                    display_grid[r, c] = [0.2, 0.2, 0.2] # Dark grey walls
                elif grid[r, c] == 2:
                    display_grid[r, c] = [0.9, 0.3, 0.1] # Orange obstacle
                    
        ax.imshow(display_grid, origin='upper')
        
        # Draw path
        if path:
            px, py = zip(*path)
            ax.plot(px, py, color='#1f77b4', linewidth=4, label='Path', zorder=2)
            ax.scatter(px[1:-1], py[1:-1], color='#3182bd', s=80, zorder=3)
            
        # Draw Start, Goal, and Obstacle
        if start:
            ax.scatter(start[0], start[1], color='#2ca02c', s=200, label='Start', zorder=4, edgecolor='black')
        if goal:
            ax.scatter(goal[0], goal[1], color='#d62728', s=200, label='Goal', zorder=4, edgecolor='black')
        if obstacle:
            ax.scatter(obstacle[0], obstacle[1], color='#ff7f0e', s=150, label='Box', zorder=4, marker='s', edgecolor='black')
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True, which='both', color='white', linestyle='-', linewidth=2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved visualization to {save_path}")

def generate_layout(size):
    """
    Returns (grid, start, goal, obstacle) for a given size.
    """
    grid = np.zeros((size, size), dtype=int)
    if size == 1:
        grid[0, 0] = 2
        return grid, None, None, (0, 0)
        
    elif size == 3:
        grid[0, :] = 1
        grid[2, :] = 1
        start = (0, 1)
        goal = (2, 1)
        obstacle = (1, 1)
        return grid, start, goal, obstacle
        
    elif size == 5:
        grid[0, :] = 1
        grid[4, :] = 1
        grid[:, 0] = 1
        grid[:, 4] = 1
        grid[1, 1] = 1; grid[1, 3] = 1
        grid[3, 1] = 1; grid[3, 3] = 1
        start = (1, 2)
        goal = (3, 2)
        obstacle = (2, 2)
        return grid, start, goal, obstacle
        
    elif size == 10:
        grid[0, :] = 1; grid[9, :] = 1; grid[:, 0] = 1; grid[:, 9] = 1
        for col in range(1, 9):
            if col != 4 and col != 6:
                grid[5, col] = 1
        start = (4, 2)
        goal = (4, 7)
        obstacle = (4, 5)
        return grid, start, goal, obstacle
        
    return grid, None, None, None

def run():
    from pathlib import Path
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results" / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for size in [1, 3, 5, 10]:
        grid, start, goal, obstacle = generate_layout(size)
        
        # 1. Bypass Scenario
        grid_bypass = grid.copy()
        if obstacle:
            grid_bypass[obstacle[1], obstacle[0]] = 2
            
        bypass_path = []
        if start and goal:
            bypass_path = Visualizer(size).a_star(grid_bypass, start, goal)
            
        bypass_img = str(results_dir / f"namo_{size}x{size}_bypass.png")
        Visualizer(size).plot_scenario(grid_bypass, start, goal, obstacle, bypass_path, f"{size}x{size} Bypass Path", bypass_img)
        
        # 2. Removal Scenario
        removal_path = []
        if start and goal and obstacle:
            path_to_obs = Visualizer(size).a_star(grid_bypass, start, obstacle)
            path_from_obs = Visualizer(size).a_star(grid, obstacle, goal)
            if path_to_obs and path_from_obs:
                removal_path = path_to_obs + path_from_obs[1:]
                
        removal_img = str(results_dir / f"namo_{size}x{size}_removal.png")
        Visualizer(size).plot_scenario(grid_bypass, start, goal, obstacle, removal_path, f"{size}x{size} Removal Path", removal_img)


if __name__ == "__main__":
    run()
