import numpy as np
import matplotlib.pyplot as plt

class NAMOEnvironment:
    def __init__(self, grid_size=15, name="NAMO Environment"):
        self.grid_size = grid_size
        self.name = name
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Grid Legend:
        # 0 = Free Space
        # 1 = Fixed Wall/Shelf
        # 2 = Movable Obstacle (Box)
        # 3+ = Robots (Unique IDs)
        
        self.walls = set()
        self.obstacles = set()
        self.robots = [] # List of dicts: {'id': int, 'start': (x, y), 'goal': (x, y), 'pos': (x, y)}
        self.goals = {} # Maps robot_id -> goal_pos
        self.robot_counter = 3
        
        # Build default outer borders
        self._add_border_walls()

    def _add_border_walls(self):
        for i in range(self.grid_size):
            self.add_wall(i, 0)
            self.add_wall(i, self.grid_size - 1)
            self.add_wall(0, i)
            self.add_wall(self.grid_size - 1, i)

    def add_wall(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.walls.add((x, y))

    def add_obstacle(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.obstacles.add((x, y))

    def add_robot(self, start_pos, goal_pos):
        """
        Dynamically adds a robot with a specified start and goal position.
        """
        robot_id = self.robot_counter
        self.robot_counter += 1
        
        robot_entry = {
            'id': robot_id,
            'start': start_pos,
            'goal': goal_pos,
            'pos': start_pos
        }
        self.robots.append(robot_entry)
        self.goals[robot_id] = goal_pos
        return robot_id

    def generate_occupancy_grid(self):
        """
        Generates the 2D numpy occupancy grid.
        """
        # Reset grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Place fixed walls
        for (w_x, w_y) in self.walls:
            self.grid[w_y, w_x] = 1
            
        # Place movable obstacles
        for (o_x, o_y) in self.obstacles:
            self.grid[o_y, o_x] = 2
            
        # Place active robots
        for robot in self.robots:
            rx, ry = robot['pos']
            self.grid[ry, rx] = robot['id']
            
        return self.grid

    def visualize(self, save_path):
        """
        Generates a beautiful Matplotlib visualization showing layout, robots, goals, and obstacles.
        """
        grid_data = self.generate_occupancy_grid()
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Base representation color grid
        color_grid = np.zeros((self.grid_size, self.grid_size, 3))
        color_grid[:, :] = [0.96, 0.96, 0.96] # Clean light grey free space
        
        # Color definitions
        wall_color = [0.2, 0.2, 0.2]      # Charcoal for walls/shelves
        obs_color = [0.95, 0.6, 0.1]      # Bright orange for boxes
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid_data[r, c] == 1:
                    color_grid[r, c] = wall_color
                elif grid_data[r, c] == 2:
                    color_grid[r, c] = obs_color
                    
        ax.imshow(color_grid, origin='upper')
        
        # Color palette for multiple robots (and their matching goals)
        robot_colors = ['#1f77b4', '#9467bd', '#2ca02c', '#d62728', '#bcbd22', '#17becf']
        
        # Plot robots, goals, and lines connecting them
        for idx, robot in enumerate(self.robots):
            r_id = robot['id']
            color = robot_colors[idx % len(robot_colors)]
            rx, ry = robot['pos']
            gx, gy = robot['goal']
            
            # Plot Goal with translucent color
            ax.scatter(gx, gy, color=color, s=250, marker='*', zorder=4, edgecolor='black', label=f'Goal R{r_id-2}')
            # Plot Robot
            ax.scatter(rx, ry, color=color, s=200, marker='o', zorder=5, edgecolor='black', label=f'Robot R{r_id-2}')
            # Draw helper path indicator line
            ax.plot([rx, gx], [ry, gy], color=color, linestyle=':', alpha=0.6, linewidth=2)
            
        # Draw obstacle markers
        for (ox, oy) in self.obstacles:
            ax.scatter(ox, oy, color='#ff7f0e', s=120, marker='s', zorder=3, edgecolor='black')

        ax.set_title(self.name, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True, which='both', color='#d3d3d3', linestyle='-', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Unique legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.25, 1.0))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved layout: {self.name} to {save_path}")

# =====================================================================
# Specific Layout Generators
# =====================================================================

class WarehouseEnvironment(NAMOEnvironment):
    def __init__(self, grid_size=15):
        super().__init__(grid_size, "Warehouse Aisle Deadlock Layout")
        self._build_warehouse()

    def _build_warehouse(self):
        # Build vertical shelves (walls)
        # Shelf rows running from row 2 to 12, with aisles at col 3, 7, 11
        for col in [2, 4, 6, 8, 10, 12]:
            for row in range(2, 13):
                # Leave middle cross-aisle open at row 7
                if row != 7:
                    self.add_wall(col, row)


class IntersectionEnvironment(NAMOEnvironment):
    def __init__(self, grid_size=15):
        super().__init__(grid_size, "4-Way Corridor Intersection Layout")
        self._build_intersection()

    def _build_intersection(self):
        # Create a central 4-way intersection.
        # Main corridors run vertically (cols 6-8) and horizontally (rows 6-8)
        # Fill everything else with walls to make it narrow
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Ignore boundary walls (handled by super)
                if x == 0 or x == self.grid_size - 1 or y == 0 or y == self.grid_size - 1:
                    continue
                # If outside corridors, it's a solid block
                if not (6 <= x <= 8 or 6 <= y <= 8):
                    self.add_wall(x, y)


# =====================================================================
# Demonstration & Output
# =====================================================================

def generate_demos():
    # 1. Warehouse Deadlock Demo
    wh = WarehouseEnvironment(grid_size=15)
    # Add multiple robots crossing in the narrow aisles
    wh.add_robot(start_pos=(3, 2), goal_pos=(3, 12))  # Robot 1 goes South down aisle 3
    wh.add_robot(start_pos=(3, 12), goal_pos=(3, 2))  # Robot 2 goes North up aisle 3 (causing deadlock!)
    # Add a movable box blocking a cross-aisle exit
    wh.add_obstacle(3, 7) # Box directly in the middle intersection of aisle 3
    
    wh_img = "/Users/saitejamantha/.gemini/antigravity/brain/89fa2c6d-0c06-4ff1-b436-ff754208e534/namo_warehouse.png"
    wh.visualize(wh_img)

    # 2. Intersection Crossing Deadlock Demo
    inter = IntersectionEnvironment(grid_size=15)
    # Add multiple robots meeting in the center from 4 directions
    inter.add_robot(start_pos=(7, 2), goal_pos=(7, 12))  # Robot 1 North -> South
    inter.add_robot(start_pos=(7, 12), goal_pos=(7, 2))  # Robot 2 South -> North
    inter.add_robot(start_pos=(2, 7), goal_pos=(12, 7))  # Robot 3 West -> East
    inter.add_robot(start_pos=(12, 7), goal_pos=(2, 7))  # Robot 4 East -> West
    # Add movable obstacles cluttering the center intersection
    inter.add_obstacle(7, 6)
    inter.add_obstacle(6, 7)
    inter.add_obstacle(8, 7)

    inter_img = "/Users/saitejamantha/.gemini/antigravity/brain/89fa2c6d-0c06-4ff1-b436-ff754208e534/namo_intersection.png"
    inter.visualize(inter_img)

if __name__ == "__main__":
    generate_demos()
