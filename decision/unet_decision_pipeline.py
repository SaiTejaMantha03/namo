import sys
import torch
import numpy as np
import heapq
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from unet.unet import AttentionUNet
from maps.namo_environments import WarehouseEnvironment, IntersectionEnvironment

class UNetDecisionPipeline:
    def __init__(self, weights_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AttentionUNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def risk_aware_a_star(self, grid, risk_map, start, goal, risk_penalty_weight=10.0):
        grid_size = grid.shape[0]
        h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return g_score[goal], path[::-1]
                
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                    if grid[neighbor[1], neighbor[0]] in [1, 2] and neighbor != goal:
                        continue
                        
                    step_cost = 1.0 + (risk_penalty_weight * risk_map[neighbor[1], neighbor[0]])
                    tentative_g = g_score[current] + step_cost
                    
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
                        
        return float('inf'), []

    def evaluate_decision(self, grid, start, goal, obstacle_pos, push_cost=3.0, save_plot_path=None):
        """
        Generic decision pipeline supporting any size grid:
        - Automatically pads small grids to 20x20 for UNet max pooling compatibility.
        - Computes bypass vs removal costs.
        - Optionally saves a contrast comparison heatmap.
        """
        grid_size = grid.shape[0]
        
        # 1. Pad layout to 20x20 container if needed
        target_size = 20
        offset_x = max(0, (target_size - grid_size) // 2)
        offset_y = max(0, (target_size - grid_size) // 2)
        
        padded_grid = np.ones((target_size, target_size), dtype=int) # Default to walls
        # Copy original grid
        padded_grid[offset_y:offset_y+grid_size, offset_x:offset_x+grid_size] = grid
        
        # Shift coords
        p_start = (start[0] + offset_x, start[1] + offset_y)
        p_goal = (goal[0] + offset_x, goal[1] + offset_y)
        p_obstacle = (obstacle_pos[0] + offset_x, obstacle_pos[1] + offset_y)
        
        # Create 4-channel 20x20 input
        occ = np.zeros((4, target_size, target_size), dtype=float)
        occ[0] = (padded_grid == 1).astype(float)
        occ[1] = (padded_grid == 2).astype(float)
        occ[2, p_start[1], p_start[0]] = 1.0
        occ[3, p_goal[1], p_goal[0]] = 1.0
        
        # 2. Forward pass & crop back
        input_tensor = torch.tensor(occ, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            risk_map_20 = self.model(input_tensor).squeeze().cpu().numpy()
            
        risk_map = risk_map_20[offset_y:offset_y+grid_size, offset_x:offset_x+grid_size]
        
        # 3. Compute Bypass Cost (keeping obstacle in place)
        grid_bypass = grid.copy()
        grid_bypass[obstacle_pos[1], obstacle_pos[0]] = 2
        bypass_cost, bypass_path = self.risk_aware_a_star(grid_bypass, risk_map, start, goal)
        
        # 4. Compute Removal Cost (reaching obstacle + pushing + reaching goal)
        reach_cost, _ = self.risk_aware_a_star(grid_bypass, risk_map, start, obstacle_pos)
        grid_cleared = grid.copy()
        grid_cleared[obstacle_pos[1], obstacle_pos[0]] = 0
        remain_cost, _ = self.risk_aware_a_star(grid_cleared, risk_map, obstacle_pos, goal)
        
        removal_cost = reach_cost + push_cost + remain_cost
        decision = "REMOVE" if removal_cost < bypass_cost else "BYPASS"
        
        # 5. Optional Plotting using Custom Heatmap
        if save_plot_path:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
            
            # Input layout representation
            render_img = np.ones((grid_size, grid_size, 3)) * 0.95
            render_img[grid == 1] = [0.2, 0.2, 0.2] # Walls
            render_img[grid == 2] = [0.95, 0.6, 0.1] # Box
            render_img[start[1], start[0]] = [0.2, 0.6, 0.9] # Robot
            render_img[goal[1], goal[0]] = [0.9, 0.2, 0.2] # Goal
            
            axes[0].imshow(render_img, origin='upper')
            axes[0].set_title(f"Input Layout ({grid_size}x{grid_size})", fontsize=11, fontweight='bold')
            axes[0].axis('off')
            
            from matplotlib.colors import LinearSegmentedColormap
            custom_cmap = LinearSegmentedColormap.from_list("yellow_orange_red", ["yellow", "orange", "red"])
            
            im = axes[1].imshow(risk_map, cmap=custom_cmap, origin='upper', vmin=0.0, vmax=1.0)
            axes[1].set_title("Predicted Risk Heatmap", fontsize=11, fontweight='bold')
            axes[1].axis('off')
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(save_plot_path, dpi=120)
            plt.close()
            
        return decision, bypass_cost, removal_cost

def run_all_evaluations():
    project_dir = Path(__file__).resolve().parent.parent
    weights_path = str(project_dir / "models" / "namo_unet.pth")
    pipeline = UNetDecisionPipeline(weights_path)
    
    # Define test setups: (grid, start, goal, obstacle, size, title, filename)
    scenarios = []
    
    # 1. 3x3 Corridor
    g3 = np.zeros((3, 3), dtype=int)
    g3[0, :] = 1; g3[2, :] = 1
    scenarios.append((g3, (0, 1), (2, 1), (1, 1), 3, "3x3 Corridor", "namo_3x3_decision.png"))
    
    # 2. 5x5 Divider
    g5 = np.zeros((5, 5), dtype=int)
    g5[0, :] = 1; g5[4, :] = 1; g5[:, 0] = 1; g5[:, 4] = 1
    g5[1, 1] = 1; g5[1, 3] = 1; g5[3, 1] = 1; g5[3, 3] = 1
    scenarios.append((g5, (1, 2), (3, 2), (2, 2), 5, "5x5 Divider", "namo_5x5_decision.png"))
    
    # 3. 20x20 Warehouse
    wh = WarehouseEnvironment(grid_size=20)
    wh.add_robot((3, 2), (3, 17))
    wh.add_obstacle(3, 10)
    scenarios.append((wh.generate_occupancy_grid(), (3, 2), (3, 17), (3, 10), 20, "20x20 Warehouse", "namo_unet_prediction.png"))

    print("\nRunning generic NAMO Decision Pipeline Evaluations...")
    print("="*65)
    
    results_dir = project_dir / "results" / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for grid, start, goal, obstacle, size, title, filename in scenarios:
        save_path = str(results_dir / filename)
        decision, c_by, c_re = pipeline.evaluate_decision(grid, start, goal, obstacle, push_cost=4.0, save_plot_path=save_path)
        print(f"{title:15s} | C_by: {c_by:.2f} | C_re: {c_re:.2f} | Decision: {decision:6s} | Saved: {filename}")
    print("="*65 + "\n")


if __name__ == "__main__":
    run_all_evaluations()
