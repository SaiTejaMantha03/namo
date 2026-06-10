import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from unet.unet import AttentionUNet
from maps.namo_environments import WarehouseEnvironment, IntersectionEnvironment
from core.planner import a_star
from uncertainty.bypass_model import TrajectoryRegressionModel
from uncertainty.interval_decision import choose_action
from uncertainty.action_uncertainty import ManipulationBeliefModel

class UNetDecisionPipeline:
    def __init__(self, weights_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AttentionUNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def risk_aware_a_star(self, grid, risk_map, start, goal, risk_penalty_weight=10.0):
        """Backward-compatible wrapper around core.planner.a_star."""
        path = a_star(
            start, goal, grid, grid.shape[0],
            risk_map=risk_map, risk_weight=risk_penalty_weight,
        )
        if not path:
            return float('inf'), []
        # recompute g-score from path for cost reporting
        cost = 0.0
        for i in range(1, len(path)):
            nb = path[i]
            cost += 1.0 + risk_penalty_weight * float(risk_map[nb[1], nb[0]])
        return cost, path

    def get_risk_map(self, grid, start, goal):
        grid_size = grid.shape[0]
        target_size = 20
        offset_x = max(0, (target_size - grid_size) // 2)
        offset_y = max(0, (target_size - grid_size) // 2)
        
        padded_grid = np.ones((target_size, target_size), dtype=int)
        padded_grid[offset_y:offset_y+grid_size, offset_x:offset_x+grid_size] = grid
        
        p_start = (start[0] + offset_x, start[1] + offset_y)
        p_goal = (goal[0] + offset_x, goal[1] + offset_y)
        
        occ = np.zeros((4, target_size, target_size), dtype=float)
        occ[0] = (padded_grid == 1).astype(float)
        occ[1] = (padded_grid == 2).astype(float)
        occ[2, p_start[1], p_start[0]] = 1.0
        occ[3, p_goal[1], p_goal[0]] = 1.0
        
        input_tensor = torch.tensor(occ, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            risk_map_20 = self.model(input_tensor).squeeze().cpu().numpy()
            
        return risk_map_20

    def predict_trajectory_cost_interval(self, base_cost, path):
        """
        Simulates a Gaussian Linear Regressor that predicts navigation cost
        with uncertainty based on trajectory smoothness features.
        Returns: [min_cost, max_cost]
        """
        if not path or base_cost == float('inf'):
            return [float('inf'), float('inf')]
            
        length = len(path) - 1 # number of steps
        num_turns = 0
        for i in range(1, len(path) - 1):
            prev_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            next_dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if prev_dir != next_dir:
                num_turns += 1
                
        # The base_cost already includes length + risk penalty from UNet.
        # We add the turn penalty to the mean and compute std dev (sigma).
        w_turn_mu = 0.5
        w_len_sigma = 0.05
        w_turn_sigma = 0.2
        
        mu = base_cost + (num_turns * w_turn_mu)
        sigma = (length * w_len_sigma) + (num_turns * w_turn_sigma)
        
        # Interval is mu +/- 1 standard deviation
        return [max(0.0, mu - sigma), mu + sigma]

    def evaluate_decision(self, grid, start, goal, obstacle_pos, base_push_cost=3.0, alpha=5.0, beta=1.0, save_plot_path=None):
        """
        Generic decision pipeline supporting any size grid:
        - Automatically pads small grids to 20x20 for UNet max pooling compatibility.
        - Computes bypass vs removal costs using Gaussian Linear Regressor intervals.
        - Models pushing success rate using a Beta(alpha, beta) distribution.
        - Makes decisions using U = (max + min) / 2 logic.
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
        
        # 3. Compute Bypass Cost Interval (keeping obstacle in place)
        grid_bypass = grid.copy()
        grid_bypass[obstacle_pos[1], obstacle_pos[0]] = 2
        bypass_base, bypass_path = self.risk_aware_a_star(grid_bypass, risk_map, start, goal)
        bypass_cost_interval = self.predict_trajectory_cost_interval(bypass_base, bypass_path)
        
        # 4. Compute Removal Cost Interval (reaching obstacle + pushing + reaching goal)
        reach_base, reach_path = self.risk_aware_a_star(grid_bypass, risk_map, start, obstacle_pos)
        reach_interval = self.predict_trajectory_cost_interval(reach_base, reach_path)
        
        grid_cleared = grid.copy()
        grid_cleared[obstacle_pos[1], obstacle_pos[0]] = 0
        remain_base, remain_path = self.risk_aware_a_star(grid_cleared, risk_map, obstacle_pos, goal)
        remain_interval = self.predict_trajectory_cost_interval(remain_base, remain_path)
        
        # Calculate Success Rate interval from Beta distribution
        S_mean = alpha / (alpha + beta)
        S_var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
        S_std = np.sqrt(S_var)
        S_min = max(0.05, S_mean - 2 * S_std) 
        S_max = min(1.0, S_mean + 2 * S_std)
        push_interval = [base_push_cost / S_max, base_push_cost / S_min]
        
        removal_cost_interval = [
            reach_interval[0] + push_interval[0] + remain_interval[0],
            reach_interval[1] + push_interval[1] + remain_interval[1]
        ]
        
        # 5. Interval-Based Decision Making: U = (max + min)/2
        U_bypass = (bypass_cost_interval[0] + bypass_cost_interval[1]) / 2.0
        U_removal = (removal_cost_interval[0] + removal_cost_interval[1]) / 2.0
        
        decision = "REMOVE" if U_removal < U_bypass else "BYPASS"
        
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
            
        return decision, bypass_cost_interval, removal_cost_interval

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
        decision, c_by, c_re = pipeline.evaluate_decision(grid, start, goal, obstacle, base_push_cost=4.0, save_plot_path=save_path)
        by_str = f"[{c_by[0]:.1f}, {c_by[1]:.1f}]" if c_by[0] != float('inf') else "[inf, inf]"
        re_str = f"[{c_re[0]:.1f}, {c_re[1]:.1f}]" if c_re[0] != float('inf') else "[inf, inf]"
        print(f"{title:15s} | C_by: {by_str:13s} | C_re: {re_str:13s} | Decision: {decision:6s} | Saved: {filename}")
    print("="*65 + "\n")


if __name__ == "__main__":
    run_all_evaluations()
