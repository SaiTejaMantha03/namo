import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from decision.unet_decision_pipeline import UNetDecisionPipeline
from maps.namo_environments import WarehouseEnvironment
from decision.namo_visualizer import Visualizer

def run_evaluation():
    project_dir = Path(__file__).resolve().parent
    weights_path = str(project_dir / "models" / "namo_unet.pth")
    pipeline = UNetDecisionPipeline(weights_path)
    
    # Define scenarios to evaluate
    scenarios = [
        {
            "name": "3x3 Corridor (Single corridor, blocked)",
            "size": 3,
            "grid": np.array([
                [1, 1, 1],
                [0, 2, 0],
                [1, 1, 1]
            ]),
            "start": (0, 1),
            "goal": (2, 1),
            "obstacle": (1, 1),
            "push_cost": 2.0
        },
        {
            "name": "5x5 Divider (Long detour vs push)",
            "size": 5,
            "grid": np.array([
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 0, 2, 0, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1]
            ]),
            "start": (1, 2),
            "goal": (3, 2),
            "obstacle": (2, 2),
            "push_cost": 2.0
        },
        {
            "name": "20x20 Warehouse Aisle layout",
            "size": 20,
            "grid": None, # Will generate dynamically
            "start": (3, 2),
            "goal": (3, 17),
            "obstacle": (3, 10),
            "push_cost": 4.0
        }
    ]
    
    table_lines = []
    table_lines.append("| Scenario Name | Grid Size | Decision | Bypass Cost ($C_{by}$) | Removal Cost ($C_{re}$) | Obstacle Pushes | Success Rate |")
    table_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    print("\n" + "="*80)
    print("NAMO EVALUATION PIPELINE RUN")
    print("="*80)
    
    for s in scenarios:
        if s["grid"] is None:
            # Dynamic Warehouse layout
            wh = WarehouseEnvironment(grid_size=20)
            wh.add_robot(s["start"], s["goal"])
            wh.add_obstacle(s["obstacle"][0], s["obstacle"][1])
            grid = wh.generate_occupancy_grid()
        else:
            grid = s["grid"]
            
        decision, c_by, c_re = pipeline.evaluate_decision(
            grid, s["start"], s["goal"], s["obstacle"], base_push_cost=s["push_cost"]
        )
        
        # Calculate evaluation metrics
        pushes = 1 if decision == "REMOVE" else 0
        success_rate = "100%" if c_re[0] != float('inf') or c_by[0] != float('inf') else "0%"
        
        c_by_str = f"[{c_by[0]:.1f}, {c_by[1]:.1f}]" if c_by[0] != float('inf') else "[inf, inf]"
        c_re_str = f"[{c_re[0]:.1f}, {c_re[1]:.1f}]" if c_re[0] != float('inf') else "[inf, inf]"
        
        row = f"| {s['name']} | {s['size']}x{s['size']} | {decision} | {c_by_str} | {c_re_str} | {pushes} | {success_rate} |"
        table_lines.append(row)
        print(f"Evaluated: {s['name']:28s} | Decision: {decision:6s} | Success: {success_rate}")
        
        # Plot dynamic visualization of the decision
        vis_path = project_dir / "results" / "visualizations" / f"eval_{s['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}_{decision}.png"
        vis = Visualizer(s["size"])
        path_to_plot = []
        grid_for_plot = grid.copy()
        
        if decision == "BYPASS":
            # treat obstacle as 2 (wall)
            if s["obstacle"]:
                grid_for_plot[s["obstacle"][1], s["obstacle"][0]] = 2
            path_to_plot = vis.a_star(grid_for_plot, s["start"], s["goal"])
        else: # REMOVE
            # Need path to obstacle, then path to goal
            path_to_obs = vis.a_star(grid_for_plot, s["start"], s["obstacle"])
            path_from_obs = vis.a_star(grid, s["obstacle"], s["goal"])
            if path_to_obs and path_from_obs:
                path_to_plot = path_to_obs + path_from_obs[1:]
                
        vis.plot_scenario(grid_for_plot, s["start"], s["goal"], s["obstacle"], path_to_plot, f"{s['name']} - {decision}", str(vis_path))
        
    print("="*80)
    
    # Save the evaluation table
    table_md = "\n".join(table_lines)
    eval_file = project_dir / "results" / "evaluation_tables" / "metrics_table.md"
    eval_file.write_text(table_md)
    print(f"Saved evaluation metrics table to: results/evaluation_tables/metrics_table.md\n")

if __name__ == "__main__":
    run_evaluation()
