import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from decision.unet_decision_pipeline import UNetDecisionPipeline
from maps.namo_environments import WarehouseEnvironment

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
            grid, s["start"], s["goal"], s["obstacle"], push_cost=s["push_cost"]
        )
        
        # Calculate evaluation metrics
        pushes = 1 if decision == "REMOVE" else 0
        success_rate = "100%" if c_re != float('inf') or c_by != float('inf') else "0%"
        
        c_by_str = f"{c_by:.2f}" if c_by != float('inf') else "inf"
        c_re_str = f"{c_re:.2f}" if c_re != float('inf') else "inf"
        
        row = f"| {s['name']} | {s['size']}x{s['size']} | {decision} | {c_by_str} | {c_re_str} | {pushes} | {success_rate} |"
        table_lines.append(row)
        print(f"Evaluated: {s['name']:28s} | Decision: {decision:6s} | Success: {success_rate}")
        
    print("="*80)
    
    # Save the evaluation table
    table_md = "\n".join(table_lines)
    eval_file = project_dir / "results" / "evaluation_tables" / "metrics_table.md"
    eval_file.write_text(table_md)
    print(f"Saved evaluation metrics table to: results/evaluation_tables/metrics_table.md\n")

if __name__ == "__main__":
    run_evaluation()
