import yaml
import numpy as np
from decision.snamo_planner import SNAMOPlanner
from simulation.snamo_simulator import build_clean_grid

with open("configs/custom_reconstructed_map_robots.yaml") as f:
    cfg = yaml.safe_load(f)

grid = build_clean_grid(cfg)
gs = cfg["world"]["grid_size"]

planner = SNAMOPlanner(grid, gs)

print("--- Planning at (13, 4) ---")
action, waypoints = planner.plan(
    start=(13, 4),
    goal=(18, 18),
    box_cells=[(2, 1), (3, 1), (16, 1), (17, 1), (8, 2), (8, 3), (11, 3), (11, 4), (3, 5), (4, 5), (13, 5), (15, 6)],
)
print("Action:", action)
print("Waypoints:", waypoints[:10])
