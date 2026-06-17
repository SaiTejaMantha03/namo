import yaml
import numpy as np
from simulation.snamo_simulator import build_clean_grid, run_simulation

r = run_simulation("configs/custom_reconstructed_map_robots.yaml", gui=False, dr_strategy="sr_social")
print("Result:", r)
