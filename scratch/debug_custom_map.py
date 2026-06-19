import sys
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.snamo_simulator import SNAMOSimulator, world_to_cell
import pybullet as p

def main():
    cfg_path = "configs/custom_reconstructed_map_robots.yaml"
    sim = SNAMOSimulator(cfg_path, gui=False, dr_strategy="sr_social")
    sim.reset()
    
    print(f"Starting debug of custom reconstructed map scenario")
    print(f"Robots: {sim.robot_ids}")
    
    last_cells = {}
    stuck_ticks = 0
    
    for step in range(3000):
        # Gather current robot cells
        current_cells = {}
        for rid in sim.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            current_cells[rid] = world_to_cell(pos[:2], sim.cs)
            
        # Check if robots moved
        moved = False
        for rid in sim.robot_ids:
            if rid not in last_cells or last_cells[rid] != current_cells[rid]:
                moved = True
        last_cells = current_cells.copy()
        
        if not moved:
            stuck_ticks += 1
        else:
            stuck_ticks = 0
            
        # Print state if stuck ticks accumulate or periodically
        if stuck_ticks == 1 or stuck_ticks == 20 or step % 100 == 0 or sim.success:
            print(f"\n--- Step {step} (stuck={stuck_ticks}) ---")
            for rid in sim.robot_ids:
                state = sim.coord_states[rid]
                pos, _ = p.getBasePositionAndOrientation(rid)
                cell = current_cells[rid]
                print(f"  Robot {rid}: pos={pos[:2]} cell={cell} coord_cell={state.cell} "
                      f"status={state.status} wait={state.wait_ticks} waiting_for={state.waiting_for_robot} "
                      f"target={state.evasion_target} plan={state.plan[:3]}")
                      
        sim.step()
        
        if sim.success:
            print(f"\nSuccess in {step} steps!")
            break
        if getattr(sim, "stalled", False):
            print(f"\nStalled at step {step}!")
            break
            
    sim.close()

if __name__ == "__main__":
    main()
