import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.snamo_simulator import SNAMOSimulator, world_to_cell
import pybullet as p

def main():
    cfg_path = "configs/single_corridor_yielding.yaml"
    sim = SNAMOSimulator(cfg_path, gui=False, dr_strategy="sr_social")
    sim.reset()
    
    print(f"Starting debug of scenario: {sim.name}")
    print(f"Robots: {sim.robot_ids}")
    
    for step in range(300):
        # Print states
        print(f"\n--- Step {step} ---")
        for rid in sim.robot_ids:
            state = sim.coord_states[rid]
            pos, _ = p.getBasePositionAndOrientation(rid)
            cell = world_to_cell(pos[:2], sim.cs)
            print(f"  Robot {rid}: pos={pos[:2]} cell={cell} coord_cell={state.cell} "
                  f"status={state.status} wait={state.wait_ticks} waiting_for={state.waiting_for_robot} "
                  f"target={state.evasion_target} plan={state.plan[:3]} score={state.priority_score:.2f}")
                  
        sim.step()
        
        if sim.success:
            print(f"\nSuccess in {step} steps!")
            break
            
    sim.close()

if __name__ == "__main__":
    main()
