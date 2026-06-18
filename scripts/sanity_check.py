import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from mappo.mappo_env import NAMOmappoEnv

def sanity_check():
    print("=" * 50)
    print("TEST 1: MASK COMPUTATION (warehouse_3robots)")
    print("=" * 50)
    
    env = NAMOmappoEnv("configs/warehouse_3robots.yaml", gui=False, max_steps=100)
    obs = env.reset()
    masks = env.current_action_masks
    costs = env.current_costs
    
    for rid in env.robot_ids:
        print(f"Robot {rid} | Mask: {masks[rid]} | Base Cost: {costs[rid]}")
        if not masks[rid][1] or not masks[rid][2]:
            print(f"  -> WARNING: PUSH or YIELD masked (inf cost found)")
            
    print("\n" + "=" * 50)
    print("TEST 2: REWARD COMPUTATION (forced pushes)")
    print("=" * 50)
    
    # We force all robots to push
    actions = {rid: 1 for rid in env.robot_ids}
    obs, rewards, dones, info = env.step(actions)
    
    new_costs = env.current_costs
    for rid in env.robot_ids:
        c_before = costs[rid]
        c_after = new_costs[rid]
        print(f"Robot {rid} PUSH | Cost Before: {c_before:.1f} | Cost After: {c_after:.1f} | Reward: {rewards[rid]:.2f}")
        
    env.close()

if __name__ == "__main__":
    sanity_check()
