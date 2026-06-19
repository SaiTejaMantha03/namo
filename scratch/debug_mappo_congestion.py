import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from mappo.mappo_agent import MAPPOAgent
from mappo.mappo_env import NAMOmappoEnv
import pybullet as p

def main():
    cfg_path = "configs/narrow_doorway_congestion.yaml"
    checkpoint = "checkpoints/v3_maxres/mappo_final.pth"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    env = NAMOmappoEnv(config_path=cfg_path, gui=False, max_steps=150)
    obs, info = env.reset()
    
    sample = next(iter(obs.values()))
    obs_dim = sample.shape[0]
    action_dim = env.action_dim
    max_agents = 4
    
    agent = MAPPOAgent(
        obs_dim=obs_dim, num_agents=max_agents,
        action_dim=action_dim, device=device
    )
    agent.load(checkpoint)
    
    print(f"Starting MAPPO trace on: {cfg_path}")
    act_names = {0: "NAV", 1: "PUSH", 2: "YIELD", 3: "WAIT"}
    
    for step in range(150):
        action_mask = info.get("action_mask", None)
        with torch.no_grad():
            actions, _, _ = agent.select_action(obs, action_masks=action_mask)
            
        print(f"\n--- Step {step} ---")
        for rid in env.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            cell = env.sim.world_to_cell(pos[:2]) if hasattr(env.sim, "world_to_cell") else (int(pos[0]), int(pos[1]))
            state = env.sim.coord_states[rid]
            act = actions.get(rid)
            print(f"  Robot {rid}: pos={pos[:2]} cell={cell} status={state.status} action={act_names.get(act)} mask={action_mask.get(rid) if action_mask else None}")
            
        obs, rewards, dones, info = env.step(actions)
        
        if dones.get("__all__", False):
            print(f"\nSuccess! Episode finished in {step} steps.")
            break
            
    env.close()

if __name__ == "__main__":
    main()
