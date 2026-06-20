import sys
import argparse
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mappo.mappo_agent import MAPPOAgent
from mappo.mappo_env import NAMOmappoEnv

def main():
    parser = argparse.ArgumentParser(description="Run trained MAPPO model in GUI")
    parser.add_argument("--config", default="configs/single_corridor_yielding.yaml",
                        help="YAML config path")
    parser.add_argument("--checkpoint", default="checkpoints/v3_maxres/mappo_final.pth",
                        help="Path to MAPPO checkpoint")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--control-interval", type=int, default=15,
                        help="Control interval for step simulation (default 15 to match v4)")
    args = parser.parse_args()

    gui = not args.no_gui
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        chk = torch.load(args.checkpoint, map_location="cpu")
        chk_obs_dim = chk['actor_state']['net.0.weight'].shape[1]
        include_congestion = (chk_obs_dim == 49)
        print(f"Detected checkpoint obs_dim={chk_obs_dim} (include_congestion_feats={include_congestion})")
    except Exception as e:
        print(f"WARNING: Could not auto-detect obs_dim from checkpoint: {e}. Defaulting to include_congestion_feats=True")
        include_congestion = True

    print(f"Loading environment... Device: {device} GUI: {gui}")
    env = NAMOmappoEnv(config_path=args.config, gui=gui, max_steps=300, include_congestion_feats=include_congestion, control_interval=args.control_interval)
    
    obs, info = env.reset()
    sample = next(iter(obs.values()))
    obs_dim = sample.shape[0]
    action_dim = env.action_dim
    max_agents = 4
    
    agent = MAPPOAgent(
        obs_dim=obs_dim, num_agents=max_agents,
        action_dim=action_dim, device=device
    )
    print(f"Loading checkpoint {args.checkpoint}...")
    try:
        agent.load(args.checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Running with randomly initialized agent instead.")
    
    done = False
    steps = 0
    
    try:
        while not done and steps < 300:
            action_mask = info.get("action_mask", None)
            with torch.no_grad():
                actions, _, _ = agent.select_action(obs, action_masks=action_mask)
            
            if gui:
                act_names = {0: "NAV", 1: "PUSH", 2: "YIELD", 3: "WAIT"}
                action_strs = [f"Robot {rid}: {act_names[act]}" for rid, act in actions.items()]
                print(f"Step {steps:3d} | Actions: " + "  ".join(action_strs))
            
            obs, rewards, dones, info = env.step(actions)
            steps += 1
            done = dones.get("__all__", False)
            if gui:
                time.sleep(0.05)
            
        success = info.get("success", False)
        print(f"\nEpisode finished. Success: {success} in {steps} steps.")
        print(f"[MAPPOSim] Steps={steps * 30} Status={'SUCCESS' if success else 'FAILED'}")
    except KeyboardInterrupt:
        print("\nTerminated by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
