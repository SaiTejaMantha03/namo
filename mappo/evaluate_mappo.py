import os
import sys
import numpy as np
import argparse
import torch
import pybullet as p
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mappo.mappo_env import NAMOmappoEnv
from mappo.mappo_agent import MAPPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/warehouse_3robots.yaml", help="Path to config file")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    parser.add_argument("--checkpoint", default=None, help="Path to actor checkpoint (.pth)")
    args = parser.parse_args()
    
    print(f"Loading environment with config: {args.config}")
    env = NAMOmappoEnv(args.config, gui=args.gui, max_steps=args.max_steps)
    
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result
    obs_dim = next(iter(obs.values())).shape[0]
    num_agents = len(obs)
    
    print(f"Environment initialized with {num_agents} agents, observation dimension: {obs_dim}")
    agent = MAPPOAgent(obs_dim, num_agents)
    
    # Load trained model weights
    if args.checkpoint:
        actor_path = Path(args.checkpoint)
    else:
        save_dir = Path(__file__).resolve().parent.parent / "models"
        actor_path = save_dir / "mappo_actor_checkpoint.pth"
    
    if actor_path.exists():
        state_dict = torch.load(str(actor_path))
        if "actor_state" in state_dict:
            agent.actor.load_state_dict(state_dict["actor_state"])
        else:
            agent.actor.load_state_dict(state_dict)
        print(f"Loaded actor checkpoint from {actor_path}")
    else:
        print(f"WARNING: No actor checkpoint found at {actor_path}. Evaluating with random weights.")
        
    # Set to evaluation mode
    agent.actor.eval()
    
    print("\nStarting MAPPO Evaluation Rollout...")
    print("=" * 60)
    
    episode_reward = 0
    collisions = 0
    consecutive_waits = 0
    
    for t in range(args.max_steps):
        # We don't need backprop for evaluation, so we can just use argmax
        actions = {}
        all_wait = True
        for rid, agent_obs in obs.items():
            obs_tensor = torch.tensor(agent_obs, dtype=torch.float32)
            with torch.no_grad():
                logits = agent.actor(obs_tensor)
                # Deterministic action selection for evaluation
                action = torch.argmax(logits).item()
            actions[rid] = action
            if action != 3:
                all_wait = False
                
        if all_wait:
            consecutive_waits += 1
        else:
            consecutive_waits = 0
            
        next_obs, rewards, dones, info = env.step(actions)
        
        episode_reward += sum(rewards.values())
        collisions += info["collisions"]
        
        obs = next_obs
        
        if t % 5 == 0 or dones["__all__"] or consecutive_waits > 10:
            print(f"Step {t:03d} | Actions: {list(actions.values())} | Current Reward: {episode_reward:.2f}")
            
            # Print per-robot distances
            for rid in env.robot_ids:
                pos, _ = p.getBasePositionAndOrientation(rid)
                g = env.sim.robot_goals[rid]
                dist = np.hypot(pos[0] - (g[0]+0.5)*env.cell_size, pos[1] - (g[1]+0.5)*env.cell_size)
                act_name = ["NAV", "PUSH", "YIELD", "WAIT"][actions[rid]]
                status = env.sim.coord_states[rid].status
                print(f"  Robot {rid}: {act_name} | Status: {status:<10} | Dist to goal: {dist:.2f}m")
                
        if consecutive_waits > 10:
            print(">>> WARNING: Model appears frozen (all agents outputting WAIT)")
            break
            
        if dones["__all__"]:
            print(f"\nEpisode finished at step {t}!")
            
            # Physically verify success
            real_success = True
            for rid in env.robot_ids:
                pos, _ = p.getBasePositionAndOrientation(rid)
                g = env.sim.robot_goals[rid]
                dist = np.hypot(pos[0] - (g[0]+0.5)*env.cell_size, pos[1] - (g[1]+0.5)*env.cell_size)
                if dist > 0.6:
                    real_success = False
                    break
                    
            if real_success:
                print(">>> SUCCESS: All robots reached their goals!")
            else:
                print(">>> FAILED: Not all robots reached their goals.")
            break
            
    print("=" * 60)
    print(f"Evaluation Complete | Total Reward: {episode_reward:.2f} | Collisions: {collisions}")
    env.close()

if __name__ == "__main__":
    main()
