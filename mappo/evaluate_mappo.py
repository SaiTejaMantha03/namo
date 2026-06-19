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
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--randomize-starts", action="store_true", help="Enable randomized starts to break initial symmetry")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action selection (Categorical)")
    args = parser.parse_args()
    
    print(f"Loading environment with config: {args.config} (randomize_starts={args.randomize_starts})")
    env = NAMOmappoEnv(args.config, gui=args.gui, max_steps=args.max_steps, randomize_starts=args.randomize_starts)
    
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {"action_mask": env.current_action_masks}
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
    
    print(f"\nStarting MAPPO Evaluation: {args.episodes} episodes (stochastic={args.stochastic})...")
    print("=" * 60)
    
    successes = 0
    total_steps = []
    total_collisions = []
    
    for ep in range(args.episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {"action_mask": env.current_action_masks}
            
        episode_reward = 0
        collisions = 0
        consecutive_waits = 0
        ep_success = False
        
        for t in range(args.max_steps):
            actions = {}
            all_wait = True
            action_mask = info.get("action_mask", {})
            for rid, agent_obs in obs.items():
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32)
                mask = action_mask.get(rid, [True, True, True, True])
                mask_tensor = torch.tensor(mask, dtype=torch.bool)
                with torch.no_grad():
                    logits = agent.actor(obs_tensor, mask_tensor)
                    if args.stochastic:
                        from torch.distributions import Categorical
                        probs = torch.softmax(logits, dim=-1)
                        dist = Categorical(probs)
                        action = dist.sample().item()
                    else:
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
            
            if consecutive_waits > 20:
                break
                
            if dones["__all__"]:
                # Verify success
                real_success = True
                for rid in env.robot_ids:
                    pos, _ = p.getBasePositionAndOrientation(rid)
                    g = env.sim.robot_goals[rid]
                    dist = np.hypot(pos[0] - (g[0]+0.5)*env.cell_size, pos[1] - (g[1]+0.5)*env.cell_size)
                    if dist > 0.6:
                        real_success = False
                        break
                ep_success = real_success
                break
        
        if ep_success:
            successes += 1
        total_steps.append(t + 1)
        total_collisions.append(collisions)
        
        print(f"Episode {ep+1:>2}/{args.episodes} | Success: {ep_success:<5} | Steps: {t+1:>3} | Collisions: {collisions}")
        
    print("=" * 60)
    success_rate = (successes / args.episodes) * 100
    mean_steps = np.mean(total_steps)
    mean_col = np.mean(total_collisions)
    print(f"Evaluation Complete for {args.config}")
    print(f"Success Rate: {success_rate:.1f}% ({successes}/{args.episodes})")
    print(f"Average Steps: {mean_steps:.1f}")
    print(f"Average Collisions: {mean_col:.2f}")
    env.close()

if __name__ == "__main__":
    main()
