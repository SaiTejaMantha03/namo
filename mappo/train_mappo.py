import os
import sys
import numpy as np
import argparse
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mappo.mappo_env import NAMOmappoEnv
from mappo.mappo_agent import MAPPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/single_corridor_yielding.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--episodes-per-epoch", type=int, default=5, help="Number of episodes per training epoch")
    parser.add_argument("--max-steps", type=int, default=40, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()
    
    print(f"Loading environment with config: {args.config}")
    env = NAMOmappoEnv(args.config, gui=False, max_steps=args.max_steps)
    
    # Initialize observation dimension and number of agents
    obs = env.reset()
    obs_dim = next(iter(obs.values())).shape[0]
    num_agents = len(obs)
    
    print(f"Environment initialized with {num_agents} agents, observation dimension: {obs_dim}")
    agent = MAPPOAgent(obs_dim, num_agents, lr=args.lr)
    
    print("Starting MAPPO Training Loop...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        obs_history = []
        joint_obs_history = []
        action_history = []
        reward_history = []
        log_prob_history = []
        done_history = []
        
        epoch_rewards = []
        epoch_collisions = 0
        success_count = 0
        
        for ep in range(args.episodes_per_epoch):
            obs = env.reset()
            episode_reward = 0
            
            for t in range(args.max_steps):
                obs_list = [obs[rid] for rid in env.robot_ids]
                joint_obs = np.concatenate(obs_list)
                
                actions, action_log_probs, _ = agent.select_action(obs)
                next_obs, rewards, dones, info = env.step(actions)
                
                # Record trajectory
                obs_history.append(obs_list)
                joint_obs_history.append(joint_obs)
                action_history.append([actions[rid] for rid in env.robot_ids])
                reward_history.append([rewards[rid] for rid in env.robot_ids])
                log_prob_history.append([action_log_probs[rid] for rid in env.robot_ids])
                done_history.append([dones[rid] for rid in env.robot_ids])
                
                episode_reward += sum(rewards.values())
                epoch_collisions += info["collisions"]
                
                obs = next_obs
                
                if dones["__all__"]:
                    if info["success"]:
                        success_count += 1
                    break
                    
            epoch_rewards.append(episode_reward)
            
        # Optimize policy
        actor_loss, critic_loss = agent.train_step(
            obs_history, joint_obs_history, action_history, reward_history, log_prob_history, done_history
        )
        
        avg_reward = np.mean(epoch_rewards)
        success_rate = (success_count / args.episodes_per_epoch) * 100.0
        
        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
            f"Avg Reward: {avg_reward:7.2f} | "
            f"Collisions: {epoch_collisions:4d} | "
            f"Success Rate: {success_rate:5.1f}% | "
            f"Loss Actor: {actor_loss:6.4f} | "
            f"Critic: {critic_loss:6.4f}"
        )
        
    env.close()
    print("=" * 60)
    print("Training Complete!")
    
    # Save the trained model checkpoint
    save_dir = Path(__file__).resolve().parent.parent / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), str(save_dir / "mappo_actor_checkpoint.pth"))
    torch.save(agent.critic.state_dict(), str(save_dir / "mappo_critic_checkpoint.pth"))
    print(f"Saved policy checkpoints to: models/mappo_*_checkpoint.pth")

if __name__ == "__main__":
    main()
