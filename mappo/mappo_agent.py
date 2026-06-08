import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mappo.mappo_env import NAMOmappoEnv

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs):
        logits = self.net(obs)
        return logits

class CriticNetwork(nn.Module):
    def __init__(self, joint_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, joint_obs):
        value = self.net(joint_obs)
        return value

class MAPPOAgent:
    def __init__(self, obs_dim, num_agents, lr=3e-4, gamma=0.99, lmbda=0.95, eps_clip=0.2, c_val=0.5, c_ent=0.01):
        self.num_agents = num_agents
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.c_val = c_val
        self.c_ent = c_ent
        
        self.actor = ActorNetwork(obs_dim)
        self.critic = CriticNetwork(obs_dim * num_agents)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, obs_dict):
        actions = {}
        action_log_probs = {}
        entropy = {}
        
        for rid, obs in obs_dict.items():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            actions[rid] = action.item()
            action_log_probs[rid] = dist.log_prob(action).item()
            entropy[rid] = dist.entropy().item()
            
        return actions, action_log_probs, entropy
        
    def train_step(self, obs_history, joint_obs_history, action_history, reward_history, log_prob_history, done_history):
        # Convert list to tensors
        obs_t = torch.tensor(np.array(obs_history), dtype=torch.float32) # Shape: (T, N, obs_dim)
        joint_obs_t = torch.tensor(np.array(joint_obs_history), dtype=torch.float32) # Shape: (T, joint_obs_dim)
        action_t = torch.tensor(np.array(action_history), dtype=torch.long) # Shape: (T, N)
        reward_t = torch.tensor(np.array(reward_history), dtype=torch.float32) # Shape: (T, N)
        old_log_probs_t = torch.tensor(np.array(log_prob_history), dtype=torch.float32) # Shape: (T, N)
        done_t = torch.tensor(np.array(done_history), dtype=torch.float32) # Shape: (T, N)
        
        T = obs_t.shape[0]
        N = self.num_agents
        
        # Calculate state values using Centralized Critic
        values_t = self.critic(joint_obs_t).squeeze(-1) # Shape: (T)
        
        # Calculate Returns and Advantages using Generalized Advantage Estimation (GAE)
        advantages = torch.zeros(T, N)
        returns = torch.zeros(T, N)
        
        # Calculate advantages per agent using shared critic values
        next_value = 0.0
        for agent_idx in range(N):
            gae = 0.0
            for t in reversed(range(T)):
                # Since MAPPO critic is centralized, we approximate value transition
                val = values_t[t].item()
                next_val = values_t[t+1].item() if t < T - 1 else 0.0
                delta = reward_t[t, agent_idx] + self.gamma * next_val * (1.0 - done_t[t, agent_idx]) - val
                gae = delta + self.gamma * self.lmbda * (1.0 - done_t[t, agent_idx]) * gae
                advantages[t, agent_idx] = gae
                returns[t, agent_idx] = advantages[t, agent_idx] + val
                
        # Policy Loss (Actor update)
        # Flatten time and agent dimensions for Actor update since they share parameters
        obs_flat = obs_t.view(-1, obs_t.shape[-1])
        action_flat = action_t.view(-1)
        old_log_probs_flat = old_log_probs_t.view(-1)
        advantages_flat = advantages.view(-1)
        
        logits = self.actor(obs_flat)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(action_flat)
        entropy = dist.entropy().mean()
        
        ratios = torch.exp(new_log_probs - old_log_probs_flat)
        surr1 = ratios * advantages_flat
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_flat
        actor_loss = -torch.min(surr1, surr2).mean() - self.c_ent * entropy
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # Value Loss (Critic update)
        pred_values = self.critic(joint_obs_t).squeeze(-1)
        target_returns = returns.mean(dim=-1) # Centralized average return target
        critic_loss = nn.MSELoss()(pred_values, target_returns)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        return actor_loss.item(), critic_loss.item()

def run_test():
    print("Initializing environment...")
    config_path = "configs/single_corridor_yielding.yaml"
    env = NAMOmappoEnv(config_path, gui=False, max_steps=10)
    
    obs = env.reset()
    obs_dim = next(iter(obs.values())).shape[0]
    num_agents = len(obs)
    
    print(f"Loaded scenario: {config_path}")
    print(f"Num agents: {num_agents} | Obs dimension: {obs_dim}")
    
    agent = MAPPOAgent(obs_dim, num_agents)
    
    # Store trajectory
    obs_history = []
    joint_obs_history = []
    action_history = []
    reward_history = []
    log_prob_history = []
    done_history = []
    
    print("\nRunning test rollouts...")
    for t in range(5):
        # Format observations to match training function expected shape
        obs_list = [obs[rid] for rid in env.robot_ids]
        joint_obs = np.concatenate(obs_list)
        
        actions, action_log_probs, _ = agent.select_action(obs)
        next_obs, rewards, dones, info = env.step(actions)
        
        obs_history.append(obs_list)
        joint_obs_history.append(joint_obs)
        action_history.append([actions[rid] for rid in env.robot_ids])
        reward_history.append([rewards[rid] for rid in env.robot_ids])
        log_prob_history.append([action_log_probs[rid] for rid in env.robot_ids])
        done_history.append([dones[rid] for rid in env.robot_ids])
        
        obs = next_obs
        print(f" Step {t+1} -> Actions selected: {list(actions.values())} | Rewards: {list(rewards.values())}")
        if dones["__all__"]:
            break
            
    print("\nExecuting policy optimization train step...")
    actor_loss, critic_loss = agent.train_step(
        obs_history, joint_obs_history, action_history, reward_history, log_prob_history, done_history
    )
    print(f"Training Loss -> Actor: {actor_loss:.4f} | Critic: {critic_loss:.4f}")
    
    env.close()
    print("\n>>> SUCCESS: MAPPO agent validation passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=True, help="Run validation tests")
    args = parser.parse_args()
    
    if args.test:
        run_test()
