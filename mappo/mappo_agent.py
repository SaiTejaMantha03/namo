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
    """Shared-parameter actor with LayerNorm for training stability on MPS."""
    def __init__(self, obs_dim, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        # Orthogonal init for fast convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, action_mask=None):
        logits = self.net(obs)
        if action_mask is not None:
            INF = torch.finfo(logits.dtype).max
            logits = logits.masked_fill(~action_mask, -INF)
        return logits

class CriticNetwork(nn.Module):
    """Centralised critic — sees joint obs of all agents."""
    def __init__(self, joint_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, joint_obs):
        return self.net(joint_obs)

class MAPPOAgent:
    def __init__(self, obs_dim, num_agents, action_dim=4, device=None,
                 lr=3e-4, gamma=0.99, lmbda=0.95, eps_clip=0.2,
                 c_val=0.5, c_ent=0.02,
                 ppo_epochs=4, minibatch_size=256, max_grad_norm=0.5):
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.c_val = c_val
        self.c_ent = c_ent
        self.ppo_epochs = ppo_epochs          # K update passes per batch
        self.minibatch_size = minibatch_size  # mini-batch size for each pass
        self.max_grad_norm = max_grad_norm    # gradient clipping
        self.device = device if device is not None else torch.device("cpu")

        self.actor = ActorNetwork(obs_dim, action_dim=action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim * num_agents).to(self.device)

        # Separate LRs: critic needs higher LR to keep up with actor
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr * 2, eps=1e-5)
        
        # Experience buffers
        self.obs_history = []
        self.joint_obs_history = []
        self.action_history = []
        self.reward_history = []
        self.log_prob_history = []
        self.done_history = []
        self.action_mask_history = []
        
    def select_action(
        self,
        obs_dict,
        action_masks=None,
    ):
        actions = {}
        action_log_probs = {}
        entropy = {}
        
        for rid, obs in obs_dict.items():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            
            mask_tensor = None
            if action_masks and rid in action_masks:
                mask_tensor = torch.tensor(action_masks[rid], dtype=torch.bool).to(self.device)
                
            logits = self.actor(obs_tensor, action_mask=mask_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            actions[rid] = action.item()
            action_log_probs[rid] = dist.log_prob(action).item()
            entropy[rid] = dist.entropy().item()
            
        return actions, action_log_probs, entropy
        
    def train_step(self, obs_history, joint_obs_history, action_history,
                   reward_history, log_prob_history, done_history):
        """PPO update with K mini-batch epochs — fully vectorised on MPS."""
        obs_t        = torch.tensor(np.array(obs_history),       dtype=torch.float32).to(self.device)  # (T, N, obs)
        joint_obs_t  = torch.tensor(np.array(joint_obs_history), dtype=torch.float32).to(self.device)  # (T, joint)
        action_t     = torch.tensor(np.array(action_history),    dtype=torch.long).to(self.device)     # (T, N)
        reward_t     = torch.tensor(np.array(reward_history),    dtype=torch.float32).to(self.device)  # (T, N)
        old_lp_t     = torch.tensor(np.array(log_prob_history),  dtype=torch.float32).to(self.device)  # (T, N)
        done_t       = torch.tensor(np.array(done_history),      dtype=torch.float32).to(self.device)  # (T, N)

        T, N = obs_t.shape[0], self.num_agents

        # ── GAE returns & advantages (computed once, no grad) ────────────
        with torch.no_grad():
            values_t = self.critic(joint_obs_t).squeeze(-1)  # (T,)

            advantages = torch.zeros(T, N, device=self.device)
            returns    = torch.zeros(T, N, device=self.device)
            for agent_idx in range(N):
                gae = 0.0
                for t in reversed(range(T)):
                    val      = values_t[t].item()
                    next_val = values_t[t + 1].item() if t < T - 1 else 0.0
                    delta    = (reward_t[t, agent_idx]
                                + self.gamma * next_val * (1.0 - done_t[t, agent_idx])
                                - val)
                    gae = delta + self.gamma * self.lmbda * (1.0 - done_t[t, agent_idx]) * gae
                    advantages[t, agent_idx] = gae
                    returns[t, agent_idx]    = gae + val

        # Flatten across time × agents for the actor
        obs_flat   = obs_t.view(-1, obs_t.shape[-1])         # (T*N, obs)
        act_flat   = action_t.view(-1)                       # (T*N,)
        old_lp_flat = old_lp_t.view(-1)                      # (T*N,)
        adv_flat   = advantages.view(-1)                     # (T*N,)
        ret_flat   = returns.view(-1)                        # (T*N,)

        # Normalise advantages globally — prevents scale issues across scenarios
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        D = obs_flat.shape[0]  # total data points
        mb = min(self.minibatch_size, D)

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        n_updates = 0

        # ── K PPO mini-batch epochs — saturates MPS compute ─────────────
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(D, device=self.device)
            for start in range(0, D, mb):
                idx = perm[start: start + mb]

                # ── Actor update ────────────────────────────────────────
                logits   = self.actor(obs_flat[idx])
                dist     = Categorical(logits=logits)
                new_lp   = dist.log_prob(act_flat[idx])
                entropy  = dist.entropy().mean()

                ratios = torch.exp(new_lp - old_lp_flat[idx])
                surr1  = ratios * adv_flat[idx]
                surr2  = torch.clamp(ratios,
                                     1.0 - self.eps_clip,
                                     1.0 + self.eps_clip) * adv_flat[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.c_ent * entropy

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                # ── Critic update (uses joint obs — slice per timestep) ──
                # Map flat actor indices back to timestep indices
                t_idx = idx // N
                t_idx = t_idx.clamp(0, T - 1)
                pred_v   = self.critic(joint_obs_t[t_idx]).squeeze(-1)
                tgt_ret  = ret_flat[idx]
                critic_loss = nn.MSELoss()(pred_v, tgt_ret)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_updates += 1

        return total_actor_loss / n_updates, total_critic_loss / n_updates

    def store_transition(self, obs, actions, log_probs, entropy, rewards, dones, action_masks=None):
        """Store transition for experience replay."""
        # obs is a dict {rid: obs_vec}
        # Get sorted robot IDs (excluding "__all__")
        robot_ids = sorted([rid for rid in obs.keys() if rid != "__all__"])
        
        obs_list = [obs[rid] for rid in robot_ids]
        joint_obs = np.concatenate(obs_list)
        
        action_list = [actions.get(rid, 0) for rid in robot_ids]
        log_prob_list = [log_probs.get(rid, 0.0) for rid in robot_ids]
        entropy_list = [entropy.get(rid, 0.0) for rid in robot_ids]
        reward_list = [rewards.get(rid, 0.0) for rid in robot_ids]
        done_list = [dones.get(rid, False) for rid in robot_ids]
        
        self.obs_history.append(obs_list)
        self.joint_obs_history.append(joint_obs)
        self.action_history.append(action_list)
        self.log_prob_history.append(log_prob_list)
        self.reward_history.append(reward_list)
        self.done_history.append(done_list)
        self.action_mask_history.append(action_masks)

    def update(self):
        """Perform policy update using collected experience."""
        if len(self.obs_history) == 0:
            return
        
        actor_loss, critic_loss = self.train_step(
            self.obs_history,
            self.joint_obs_history,
            self.action_history,
            self.reward_history,
            self.log_prob_history,
            self.done_history,
        )
        
        # Clear buffers
        self.obs_history = []
        self.joint_obs_history = []
        self.action_history = []
        self.reward_history = []
        self.log_prob_history = []
        self.done_history = []
        self.action_mask_history = []
        
        return actor_loss, critic_loss

    def save(self, checkpoint_path):
        """Save model checkpoint."""
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'optimizer_actor_state': self.optimizer_actor.state_dict(),
            'optimizer_critic_state': self.optimizer_critic.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic.load_state_dict(checkpoint['critic_state'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state'])

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
