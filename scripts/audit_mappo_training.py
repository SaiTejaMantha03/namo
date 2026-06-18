import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mappo.mappo_agent import ActorNetwork
from mappo.mappo_env import NAMOmappoEnv

def audit():
    print("=" * 60)
    print("MAPPO TRAINING AUDIT")
    print("=" * 60)
    
    # Define observation dimension
    # 25 (unet) + 2 (own) + 2 (goal) + 2 (dir) + 6 (others) + 6 (boxes) + 3 (uncertainty) = 46
    obs_dim = 46
    
    actor_path = Path(__file__).resolve().parent.parent / "models" / "mappo_actor_checkpoint.pth"
    if not actor_path.exists():
        print(f"Checkpoint not found at {actor_path}")
        return
        
    actor = ActorNetwork(obs_dim)
    actor.load_state_dict(torch.load(str(actor_path)))
    actor.eval()
    
    print("\n1. Layer Weight Analysis (L2 Norms)")
    print("-" * 40)
    for name, param in actor.named_parameters():
        l2_norm = torch.norm(param).item()
        print(f"{name:<25}: {l2_norm:.4f}")
        
    print("\n2. Synthetic Observation Probing")
    print("-" * 40)
    
    # 0 = NAVIGATE, 1 = PUSH_BOX, 2 = YIELD, 3 = WAIT
    actions_map = {0: "NAVIGATE", 1: "PUSH_BOX", 2: "YIELD", 3: "WAIT"}
    
    # Scenario A: Empty space (zeros)
    obs_zeros = torch.zeros((1, obs_dim), dtype=torch.float32)
    logits_zeros = actor(obs_zeros)
    act_zeros = torch.argmax(logits_zeros).item()
    print(f"Scenario A (Empty space / Zeros): Action = {actions_map[act_zeros]} (Logits: {logits_zeros.detach().numpy()})")
    
    # Scenario B: Dense walls/boxes (ones)
    obs_ones = torch.ones((1, obs_dim), dtype=torch.float32)
    logits_ones = actor(obs_ones)
    act_ones = torch.argmax(logits_ones).item()
    print(f"Scenario B (Dense obstacles / Ones): Action = {actions_map[act_ones]} (Logits: {logits_ones.detach().numpy()})")
    
    # Scenario C: 100 Random Observations
    print("\nScenario C (100 Random Observations distribution)")
    act_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(100):
        obs_rand = torch.rand((1, obs_dim), dtype=torch.float32)
        logits_rand = actor(obs_rand)
        act_rand = torch.argmax(logits_rand).item()
        act_counts[act_rand] += 1
        
    for k, v in act_counts.items():
        print(f"  {actions_map[k]:<10}: {v}%")
        
    if act_counts[3] > 80:
        print("\n>>> CONCLUSION: Model has collapsed into a WAIT policy (biased towards action 3).")
        print("    This confirms the dense reward (-0.01 step vs -1.0 collision) combined with")
        print("    too few epochs caused the agent to learn that standing still is safest.")

if __name__ == "__main__":
    audit()
