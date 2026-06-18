import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.snamo_simulator import ALL_CONFIGS, run_simulation
from mappo.mappo_env import NAMOmappoEnv
from mappo.mappo_agent import MAPPOAgent
import torch
import numpy as np
import pybullet as p

def evaluate_mappo_on_map(cfg_path, gui=True, max_steps=200):
    try:
        env = NAMOmappoEnv(cfg_path, gui=gui, max_steps=max_steps)
        obs = env.reset()
        obs_dim = next(iter(obs.values())).shape[0]
        num_agents = len(obs)
        
        agent = MAPPOAgent(obs_dim, num_agents)
        actor_path = Path(__file__).resolve().parent.parent / "models" / "mappo_actor_checkpoint.pth"
        if actor_path.exists():
            agent.actor.load_state_dict(torch.load(str(actor_path)))
        agent.actor.eval()
        
        collisions = 0
        consecutive_waits = 0
        real_success = False
        
        for t in range(max_steps):
            actions = {}
            all_wait = True
            for rid, agent_obs in obs.items():
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32)
                with torch.no_grad():
                    logits = agent.actor(obs_tensor)
                    action = torch.argmax(logits).item()
                actions[rid] = action
                if action != 3:
                    all_wait = False
                    
            if all_wait:
                consecutive_waits += 1
            else:
                consecutive_waits = 0
                
            next_obs, rewards, dones, info = env.step(actions)
            collisions += info["collisions"]
            obs = next_obs
            
            if consecutive_waits > 10:
                print(">>> WARNING: MAPPO frozen (all agents outputting WAIT)")
                break
                
            if dones["__all__"]:
                real_success = True
                for rid in env.robot_ids:
                    pos, _ = p.getBasePositionAndOrientation(rid)
                    g = env.sim.robot_goals[rid]
                    dist = np.hypot(pos[0] - (g[0]+0.5)*env.cell_size, pos[1] - (g[1]+0.5)*env.cell_size)
                    if dist > 0.6:
                        real_success = False
                        break
                break
                
        env.close()
        return {
            "success": real_success,
            "stalled": consecutive_waits > 10,
            "steps": t,
            "collisions": collisions
        }
    except Exception as e:
        print(f"Error evaluating MAPPO on {cfg_path}: {e}")
        try:
            env.close()
        except: pass
        return {"success": False, "stalled": False, "steps": 0, "collisions": 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--interactive", action="store_true", help="Pause for input between maps")
    args = parser.parse_args()
    
    gui = not args.headless
    results = []
    
    print("=" * 100)
    print("Running Benchmark for SNAMO and MAPPO across all configs")
    print("=" * 100)
    
    for cfg in ALL_CONFIGS:
        name = cfg.split("/")[-1]
        print(f"\n--- Testing Map: {name} ---")
        
        if args.interactive:
            input(f"Press Enter to run SNAMO on {name}...")
            
        print("Running SNAMO...")
        snamo_res = run_simulation(cfg, gui=gui, dr_strategy="sr_social")
        
        if args.interactive:
            input(f"Press Enter to run MAPPO on {name}...")
            
        print("Running MAPPO...")
        mappo_res = evaluate_mappo_on_map(cfg, gui=gui, max_steps=400)
        
        results.append({
            "map": name,
            "snamo": snamo_res,
            "mappo": mappo_res
        })
        
    print("\n\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Map Config':<40} | {'SNAMO Status':<12} | {'SNAMO Steps':<11} | {'MAPPO Status':<12} | {'MAPPO Steps'}")
    print("-" * 100)
    for r in results:
        s_status = "SUCCESS" if r["snamo"]["success"] else ("STALLED" if r["snamo"].get("stalled", False) else "FAILED")
        m_status = "SUCCESS" if r["mappo"]["success"] else ("STALLED" if r["mappo"].get("stalled", False) else "FAILED")
        
        s_steps = f"{r['snamo']['steps']} (C:{r['snamo']['collisions']})"
        m_steps = f"{r['mappo']['steps']} (C:{r['mappo']['collisions']})"
        
        print(f"{r['map']:<40} | {s_status:<12} | {s_steps:<11} | {m_status:<12} | {m_steps}")
        
if __name__ == "__main__":
    main()
