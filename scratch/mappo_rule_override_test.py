import sys
import torch
import numpy as np
import math
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mappo.mappo_agent import MAPPOAgent
from mappo.mappo_env import NAMOmappoEnv
import pybullet as p
from simulation.snamo_simulator import world_to_cell

SCENARIOS = [
    {"name": "narrow_doorway_congestion", "path": "configs/narrow_doorway_congestion.yaml"},
    {"name": "symmetric_bottleneck_4robots", "path": "configs/symmetric_bottleneck_4robots.yaml"},
    {"name": "cross_intersection", "path": "configs/cross_intersection_coordination.yaml"},
]

def rule_based_override(actions, robot_cells, robot_goals, bottleneck_cell=(7, 7)):
    # Calculate distance to bottleneck for each active robot
    dists = {}
    uncrossed_robots = []
    
    for rid, cell in robot_cells.items():
        goal = robot_goals[rid]
        
        # Vector from bottleneck to robot and goal
        v_rob = (cell[0] - bottleneck_cell[0], cell[1] - bottleneck_cell[1])
        v_goal = (goal[0] - bottleneck_cell[0], goal[1] - bottleneck_cell[1])
        
        # Dot product to check if crossed
        dot = v_rob[0] * v_goal[0] + v_rob[1] * v_goal[1]
        
        # Manhattan distance to bottleneck
        d = abs(cell[0] - bottleneck_cell[0]) + abs(cell[1] - bottleneck_cell[1])
        dists[rid] = d
        
        if dot <= 0:  # Robot has not crossed the bottleneck yet
            uncrossed_robots.append(rid)
            
    # Check if multiple uncrossed robots are near the bottleneck (distance <= 5)
    near_uncrossed = [rid for rid in uncrossed_robots if dists[rid] <= 10]
    
    if len(near_uncrossed) > 1:
        # Sort near uncrossed robots by distance to bottleneck (closest first)
        near_uncrossed.sort(key=lambda rid: (dists[rid], rid))
        
        # The closest uncrossed robot is allowed to NAVIGATE
        allowed_robot = near_uncrossed[0]
        
        new_actions = actions.copy()
        for rid in near_uncrossed:
            if rid == allowed_robot:
                new_actions[rid] = 0  # NAVIGATE
            else:
                new_actions[rid] = 3  # FORCE WAIT
        return new_actions
        
    return actions

def main():
    checkpoint_path = "checkpoints/v3_maxres/mappo_final.pth"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    trials = 5
    
    print("="*60)
    print(" MAPPO CONGESTION RULE-BASED OVERRIDE TEST")
    print("="*60)
    
    for sc in SCENARIOS:
        cfg_path = sc["path"]
        if not Path(cfg_path).exists():
            continue
            
        success_count = 0
        total_steps = 0
        
        for t in range(trials):
            env = NAMOmappoEnv(config_path=cfg_path, gui=False, max_steps=150)
            obs, info = env.reset()
            
            # Setup agent
            sample = next(iter(obs.values()))
            obs_dim = sample.shape[0]
            action_dim = env.action_dim
            agent = MAPPOAgent(
                obs_dim=obs_dim, num_agents=4,
                action_dim=action_dim, device=device
            )
            agent.load(checkpoint_path)
            
            done = False
            steps = 0
            
            while not done and steps < 150:
                action_mask = info.get("action_mask", None)
                with torch.no_grad():
                    actions, _, _ = agent.select_action(obs, action_masks=action_mask)
                
                # Gather actual robot cells and goals from env
                robot_cells = {}
                robot_goals = {}
                for rid in env.robot_ids:
                    pos, _ = p.getBasePositionAndOrientation(rid)
                    cell = world_to_cell(pos[:2], env.cell_size)
                    robot_cells[rid] = cell
                    robot_goals[rid] = env.sim.robot_goals[rid]
                    
                # Apply override
                actions = rule_based_override(actions, robot_cells, robot_goals)
                
                obs, rewards, dones, info = env.step(actions)
                steps += 1
                done = dones.get("__all__", False)
                
            success = info.get("success", False)
            if success:
                success_count += 1
            total_steps += steps
            env.close()
            
        sr = 100.0 * success_count / trials
        avg_steps = float(total_steps) / trials
        print(f"Scenario: {sc['name']:<30} | Success Rate: {sr:5.1f}% | Avg Steps: {avg_steps:5.1f}")
        
if __name__ == "__main__":
    main()
