"""
run_coordination_eval.py
--------------------------
Evaluation harness for multi-robot coordination.

Loads a YAML scenario config, runs the RobotCoordinator for all robots,
and reports:
  - Success rate (all robots reach goals)
  - Steps to completion
  - Deadlock frequency
  - Conflict resolution counts by type
  - Belief broadcasting SR interval evolution (when --belief-sharing is set)

Usage
-----
    python run_coordination_eval.py --config configs/symmetric_bottleneck_deadlock.yaml
    python run_coordination_eval.py --config configs/warehouse_3robots.yaml \
        --resolution sr_width --belief-sharing --episodes 20
"""

import sys
import argparse
import random
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maps.namo_environments import WarehouseEnvironment, NAMOEnvironment
from multi_robot.coordinator import RobotCoordinator, RobotState
from multi_robot.deadlock_resolution import DeadlockResolver
from multi_robot.conflict_detection import ConflictDetector
from multi_robot.belief_broadcaster import BeliefBroadcaster
from social.social_costmap import SocialCostmap
from social.taboo_zones import TabooZoneManager


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_grid_from_config(config: dict) -> np.ndarray:
    """Build the numpy occupancy grid from a YAML config."""
    world = config.get("world", {})
    grid_size = world.get("grid_size", 20)
    layout_type = config.get("layout", {}).get("type", "")

    if layout_type == "warehouse":
        env = WarehouseEnvironment(grid_size=grid_size)
    else:
        env = NAMOEnvironment(grid_size=grid_size)

    for obs in config.get("obstacles", []):
        pos = obs.get("pos", obs.get("position", None))
        if pos:
            env.add_obstacle(pos[0], pos[1])

    # Load explicit wall cells defined in YAML
    for wall in config.get("walls", []):
        pos = wall.get("pos", wall.get("position", None))
        if pos:
            env.add_wall(pos[0], pos[1])

    return env.generate_occupancy_grid()


def run_episode(
    config: dict,
    grid: np.ndarray,
    dr_strategy: str,
    belief_sharing: bool,
    max_steps: int = 300,
    seed: int = 0,
) -> dict:
    """Run a single coordination episode. Returns result dict."""
    random.seed(seed)
    np.random.seed(seed)

    world = config.get("world", {})
    grid_size = world.get("grid_size", 20)
    robots_cfg = config.get("robots", [])

    # Build RobotState objects
    states: dict[int, RobotState] = {}
    for r in robots_cfg:
        rid = r["id"]
        start = tuple(r["start"])
        goal  = tuple(r["goal"])
        states[rid] = RobotState(robot_id=rid, cell=start, goal=goal)

    robot_ids = list(states.keys())

    # Box states (static for now — no manipulation in coord eval)
    box_states: dict[int, tuple] = {}
    for i, obs in enumerate(config.get("obstacles", [])):
        pos = obs.get("pos", obs.get("position", None))
        if pos:
            box_states[i] = tuple(pos)

    # Build components
    resolver = DeadlockResolver(grid, grid_size)

    broadcaster = None
    if belief_sharing:
        broadcaster = BeliefBroadcaster(robot_ids)

    social_map = None
    if dr_strategy == "social":
        social_map = SocialCostmap(grid)
        taboo_cfg = config.get("social", {}).get("taboo_zones", [])
        if taboo_cfg:
            taboo = TabooZoneManager(taboo_cfg, grid_size)
            # Mark taboo cells in social map
            for cell in taboo.blocked_cells():
                col, row = cell
                social_map._map[row, col] = 1.0

    coordinator = RobotCoordinator(
        grid=grid,
        grid_size=grid_size,
        h=10,
        dr_strategy=dr_strategy,
        resolver=resolver,
        broadcaster=broadcaster,
        social_map=social_map,
    )

    conflict_counts = {ct.name: 0 for ct in __import__(
        "multi_robot.conflict_detection", fromlist=["ConflictType"]).ConflictType}
    deadlock_count = 0
    steps_taken = 0
    success = False

    detector = ConflictDetector()

    for step in range(max_steps):
        steps_taken = step + 1

        # Count conflicts this step
        robot_plans = {rid: (states[rid].plan or [states[rid].cell]) for rid in robot_ids}
        robot_states_cd = {
            rid: {"cell": states[rid].cell, "active_box": None, "planned_obstacle_cell": None}
            for rid in robot_ids
        }
        conflicts = detector.detect(robot_plans, box_states, robot_states_cd, h=5)
        for c in conflicts:
            conflict_counts[c.conflict_type.name] += 1
            deadlock_count += 1

        # Step all robots
        coordinator.step(states, box_states)

        # Check termination
        all_done = all(s.status == "DONE" or s.cell == s.goal for s in states.values())
        if all_done:
            success = True
            break

    return {
        "success": success,
        "steps": steps_taken,
        "deadlock_count": deadlock_count,
        "conflict_counts": conflict_counts,
        "broadcaster_summary": broadcaster.summary() if broadcaster else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-robot NAMO coordination evaluator")
    parser.add_argument("--config", default="configs/symmetric_bottleneck_deadlock.yaml")
    parser.add_argument("--resolution", choices=["repulsive", "social", "sr_width"],
                        default="repulsive", help="Deadlock resolution strategy")
    parser.add_argument("--belief-sharing", action="store_true",
                        help="Enable cooperative belief sharing (Phase 5)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Multi-Robot NAMO Coordination Evaluation")
    print(f"  Config    : {args.config}")
    print(f"  Strategy  : {args.resolution}")
    print(f"  Belief    : {'shared' if args.belief_sharing else 'isolated'}")
    print(f"  Episodes  : {args.episodes}")
    print(f"{'='*70}\n")

    config = load_config(args.config)
    grid = build_grid_from_config(config)

    results = []
    for ep in range(args.episodes):
        res = run_episode(
            config, grid,
            dr_strategy=args.resolution,
            belief_sharing=args.belief_sharing,
            max_steps=args.max_steps,
            seed=ep,
        )
        results.append(res)
        status = "✓" if res["success"] else "✗"
        print(f"  Episode {ep+1:02d} {status} | Steps: {res['steps']:4d} | "
              f"Deadlocks: {res['deadlock_count']:4d}")

    # Aggregate
    n = len(results)
    success_rate = sum(r["success"] for r in results) / n * 100.0
    avg_steps    = sum(r["steps"] for r in results) / n
    avg_deadlocks= sum(r["deadlock_count"] for r in results) / n

    print(f"\n{'='*70}")
    print(f"  SUMMARY ({args.resolution} DR, {'shared' if args.belief_sharing else 'isolated'} belief)")
    print(f"  Success Rate : {success_rate:.1f}%")
    print(f"  Avg Steps    : {avg_steps:.1f}")
    print(f"  Avg Deadlocks: {avg_deadlocks:.1f}")

    # Conflict breakdown
    all_ct_keys = list(results[0]["conflict_counts"].keys())
    print(f"\n  Conflict Breakdown (avg over {n} episodes):")
    for ct in all_ct_keys:
        avg = sum(r["conflict_counts"][ct] for r in results) / n
        if avg > 0:
            print(f"    {ct:30s}: {avg:.1f}")

    if args.belief_sharing and results[-1]["broadcaster_summary"]:
        print(f"\n  Final Belief State:\n{results[-1]['broadcaster_summary']}")

    print(f"{'='*70}\n")

    # Save results
    out_dir = Path("results/coordination")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_name = Path(args.config).stem
    out_file = out_dir / f"{config_name}_{args.resolution}.txt"
    with open(out_file, "w") as f:
        f.write(f"Config: {args.config}\n")
        f.write(f"Strategy: {args.resolution}\n")
        f.write(f"Belief sharing: {args.belief_sharing}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n")
        f.write(f"Avg steps: {avg_steps:.1f}\n")
        f.write(f"Avg deadlocks: {avg_deadlocks:.1f}\n")
    print(f"  Results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
