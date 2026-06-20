"""
eval_pure_snamo.py
------------------
Runs the Pure S-NAMO baseline (social costmap + coordinator, but uncertainty model disabled)
across all 9 benchmark configs, for a given number of trials (default: 10).
Saves results to results/evaluation_tables/pure_snamo_eval_results.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation.snamo_simulator import run_simulation

BENCHMARK_CONFIGS = [
    ("movable_obstacle_choke_namo",   "configs/movable_obstacle_choke_namo.yaml"),
    ("warehouse_small",               "configs/warehouse_small.yaml"),
    ("warehouse_3robots",             "configs/warehouse_3robots.yaml"),
    ("single_corridor_yielding",      "configs/single_corridor_yielding.yaml"),
    ("symmetric_bottleneck_deadlock", "configs/symmetric_bottleneck_deadlock.yaml"),
    ("cross_intersection",            "configs/cross_intersection_coordination.yaml"),
    ("warehouse_large",               "configs/warehouse_large.yaml"),
    ("narrow_doorway_congestion",     "configs/narrow_doorway_congestion.yaml"),
    ("symmetric_bottleneck_4robots",  "configs/symmetric_bottleneck_4robots.yaml"),
]

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pure S-NAMO (Deterministic) Baseline")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per scenario")
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    args = parser.parse_args()

    results = {}
    project_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 70)
    print(f"  Pure S-NAMO Batch Evaluation  |  trials={args.trials}  gui={args.gui}")
    print("=" * 70)

    for name, cfg_rel in BENCHMARK_CONFIGS:
        cfg_path = str(project_dir / cfg_rel)
        if not Path(cfg_path).exists():
            print(f"[SKIP] {name} — config not found: {cfg_path}")
            continue

        successes = []
        steps_list = []
        pushes_list = []
        collisions_list = []

        print(f"\nEvaluating scenario: {name}")
        for t in range(args.trials):
            print(f"  Trial {t+1}/{args.trials}...", end="", flush=True)
            try:
                # Run with pure_snamo=True
                res = run_simulation(cfg_path, gui=args.gui, pure_snamo=True)
                successes.append(int(res["success"]))
                steps_list.append(res["steps"] / 15.0)
                pushes_list.append(res["pushes"])
                collisions_list.append(res["collisions"])
                print(f" {'Success' if res['success'] else 'Failed'} (ctrl_steps={res['steps'] / 15.0:.1f})")
            except Exception as e:
                print(f" Error: {e}")
                # Log as failed
                successes.append(0)
                steps_list.append(100.0)  # 1500 / 15 = 100
                pushes_list.append(0)
                collisions_list.append(0)

        sr = 100.0 * sum(successes) / len(successes) if successes else 0.0
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0.0
        avg_pushes = sum(pushes_list) / len(pushes_list) if pushes_list else 0.0
        avg_collisions = sum(collisions_list) / len(collisions_list) if collisions_list else 0.0

        results[name] = {
            "sr": sr,
            "avg_steps": avg_steps,
            "avg_pushes": avg_pushes,
            "avg_collisions": avg_collisions
        }

        status = "✅" if sr >= 80.0 else ("⚠️" if sr >= 50.0 else "❌")
        print(f"-> {status} {name:<35} SR={sr:5.1f}%  Steps={avg_steps:6.1f}  Pushes={avg_pushes:.1f}  Collisions={avg_collisions:.1f}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"  {'Scenario':<40} {'SR':>6} {'Avg Steps':>10}")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:<40} {r['sr']:>5.1f}%  {r['avg_steps']:>8.1f}")
    print("=" * 70)

    # Save to JSON
    out_dir = project_dir / "results" / "evaluation_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pure_snamo_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Pure S-NAMO Eval] Results saved to {out_path}")

if __name__ == "__main__":
    main()
