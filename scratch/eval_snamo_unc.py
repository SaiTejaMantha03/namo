"""
scratch/eval_snamo_unc.py
-------------------------
Evaluates our re-implemented S-NAMO* baseline (with uncertainty enabled, pure_snamo=False)
across 10 trials per scenario to check if the results match the hardcoded values.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    trials = 10
    results = {}
    project_dir = Path(__file__).resolve().parent.parent

    print("\n" + "=" * 70)
    print(f"  S-NAMO* (Uncertainty Enabled) Empirical Evaluation  |  trials={trials}")
    print("=" * 70)

    for name, cfg_rel in BENCHMARK_CONFIGS:
        cfg_path = str(project_dir / cfg_rel)
        if not Path(cfg_path).exists():
            print(f"[SKIP] {name} — config not found: {cfg_path}")
            continue

        successes = []
        steps_list = []

        print(f"\nEvaluating scenario: {name}")
        for t in range(trials):
            try:
                # Run with pure_snamo=False (default uncertainty enabled)
                res = run_simulation(cfg_path, gui=False, pure_snamo=False)
                successes.append(int(res["success"]))
                steps_list.append(res["steps"] / 15.0)
            except Exception as e:
                successes.append(0)
                steps_list.append(100.0)

        sr = 100.0 * sum(successes) / len(successes) if successes else 0.0
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0.0

        results[name] = {
            "sr": sr,
            "avg_steps": round(avg_steps, 2)
        }
        print(f"-> {name:<35} SR={sr:5.1f}%  Steps={avg_steps:6.1f}")

    print("\nEmpirical S-NAMO* Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
