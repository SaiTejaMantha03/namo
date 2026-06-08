import sys
from pathlib import Path

# Add project root and simulation directory for imports
sys.path.append(str(Path(__file__).resolve().parent))

from simulation.namo_simulator import simulate_env

def main():
    scenarios = [
        {"name": "single_corridor_yielding", "config": "configs/single_corridor_yielding.yaml", "purpose": "yielding"},
        {"name": "symmetric_bottleneck_deadlock", "config": "configs/symmetric_bottleneck_deadlock.yaml", "purpose": "deadlock"},
        {"name": "cross_intersection_coordination", "config": "configs/cross_intersection_coordination.yaml", "purpose": "coordination"},
        {"name": "movable_obstacle_choke_namo", "config": "configs/movable_obstacle_choke_namo.yaml", "purpose": "NAMO decision"},
        {"name": "narrow_doorway_congestion", "config": "configs/narrow_doorway_congestion.yaml", "purpose": "congestion"},
    ]
    
    results = []
    
    print("Starting NAMO Scenario Evaluations...")
    print("="*60)
    for s in scenarios:
        config_path = str(Path(__file__).resolve().parent / s["config"])
        print(f"\nRunning Scenario: {s['name']} ({s['purpose']})")
        print(f"Config path: {config_path}")
        
        # Run simulation in headless (p.DIRECT) mode
        res = simulate_env(config_path, gui=False)
        
        results.append({
            "scenario": s["name"],
            "purpose": s["purpose"],
            "success": res["success"],
            "steps": res["steps"],
            "pushes": res["pushes"],
            "collisions": res["collisions"]
        })
        
    print("\n" + "="*60)
    print("EVALUATION COMPLETED. COMPILING RESULTS TABLE...")
    print("="*60)
    
    table_lines = [
        "| Scenario | Purpose | Success | Steps Taken | Obstacle Pushes | Collisions |",
        "|:---|:---|:---:|:---:|:---:|:---:|"
    ]
    for r in results:
        success_str = "✅ Yes" if r["success"] else "❌ No"
        row = f"| {r['scenario']} | {r['purpose']} | {success_str} | {r['steps']} | {r['pushes']} | {r['collisions']} |"
        table_lines.append(row)
        
    table_md = "\n".join(table_lines)
    
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "scenario_results_table.md"
    results_file.write_text(table_md)
    
    print(f"\nSaved scenario results table to: results/scenario_results_table.md\n")
    print(table_md)

if __name__ == '__main__':
    main()
