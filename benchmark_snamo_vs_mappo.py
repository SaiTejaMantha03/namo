import sys
import argparse
import subprocess
import re
from pathlib import Path

ALL_CONFIGS = [
    ("namo_push_only",                  "configs/namo_push_only.yaml"),
    ("movable_obstacle_choke_namo",     "configs/movable_obstacle_choke_namo.yaml"),
    ("warehouse_small",                 "configs/warehouse_small.yaml"),
    ("warehouse_3robots",               "configs/warehouse_3robots.yaml"),
    ("warehouse_large",                 "configs/warehouse_large.yaml"),
    ("single_corridor_yielding",        "configs/single_corridor_yielding.yaml"),
    ("symmetric_bottleneck_deadlock",   "configs/symmetric_bottleneck_deadlock.yaml"),
    ("narrow_doorway_congestion",       "configs/narrow_doorway_congestion.yaml"),
    ("cross_intersection",              "configs/cross_intersection_coordination.yaml"),
    ("symmetric_bottleneck_4robots",    "configs/symmetric_bottleneck_4robots.yaml"),
    ("custom_reconstructed_map_robots", "configs/custom_reconstructed_map_robots.yaml"),
]

def run_snamo_sub(cfg_path, gui=True):
    cmd = [sys.executable, "simulation/snamo_simulator.py", "--config", cfg_path, "--dr-strategy", "sr_social"]
    if gui:
        cmd.append("--gui")
        
    print(f"    Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output
    steps = 9999
    success = False
    
    # Match: [SNAMOSim] Steps=7  Status=SUCCESS  Pushes=...
    match = re.search(r"\[SNAMOSim\]\s+Steps=(\d+)\s+Status=(\w+)", res.stdout)
    if match:
        steps = int(match.group(1))
        status = match.group(2)
        success = (status == "SUCCESS")
    else:
        # Check stderr or check if output contains errors
        if "Only one local" in res.stdout or "Only one local" in res.stderr:
            print("    [WARN] PyBullet GUI connection limit hit.")
        else:
            print(f"    [WARN] S-NAMO output parse fail. stdout: {res.stdout[-200:] if res.stdout else ''}")
            
    return success, steps

def run_mappo_sub(cfg_path, checkpoint_path, gui=True):
    cmd = [sys.executable, "run_mappo_gui.py", "--config", cfg_path, "--checkpoint", checkpoint_path]
    if not gui:
        cmd.append("--no-gui")
        
    print(f"    Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output
    steps = 9999
    success = False
    
    # Match: [MAPPOSim] Steps=210 Status=SUCCESS
    match = re.search(r"\[MAPPOSim\]\s+Steps=(\d+)\s+Status=(\w+)", res.stdout)
    if match:
        steps = int(match.group(1))
        status = match.group(2)
        success = (status == "SUCCESS")
    else:
        print(f"    [WARN] MAPPO output parse fail. stdout: {res.stdout[-200:] if res.stdout else ''}")
        
    return success, steps

def main():
    parser = argparse.ArgumentParser(description="Benchmark S-NAMO vs MAPPO using Subprocesses")
    parser.add_argument("--checkpoint", default="checkpoints/v3_maxres/mappo_final.pth",
                        help="Path to trained MAPPO checkpoint")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    
    gui = not args.no_gui
    results = []
    
    print("\n" + "="*80)
    print(f" BENCHMARKING S-NAMO vs MAPPO (GUI={gui}) via Subprocesses")
    print("="*80)
    
    for name, cfg_path in ALL_CONFIGS:
        if not Path(cfg_path).exists():
            print(f"[SKIP] {name} — Config file not found at {cfg_path}")
            continue
            
        print(f"\nEvaluating scenario: {name}")
        
        # 1. Run S-NAMO
        print(f"  -> S-NAMO:")
        snamo_ok, snamo_steps = run_snamo_sub(cfg_path, gui=gui)
            
        # 2. Run MAPPO
        print(f"  -> MAPPO:")
        mappo_ok, mappo_steps = run_mappo_sub(cfg_path, args.checkpoint, gui=gui)
            
        results.append({
            "name": name,
            "snamo_ok": snamo_ok,
            "snamo_steps": snamo_steps,
            "mappo_ok": mappo_ok,
            "mappo_steps": mappo_steps
        })
        
        # Print intermediate comparison
        s_steps_str = f"{snamo_steps} steps" if snamo_ok else "FAIL"
        m_steps_str = f"{mappo_steps} steps" if mappo_ok else "FAIL"
        print(f"  Results — S-NAMO: {s_steps_str} | MAPPO: {m_steps_str}")
        
    # Print comparison table
    print("\n" + "="*95)
    print(f"  {'Scenario':<32} | {'S-NAMO SR':<9} | {'S-NAMO Steps':<12} | {'MAPPO SR':<8} | {'MAPPO Steps':<11} | {'Winner':<8}")
    print("="*95)
    
    table_lines = []
    table_lines.append("# S-NAMO vs MAPPO Benchmarking Results")
    table_lines.append("")
    table_lines.append("| Scenario | S-NAMO SR | S-NAMO Steps (Physics) | MAPPO SR | MAPPO Steps (Physics) | Winner | Speedup |")
    table_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    for r in results:
        s_ok = "✅ PASS" if r["snamo_ok"] else "❌ FAIL"
        m_ok = "✅ PASS" if r["mappo_ok"] else "❌ FAIL"
        s_steps = f"{r['snamo_steps']}" if r["snamo_ok"] else "—"
        m_steps = f"{r['mappo_steps']}" if r["mappo_ok"] else "—"
        
        # Determine winner
        if r["snamo_ok"] and r["mappo_ok"]:
            if r["snamo_steps"] < r["mappo_steps"]:
                winner = "S-NAMO"
                ratio = r["mappo_steps"] / r["snamo_steps"]
                speedup = f"S-NAMO ({ratio:.1f}x)"
            else:
                winner = "MAPPO"
                ratio = r["snamo_steps"] / r["mappo_steps"]
                speedup = f"MAPPO ({ratio:.1f}x)"
        elif r["snamo_ok"]:
            winner = "S-NAMO"
            speedup = "S-NAMO (MAPPO FAIL)"
        elif r["mappo_ok"]:
            winner = "MAPPO"
            speedup = "MAPPO (S-NAMO FAIL)"
        else:
            winner = "None"
            speedup = "Both Fail"
            
        print(f"  {r['name']:<32} | {s_ok:<9} | {s_steps:<12} | {m_ok:<8} | {m_steps:<11} | {winner:<8}")
        
        row = f"| {r['name']} | {s_ok} | {s_steps} | {m_ok} | {m_steps} | {winner} | {speedup} |"
        table_lines.append(row)
        
    print("="*95)
    
    # Save table
    out_path = Path("results/snamo_vs_mappo_benchmark.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(table_lines))
    print(f"\nBenchmark results saved to: {out_path}")

if __name__ == "__main__":
    main()
