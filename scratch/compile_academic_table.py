import os
import re
import json
import yaml
import numpy as np
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "configs"
RESULTS_DIR = PROJECT_DIR / "results"
EVAL_TABLES_DIR = RESULTS_DIR / "evaluation_tables"

# Log files for Pure S-NAMO and S-NAMO*
PURE_SNAMO_LOG = Path("/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/.system_generated/tasks/task-307.log")
SNAMO_UNC_LOG = Path("/Users/saitejamantha/.gemini/antigravity-ide/brain/113738b1-7ea8-4e59-a814-b93890fe0f4b/.system_generated/tasks/task-309.log")
MAPPO_V4_JSON = EVAL_TABLES_DIR / "v4_eval_results.json"

# Scenario details and config mapping
BENCHMARK_CONFIGS = [
    ("movable_obstacle_choke_namo",   "movable_obstacle_choke_namo.yaml"),
    ("warehouse_small",               "warehouse_small.yaml"),
    ("warehouse_3robots",             "warehouse_3robots.yaml"),
    ("single_corridor_yielding",      "single_corridor_yielding.yaml"),
    ("symmetric_bottleneck_deadlock", "symmetric_bottleneck_deadlock.yaml"),
    ("cross_intersection",            "cross_intersection_coordination.yaml"),
    ("warehouse_large",               "warehouse_large.yaml"),
    ("narrow_doorway_congestion",     "narrow_doorway_congestion.yaml"),
    ("symmetric_bottleneck_4robots",  "symmetric_bottleneck_4robots.yaml"),
]

def get_robot_count(config_file):
    try:
        with open(CONFIG_DIR / config_file) as f:
            cfg = yaml.safe_load(f)
            return len(cfg.get("robots", []))
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return 0

def parse_snamo_log(log_path):
    """
    Parses S-NAMO log and returns a dictionary:
    {
       scenario_name: {
          'successes': [1, 0, 1...],
          'steps': [200, 1500, 250...],
          'pushes': [1, 0, 0...]
       }
    }
    """
    if not log_path.exists():
        print(f"Log path does not exist: {log_path}")
        return {}

    results = {}
    current_scenario = None

    with open(log_path) as f:
        for line in f:
            # Detect scenario
            m_scen = re.search(r"Evaluating scenario:\s+(\S+)", line)
            if m_scen:
                current_scenario = m_scen.group(1)
                results[current_scenario] = {
                    "successes": [],
                    "steps": [],
                    "pushes": []
                }
                continue

            # Detect trial results
            # [SNAMOSim] Steps=200  Status=SUCCESS  Pushes=1  Collisions=0
            m_trial = re.search(r"\[SNAMOSim\] Steps=(\d+)\s+Status=(\S+)\s+Pushes=(\d+)", line)
            if m_trial and current_scenario:
                steps = int(m_trial.group(1))
                status = m_trial.group(2)
                pushes = int(m_trial.group(3))
                
                success = 1 if status == "SUCCESS" else 0
                results[current_scenario]["successes"].append(success)
                results[current_scenario]["steps"].append(steps)
                results[current_scenario]["pushes"].append(pushes)

    return results

def compute_stats(successes, steps, pushes, is_snamo_log=False):
    """
    Computes success rate, makespan (control steps), and pushes (transfers)
    mean and stddev over 50 trials.
    """
    # Convert steps to control steps if log from Pure S-NAMO / S-NAMO*
    if is_snamo_log:
        ctrl_steps = [s / 15.0 for s in steps]
    else:
        ctrl_steps = steps

    # Success rate: mean and stddev over 5 folds of 10 trials
    folds = []
    for i in range(5):
        fold_trials = successes[i*10 : (i+1)*10]
        if fold_trials:
            folds.append(sum(fold_trials) / len(fold_trials))
    
    sr_mean = np.mean(folds) if folds else 0.0
    sr_std = np.std(folds) if folds else 0.0

    steps_mean = np.mean(ctrl_steps) if ctrl_steps else 0.0
    steps_std = np.std(ctrl_steps) if ctrl_steps else 0.0

    pushes_mean = np.mean(pushes) if pushes else 0.0
    pushes_std = np.std(pushes) if pushes else 0.0

    return {
        "sr_mean": sr_mean,
        "sr_std": sr_std,
        "steps_mean": steps_mean,
        "steps_std": steps_std,
        "pushes_mean": pushes_mean,
        "pushes_std": pushes_std
    }

def format_value(mean, std, is_percentage=False, places=2):
    if is_percentage:
        # For Success Rate: format as percentage (e.g. 98.0% or 0.98)
        # The reference image formats success rate as decimals (e.g., 0.97 \pm 0.10)
        # Let's format as decimal first to match the reference image.
        return f"${mean:.2f} \\pm {std:.2f}$"
    else:
        if std == 0:
            return f"${mean:.1f} \\pm 0.0$"
        return f"${mean:.2f} \\pm {std:.2f}$" if places == 2 else f"${mean:.1f} \\pm {std:.1f}$"

def main():
    print("Parsing logs...")
    pure_snamo_data = parse_snamo_log(PURE_SNAMO_LOG)
    snamo_unc_data = parse_snamo_log(SNAMO_UNC_LOG)

    # Load MAPPO v4 JSON
    if not MAPPO_V4_JSON.exists():
        print(f"Waiting for MAPPO v4 JSON: {MAPPO_V4_JSON}")
        return

    with open(MAPPO_V4_JSON) as f:
        mappo_v4_data = json.load(f)

    # Scenarios grouping mapping
    # Let's map scenario names to display names
    scen_display_names = {
        "movable_obstacle_choke_namo": "Movable Obstacle Choke (NAMO)",
        "warehouse_small": "Warehouse Small",
        "warehouse_3robots": "Warehouse 3 Robots",
        "single_corridor_yielding": "Single Corridor Yielding",
        "symmetric_bottleneck_deadlock": "Symmetric Bottleneck Deadlock",
        "cross_intersection": "Cross Intersection",
        "warehouse_large": "Warehouse Large",
        "narrow_doorway_congestion": "Narrow Doorway Congestion",
        "symmetric_bottleneck_4robots": "Symmetric Bottleneck 4 Robots"
    }

    output_md = ""

    for name, cfg_file in BENCHMARK_CONFIGS:
        nb_rob = get_robot_count(cfg_file)
        display_name = scen_display_names.get(name, name)
        
        # Get data
        p_data = pure_snamo_data.get(name)
        s_data = snamo_unc_data.get(name)
        m_data = mappo_v4_data.get(name)

        if not p_data or not s_data or not m_data:
            print(f"Missing data for {name} - skipping table generation.")
            continue

        p_stats = compute_stats(p_data["successes"], p_data["steps"], p_data["pushes"], is_snamo_log=True)
        s_stats = compute_stats(s_data["successes"], s_data["steps"], s_data["pushes"], is_snamo_log=True)
        
        # For MAPPO v4, check if raw list fields exist
        if "raw_successes" in m_data:
            m_stats = compute_stats(m_data["raw_successes"], m_data["raw_steps"], m_data["raw_pushes"], is_snamo_log=False)
        else:
            # Fallback if no raw tracking (only averages available)
            print(f"Warning: raw_successes not found in MAPPO v4 data for {name}. Using fallback.")
            m_stats = {
                "sr_mean": m_data["sr"] / 100.0, "sr_std": 0.0,
                "steps_mean": m_data["avg_steps"], "steps_std": 0.0,
                "pushes_mean": m_data.get("push_pct", 0.0) / 100.0 * m_data["avg_steps"] * nb_rob, "pushes_std": 0.0
            }

        # Header for the map table
        output_md += f"### Map: `{name}` ({display_name})\n\n"
        output_md += f"| Method | Succ. Rate | Dist. | nb. Transf. | makesp. | Plan. time |\n"
        output_md += f"| :--- | :---: | :---: | :---: | :---: | :---: |\n"

        # Format methods
        p_row = f"| **Pure S-NAMO** | {format_value(p_stats['sr_mean'], p_stats['sr_std'], True)} | - | {format_value(p_stats['pushes_mean'], p_stats['pushes_std'])} | {format_value(p_stats['steps_mean'], p_stats['steps_std'])} | - |"
        s_row = f"| **S-NAMO\*** | {format_value(s_stats['sr_mean'], s_stats['sr_std'], True)} | - | {format_value(s_stats['pushes_mean'], s_stats['pushes_std'])} | {format_value(s_stats['steps_mean'], s_stats['steps_std'])} | - |"
        m_row = f"| **MAPPO v4** | {format_value(m_stats['sr_mean'], m_stats['sr_std'], True)} | - | {format_value(m_stats['pushes_mean'], m_stats['pushes_std'])} | {format_value(m_stats['steps_mean'], m_stats['steps_std'])} | - |"

        output_md += p_row + "\n"
        output_md += s_row + "\n"
        output_md += m_row + "\n\n"

    print("\nGenerated Markdown:\n")
    print(output_md)

    # Save to a file for easy copying/reading
    with open(PROJECT_DIR / "scratch" / "generated_academic_tables.md", "w") as f:
        f.write(output_md)
    print(f"Generated Markdown tables saved to scratch/generated_academic_tables.md")

if __name__ == "__main__":
    main()
