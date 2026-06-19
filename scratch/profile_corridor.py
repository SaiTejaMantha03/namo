import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.snamo_simulator import SNAMOSimulator, world_to_cell
import core.planner as planner
import multi_robot.coordinator as coordinator
import multi_robot.conflict_detection as conflict_detection
import multi_robot.deadlock_resolution as deadlock_resolution

# Timing statistics dictionary
stats = {
    "a_star_time": 0.0,
    "a_star_calls": 0,
    "conflict_detection_time": 0.0,
    "conflict_detection_calls": 0,
    "deadlock_resolution_time": 0.0,
    "deadlock_resolution_calls": 0,
    "total_sim_steps": 0,
}

# 1. Monkey patch A* to measure time
original_a_star = planner.a_star
def timed_a_star(*args, **kwargs):
    t_start = time.perf_counter()
    res = original_a_star(*args, **kwargs)
    t_end = time.perf_counter()
    stats["a_star_time"] += (t_end - t_start)
    stats["a_star_calls"] += 1
    return res
planner.a_star = timed_a_star
coordinator.a_star = timed_a_star

# 2. Monkey patch ConflictDetector.detect to measure time
original_detect = conflict_detection.ConflictDetector.detect
def timed_detect(self, *args, **kwargs):
    t_start = time.perf_counter()
    res = original_detect(self, *args, **kwargs)
    t_end = time.perf_counter()
    stats["conflict_detection_time"] += (t_end - t_start)
    stats["conflict_detection_calls"] += 1
    return res
conflict_detection.ConflictDetector.detect = timed_detect

# 3. Monkey patch DeadlockResolver resolution methods
for method_name in ["resolve_sr_social", "resolve_social", "resolve_sr_width", "resolve_repulsive"]:
    if hasattr(deadlock_resolution.DeadlockResolver, method_name):
        original_method = getattr(deadlock_resolution.DeadlockResolver, method_name)
        def make_timed_method(orig):
            def timed_method(self, *args, **kwargs):
                t_start = time.perf_counter()
                res = orig(self, *args, **kwargs)
                t_end = time.perf_counter()
                stats["deadlock_resolution_time"] += (t_end - t_start)
                stats["deadlock_resolution_calls"] += 1
                return res
            return timed_method
        setattr(deadlock_resolution.DeadlockResolver, method_name, make_timed_method(original_method))

def main():
    cfg_path = "configs/single_corridor_yielding.yaml"
    
    t_init_start = time.perf_counter()
    sim = SNAMOSimulator(cfg_path, gui=False, dr_strategy="sr_social")
    sim.reset()
    t_init_end = time.perf_counter()
    init_time = t_init_end - t_init_start
    
    print(f"Starting profile of scenario: {sim.name}")
    print(f"Robots: {sim.robot_ids}")
    
    t_loop_start = time.perf_counter()
    for step in range(300):
        sim.step()
        stats["total_sim_steps"] += 1
        if sim.success:
            print(f"Success in {step} steps!")
            break
    t_loop_end = time.perf_counter()
    
    loop_time = t_loop_end - t_loop_start
    total_time = loop_time + init_time
    
    sim.close()
    
    print("\n" + "="*50)
    print(" COMPUTATION PROFILING RESULTS")
    print("="*50)
    print(f"Total Wall-clock Time      : {total_time:8.4f} seconds")
    print(f"  - PyBullet Init/Setup    : {init_time:8.4f} seconds ({100*init_time/total_time:.1f}%)")
    print(f"  - Simulation Loop Run    : {loop_time:8.4f} seconds ({100*loop_time/total_time:.1f}%)")
    print("-"*50)
    print(" breakdown of calculation time during run:")
    
    a_star_t = stats["a_star_time"]
    cd_t = stats["conflict_detection_time"]
    dr_t = stats["deadlock_resolution_time"]
    
    coordination_total = a_star_t + cd_t + dr_t
    other_loop_overhead = loop_time - coordination_total
    
    print(f"  - A* Path-finding        : {a_star_t:8.4f} seconds ({100*a_star_t/loop_time:.1f}%) | Calls: {stats['a_star_calls']}")
    print(f"  - Conflict Detection     : {cd_t:8.4f} seconds ({100*cd_t/loop_time:.1f}%) | Calls: {stats['conflict_detection_calls']}")
    print(f"  - Deadlock Resolution    : {dr_t:8.4f} seconds ({100*dr_t/loop_time:.1f}%) | Calls: {stats['deadlock_resolution_calls']}")
    print(f"  - Total Coordination Time: {coordination_total:8.4f} seconds ({100*coordination_total/loop_time:.1f}%)")
    print(f"  - Simulator/Physics Step : {other_loop_overhead:8.4f} seconds ({100*other_loop_overhead/loop_time:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
