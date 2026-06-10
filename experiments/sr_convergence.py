"""
experiments/sr_convergence.py
-------------------------------
Experiment: Cooperative Belief Sharing — SR interval convergence rate.

Demonstrates Claim 1:
  A fleet of N robots sharing belief estimates converges on an accurate
  manipulation SR interval N× faster than N robots learning in isolation.

Methodology
-----------
Simulate a sequence of manipulation attempts where the true success rate
is 0.80 (20% failure rate — harder than the prior). Measure how quickly
each model's SR interval width shrinks toward the true value.

- Isolated:   each robot only observes its own attempts (5 each).
- Shared:     all robots share all observations via BeliefBroadcaster.
- Combined:   one robot that saw all 15 attempts (ideal single-agent).

Produces results/experiments/sr_convergence.md (table) and
results/experiments/sr_convergence.png (plot).
"""

import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uncertainty.action_uncertainty import ManipulationBeliefModel
from multi_robot.belief_broadcaster import BeliefBroadcaster

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
TRUE_SR        = 0.80          # true manipulation success rate
N_ROBOTS       = 3             # fleet size
N_ATTEMPTS_EACH = 10           # attempts per robot
RANDOM_SEED    = 42
N_SIGMA        = 2.0           # confidence level for intervals

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

robot_ids = list(range(1, N_ROBOTS + 1))

# Generate ground-truth outcomes for each robot
all_outcomes: dict[int, list[bool]] = {
    rid: [random.random() < TRUE_SR for _ in range(N_ATTEMPTS_EACH)]
    for rid in robot_ids
}

# -----------------------------------------------------------------------
# Scenario A: Isolated (each robot learns alone)
# -----------------------------------------------------------------------
isolated_models = {rid: ManipulationBeliefModel() for rid in robot_ids}
isolated_widths: dict[int, list[float]] = {rid: [] for rid in robot_ids}

for attempt_idx in range(N_ATTEMPTS_EACH):
    for rid in robot_ids:
        isolated_models[rid].observe(all_outcomes[rid][attempt_idx])
        isolated_widths[rid].append(isolated_models[rid].interval_width(N_SIGMA))

# -----------------------------------------------------------------------
# Scenario B: Shared (via BeliefBroadcaster)
# -----------------------------------------------------------------------
broadcaster = BeliefBroadcaster(robot_ids)
shared_widths: dict[int, list[float]] = {rid: [] for rid in robot_ids}

for attempt_idx in range(N_ATTEMPTS_EACH):
    for rid in robot_ids:
        outcome = all_outcomes[rid][attempt_idx]
        broadcaster.broadcast_outcome(rid, outcome)
        shared_widths[rid].append(
            broadcaster.get_sr_interval_width(rid, N_SIGMA)
        )

# -----------------------------------------------------------------------
# Scenario C: Ideal single-agent (sees all N_ROBOTS * N_ATTEMPTS outcomes)
# -----------------------------------------------------------------------
ideal_model = ManipulationBeliefModel()
ideal_widths: list[float] = []

flat_outcomes = []
for attempt_idx in range(N_ATTEMPTS_EACH):
    for rid in robot_ids:
        flat_outcomes.append(all_outcomes[rid][attempt_idx])

for outcome in flat_outcomes:
    ideal_model.observe(outcome)
    ideal_widths.append(ideal_model.interval_width(N_SIGMA))

# -----------------------------------------------------------------------
# Compute summary stats
# -----------------------------------------------------------------------
# Mean width across robots at each attempt index
isolated_mean = np.mean([isolated_widths[r] for r in robot_ids], axis=0)
shared_mean   = np.mean([shared_widths[r]   for r in robot_ids], axis=0)

# Shared X-axis: number of total fleet manipulation events observed
# Isolated: each robot observes 1 event per attempt
# Shared: each robot observes 1 direct + N_ROBOTS-1 indirect = N_ROBOTS events per attempt

isolated_x = list(range(1, N_ATTEMPTS_EACH + 1))          # 1 to 10
shared_x   = [i * N_ROBOTS for i in range(1, N_ATTEMPTS_EACH + 1)]  # N to 10N
ideal_x    = list(range(1, N_ROBOTS * N_ATTEMPTS_EACH + 1))          # 1 to 30

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
out_dir = Path(__file__).resolve().parent.parent / "results" / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_facecolor("#0f0f1a")
fig.patch.set_facecolor("#0f0f1a")

ax.plot(isolated_x, isolated_mean,
        color="#ff6b6b", linewidth=2.5, marker="o", markersize=5,
        label=f"Isolated (1 robot, {N_ATTEMPTS_EACH} attempts)")

ax.plot(isolated_x, shared_mean,
        color="#51cf66", linewidth=2.5, marker="s", markersize=5,
        label=f"Shared fleet ({N_ROBOTS} robots, cooperative)")

ax.plot([i / N_ROBOTS for i in ideal_x], ideal_widths,
        color="#74c0fc", linewidth=1.5, linestyle="--", alpha=0.7,
        label=f"Ideal single agent ({N_ROBOTS * N_ATTEMPTS_EACH} attempts)")

ax.axhline(y=0.0, color="white", linewidth=0.5, alpha=0.3)
ax.set_xlabel("Manipulation attempts (per robot)", color="white", fontsize=12)
ax.set_ylabel("SR Interval Width (p_hi − p_lo)", color="white", fontsize=12)
ax.set_title("Cooperative Belief Sharing: SR Interval Convergence\n"
             f"True SR = {TRUE_SR}, Fleet size = {N_ROBOTS}, Prior = Beta(9,1)",
             color="white", fontsize=13, fontweight="bold")
ax.tick_params(colors="white")
ax.spines[:].set_color("#333355")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
ax.grid(alpha=0.15, color="white")

plt.tight_layout()
plot_path = out_dir / "sr_convergence.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[SR Convergence] Plot saved → {plot_path}")

# -----------------------------------------------------------------------
# Print and save table
# -----------------------------------------------------------------------
table_lines = [
    "# SR Interval Convergence Experiment",
    "",
    f"- True SR: {TRUE_SR}",
    f"- Fleet size: {N_ROBOTS} robots",
    f"- Attempts per robot: {N_ATTEMPTS_EACH}",
    f"- Prior: Beta(9, 1) → mean SR = 0.900",
    "",
    "| Attempts/Robot | Isolated Width | Shared Width | Reduction Factor |",
    "|:---:|:---:|:---:|:---:|",
]

for i in range(N_ATTEMPTS_EACH):
    iso_w   = isolated_mean[i]
    shr_w   = shared_mean[i]
    factor  = iso_w / shr_w if shr_w > 0 else float("inf")
    table_lines.append(
        f"| {i+1:2d} | {iso_w:.4f} | {shr_w:.4f} | {factor:.2f}× |"
    )

table_lines += [
    "",
    f"**Final isolated width (after {N_ATTEMPTS_EACH} attempts): {isolated_mean[-1]:.4f}**",
    f"**Final shared width   (after {N_ATTEMPTS_EACH} attempts): {shared_mean[-1]:.4f}**",
    f"**Convergence speedup: {isolated_mean[-1]/shared_mean[-1]:.2f}×**",
    "",
    f"![SR Convergence Plot](sr_convergence.png)",
]

md = "\n".join(table_lines)
md_path = out_dir / "sr_convergence.md"
md_path.write_text(md)
print(f"[SR Convergence] Table saved → {md_path}")
print()
print(md)
