"""
eval_v3.py — Post-training evaluation for the v3 MAPPO model.
Runs the trained model against all key scenarios and prints a comparison
table: v3 (new) vs the old curriculum model results.
"""
import sys, math, json, argparse
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mappo.mappo_agent import MAPPOAgent
from mappo.mappo_env import NAMOmappoEnv

# ── Scenarios to evaluate ─────────────────────────────────────────────────────
EVAL_SCENARIOS = [
    ("namo_push_only",               "configs/namo_push_only.yaml"),
    ("movable_obstacle_choke_namo",  "configs/movable_obstacle_choke_namo.yaml"),
    ("warehouse_small",              "configs/warehouse_small.yaml"),
    ("warehouse_3robots",            "configs/warehouse_3robots.yaml"),
    ("single_corridor_yielding",     "configs/single_corridor_yielding.yaml"),
    ("symmetric_bottleneck_deadlock","configs/symmetric_bottleneck_deadlock.yaml"),
    ("cross_intersection",           "configs/cross_intersection_coordination.yaml"),
    ("warehouse_large",              "configs/warehouse_large.yaml"),
]

# Old curriculum model results (from scenario_results_table.md / training_output.log)
OLD_RESULTS = {
    "single_corridor_yielding":      {"sr": 0.0,   "push_pct": 0.0},
    "movable_obstacle_choke_namo":   {"sr": 0.0,   "push_pct": 0.0},
    "warehouse_small":               {"sr": 100.0, "push_pct": 0.0},
    "warehouse_3robots":             {"sr": 100.0, "push_pct": 0.0},
    "symmetric_bottleneck_deadlock": {"sr": 0.0,   "push_pct": 0.0},
    "cross_intersection":            {"sr": 0.0,   "push_pct": 0.0},
    "warehouse_large":               {"sr": 99.6,  "push_pct": 0.0},
    "namo_push_only":                {"sr": 0.0,   "push_pct": 0.0},
}

TRIALS = 10


def run_episode(env, agent, obs_dim, max_agents, action_dim, max_steps):
    """Run one episode, return (success, steps, push_actions, total_reward)."""
    obs, info = env.reset()
    done = False
    steps = 0
    push_count = 0
    total_reward = 0.0

    while not done and steps < max_steps:
        action_mask = info.get("action_mask", None)
        with torch.no_grad():
            actions, _, _ = agent.select_action(obs, action_masks=action_mask)

        # Count PUSH actions
        push_count += sum(1 for a in actions.values() if a == 1)

        obs, rewards, dones, info = env.step(actions)
        total_reward += np.mean(list(rewards.values()))
        steps += 1
        done = dones.get("__all__", False)

    success = info.get("success", False)
    return success, steps, push_count, total_reward


def evaluate(checkpoint_path: str, trials: int = TRIALS):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[eval] device={device}  checkpoint={checkpoint_path}")
    print(f"[eval] Running {trials} trials per scenario\n")

    results = {}

    for name, cfg_path in EVAL_SCENARIOS:
        if not Path(cfg_path).exists():
            print(f"  [SKIP] {name} — config not found")
            continue

        successes, step_list, push_list, reward_list = [], [], [], []
        agent = None

        for t in range(trials):
            try:
                env = NAMOmappoEnv(config_path=cfg_path, gui=False, max_steps=300)
                if agent is None:
                    obs, _ = env.reset()
                    sample = next(iter(obs.values()))
                    obs_dim = sample.shape[0]
                    action_dim = env.action_dim
                    max_agents = 4
                    agent = MAPPOAgent(
                        obs_dim=obs_dim, num_agents=max_agents,
                        action_dim=action_dim, device=device
                    )
                    agent.load(checkpoint_path)

                success, steps, pushes, reward = run_episode(
                    env, agent, obs_dim, max_agents, action_dim, max_steps=300
                )
                successes.append(int(success))
                step_list.append(steps)
                push_list.append(pushes)
                reward_list.append(reward)
                env.close()
            except Exception as e:
                print(f"  [ERR] {name} trial {t+1}: {e}")
                if 'env' in dir():
                    env.close()

        if not successes:
            continue

        sr      = 100.0 * sum(successes) / len(successes)
        avg_st  = sum(step_list) / len(step_list)
        total_steps = sum(step_list)
        push_pct = 100.0 * sum(push_list) / max(total_steps * max_agents, 1)
        avg_rew = sum(reward_list) / len(reward_list)

        results[name] = {
            "sr": sr, "avg_steps": avg_st,
            "push_pct": push_pct, "avg_reward": avg_rew,
        }

        old = OLD_RESULTS.get(name, {})
        old_sr = old.get("sr", "?")
        sr_delta = f"+{sr - old_sr:.0f}%" if isinstance(old_sr, float) else "N/A"

        status = "✅" if success else "❌"
        print(f"  {status} {name:<38} SR={sr:5.1f}% ({sr_delta:>6})  "
              f"Steps={avg_st:5.1f}  PUSH={push_pct:4.1f}%  R={avg_rew:.2f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print(f"  {'Scenario':<38} {'Old SR':>6} {'New SR':>6} {'Δ SR':>7} {'PUSH%':>6} {'Steps':>6}")
    print("="*90)
    for name, r in results.items():
        old = OLD_RESULTS.get(name, {})
        old_sr = old.get("sr", 0.0)
        delta = r["sr"] - old_sr
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<38} {old_sr:>5.1f}% {r['sr']:>5.1f}%  {sign}{delta:>4.1f}%  "
              f"{r['push_pct']:>5.1f}%  {r['avg_steps']:>5.1f}")
    print("="*90)

    # Save results
    out_path = Path("results/evaluation_tables/v3_eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Saved to {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v3_maxres/mappo_final.pth",
                    help="Path to trained model checkpoint")
    ap.add_argument("--trials", type=int, default=TRIALS,
                    help="Evaluation trials per scenario")
    args = ap.parse_args()
    evaluate(args.checkpoint, args.trials)
