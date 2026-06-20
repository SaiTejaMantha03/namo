"""
eval_v4.py — Post-training evaluation for MAPPO v4.

Runs the v4 model against all key benchmark scenarios and prints a
3-way comparison table: S-NAMO vs MAPPO v3 vs MAPPO v4.

Usage
-----
python eval_v4.py
python eval_v4.py --checkpoint checkpoints/v4/mappo_final.pth --trials 20
python eval_v4.py --checkpoint checkpoints/v4/mappo_final.pth --gui
"""

import sys, math, json, argparse
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mappo.mappo_agent import MAPPOAgent
from mappo.mappo_env   import NAMOmappoEnv

# ── Scenarios ────────────────────────────────────────────────────────────────
EVAL_SCENARIOS = [
    ("namo_push_only",                "configs/namo_push_only.yaml"),
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

# ── S-NAMO paper baseline (original paper Table II / our re-implementation) ──
SNAMO_RESULTS = {
    "namo_push_only":                {"sr": 100.0, "avg_steps": 8.0},
    "movable_obstacle_choke_namo":   {"sr": 100.0, "avg_steps": 12.0},
    "warehouse_small":               {"sr": 100.0, "avg_steps": 18.0},
    "warehouse_3robots":             {"sr": 100.0, "avg_steps": 24.0},
    "single_corridor_yielding":      {"sr": 100.0, "avg_steps": 35.0},
    "symmetric_bottleneck_deadlock": {"sr": 95.0,  "avg_steps": 42.0},
    "cross_intersection":            {"sr": 90.0,  "avg_steps": 55.0},
    "warehouse_large":               {"sr": 100.0, "avg_steps": 38.0},
    "narrow_doorway_congestion":     {"sr": 85.0,  "avg_steps": 60.0},
    "symmetric_bottleneck_4robots":  {"sr": 80.0,  "avg_steps": 72.0},
}

# ── MAPPO v3 results (from v3_eval_results.json; collapsed at epoch 21) ──────
MAPPO_V3_RESULTS = {
    "namo_push_only":                {"sr": 100.0, "avg_steps": 7.3},
    "movable_obstacle_choke_namo":   {"sr": 100.0, "avg_steps": 7.9},
    "warehouse_small":               {"sr": 100.0, "avg_steps": 8.0},
    "warehouse_3robots":             {"sr": 100.0, "avg_steps": 10.0},
    "single_corridor_yielding":      {"sr": 100.0, "avg_steps": 32.5},
    "symmetric_bottleneck_deadlock": {"sr": 100.0, "avg_steps": 34.6},
    "cross_intersection":            {"sr": 30.0,  "avg_steps": 180.0},  # collapsed
    "warehouse_large":               {"sr": 100.0, "avg_steps": 21.0},
    "narrow_doorway_congestion":     {"sr": 0.0,   "avg_steps": 200.0},  # never trained
    "symmetric_bottleneck_4robots":  {"sr": 0.0,   "avg_steps": 200.0},  # never trained
}

TRIALS = 10


# ── Episode runner ───────────────────────────────────────────────────────────
def run_episode(env, agent, obs_dim, max_agents, action_dim, max_steps, gui=False):
    """Run one episode; return (success, ctrl_steps, push_actions, total_reward)."""
    obs, info = env.reset()
    done  = False
    steps = 0
    push_count   = 0
    total_reward = 0.0

    while not done and steps < max_steps:
        action_mask = info.get("action_mask", None)
        with torch.no_grad():
            actions, _, _ = agent.select_action(obs, action_masks=action_mask)

        push_count += sum(1 for a in actions.values() if a == 1)

        obs, rewards, dones, info = env.step(actions)
        total_reward += np.mean(list(rewards.values()))
        steps += 1
        done = dones.get("__all__", False)

        if gui:
            import time
            time.sleep(0.04)

    success = info.get("success", False)
    return success, steps, push_count, total_reward


# ── Main evaluator ───────────────────────────────────────────────────────────
def evaluate(checkpoint_path: str, trials: int = TRIALS,
             gui: bool = False, control_interval: int = 15):
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\n[eval_v4] device={device}  trials={trials}")
    print(f"          checkpoint={checkpoint_path}")
    print(f"          control_interval={control_interval}\n")

    # Auto-detect obs_dim from checkpoint weights
    try:
        chk = torch.load(checkpoint_path, map_location="cpu")
        obs_dim_chk = chk["actor_state"]["net.0.weight"].shape[1]
        include_congestion = (obs_dim_chk == 49)
        print(f"[eval_v4] Detected obs_dim={obs_dim_chk}  "
              f"include_congestion_feats={include_congestion}")
    except Exception as e:
        print(f"[eval_v4] WARNING: Could not auto-detect obs_dim: {e}. "
              "Defaulting include_congestion_feats=True")
        include_congestion = True

    results_v4 = {}

    for name, cfg_path in EVAL_SCENARIOS:
        if not Path(cfg_path).exists():
            print(f"  [SKIP] {name} — config not found at {cfg_path}")
            continue

        successes, step_list, push_list, reward_list = [], [], [], []
        agent = None

        for t in range(trials):
            try:
                env = NAMOmappoEnv(
                    config_path=cfg_path, gui=gui,
                    max_steps=300,
                    control_interval=control_interval,
                    include_congestion_feats=include_congestion,
                )

                if agent is None:
                    obs, _ = env.reset()
                    sample  = next(iter(obs.values()))
                    obs_dim = sample.shape[0]
                    max_agents = 4
                    agent = MAPPOAgent(
                        obs_dim=obs_dim, num_agents=max_agents,
                        action_dim=env.action_dim, device=device,
                    )
                    agent.load(checkpoint_path)

                success, steps, pushes, reward = run_episode(
                    env, agent, obs_dim, max_agents,
                    env.action_dim, max_steps=300, gui=gui,
                )
                successes.append(int(success))
                step_list.append(steps)
                push_list.append(pushes)
                reward_list.append(reward)
                env.close()

            except Exception as e:
                print(f"  [ERR] {name} trial {t+1}: {e}")
                try:
                    env.close()
                except Exception:
                    pass

        if not successes:
            continue

        sr       = 100.0 * sum(successes) / len(successes)
        avg_st   = sum(step_list)  / len(step_list)
        push_pct = 100.0 * sum(push_list) / max(sum(step_list) * max_agents, 1)
        avg_rew  = sum(reward_list) / len(reward_list)

        results_v4[name] = {
            "sr": sr, "avg_steps": avg_st,
            "push_pct": push_pct, "avg_reward": avg_rew,
            "raw_successes": successes,
            "raw_steps": step_list,
            "raw_pushes": push_list,
        }

        snamo = SNAMO_RESULTS.get(name, {})
        v3    = MAPPO_V3_RESULTS.get(name, {})
        v3_sr = v3.get("sr", 0.0)
        delta  = f"{sr - v3_sr:+.0f}%" if isinstance(v3_sr, float) else "N/A"
        status = "✅" if sr >= 80.0 else ("⚠️" if sr >= 50.0 else "❌")

        print(
            f"  {status} {name:<38} "
            f"SNAMO={snamo.get('sr','?'):>5}%  "
            f"v3={v3_sr:>5.1f}%  "
            f"v4={sr:>5.1f}% ({delta:>5})  "
            f"Steps={avg_st:5.1f}"
        )

    # ── 3-way comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(
        f"  {'Scenario':<40} "
        f"{'S-NAMO SR':>10} {'S-NAMO St':>10} "
        f"{'v3 SR':>7} {'v3 St':>7} "
        f"{'v4 SR':>7} {'v4 St':>7} "
        f"{'Δ vs v3':>8} {'Δ vs SNAMO':>10}"
    )
    print("=" * 110)

    for name, r4 in results_v4.items():
        snamo = SNAMO_RESULTS.get(name, {})
        v3    = MAPPO_V3_RESULTS.get(name, {})
        delta_v3    = r4["sr"] - v3.get("sr", 0.0)
        delta_snamo = r4["sr"] - snamo.get("sr", 0.0)
        sign_v3    = "+" if delta_v3    >= 0 else ""
        sign_snamo = "+" if delta_snamo >= 0 else ""

        print(
            f"  {name:<40} "
            f"{snamo.get('sr','?'):>9.1f}% {snamo.get('avg_steps','?'):>9}  "
            f"{v3.get('sr',0):>6.1f}% {v3.get('avg_steps',0):>6.0f}  "
            f"{r4['sr']:>6.1f}% {r4['avg_steps']:>6.1f}  "
            f"{sign_v3}{delta_v3:>6.1f}%  "
            f"{sign_snamo}{delta_snamo:>7.1f}%"
        )

    print("=" * 110)

    # ── Persist results ───────────────────────────────────────────────────────
    out_dir = Path("results/evaluation_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save v4 results
    v4_path = out_dir / "v4_eval_results.json"
    with open(v4_path, "w") as f:
        json.dump(results_v4, f, indent=2)
    print(f"\n[eval_v4] v4 results saved → {v4_path}")

    # Save 3-way comparison
    comparison = {}
    for name in results_v4:
        comparison[name] = {
            "snamo": SNAMO_RESULTS.get(name, {}),
            "mappo_v3": MAPPO_V3_RESULTS.get(name, {}),
            "mappo_v4": results_v4[name],
        }
    cmp_path = out_dir / "snamo_vs_mappo_v3_vs_v4.json"
    with open(cmp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[eval_v4] 3-way comparison saved → {cmp_path}")

    return results_v4


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MAPPO v4 evaluation & comparison")
    ap.add_argument(
        "--checkpoint", default="checkpoints/v4/mappo_final.pth",
        help="Path to v4 trained model checkpoint",
    )
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--gui",    action="store_true")
    ap.add_argument(
        "--control-interval", type=int, default=15,
        dest="control_interval",
        help="Must match the value used during v4 training (default 15)",
    )
    args = ap.parse_args()
    evaluate(
        args.checkpoint, args.trials,
        gui=args.gui, control_interval=args.control_interval,
    )
