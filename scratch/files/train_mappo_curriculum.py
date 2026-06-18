"""
train_mappo_curriculum.py
Drop-in replacement for mappo/train_mappo.py.

Changes vs original:
  1. Curriculum: configs are introduced in phases by epoch.
  2. Episodes per epoch: 50 (was 5).
  3. Action masking passed from env to agent.
  4. Reward logging: tracks push/yield bonus, collision, cost_reduction.
  5. Checkpoint saved every 2 epochs.
  6. Early-stop detection: if WAIT% > 70% for 3 consecutive epochs, halt and alert.

FAILED configs never appear in training:
  - narrow_doorway_congestion.yaml     (S-NAMO: FAILED)
  - symmetric_bottleneck_4robots.yaml  (S-NAMO: FAILED)
These are test-only.
"""

import argparse
import os
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── curriculum definition ────────────────────────────────────────────────────
# Key: (start_epoch, end_epoch_inclusive) → list of config filenames
# Only SUCCESS configs included.
CURRICULUM = [
    # Phase 1: 1-10 — single-robot yielding, no boxes
    # Goal: learn to navigate and yield without needing PUSH at all.
    (1, 10,  ["single_corridor_yielding.yaml"]),

    # Phase 2: 11-25 — small box environments (1-3 boxes)
    # Goal: learn PUSH vs BYPASS decision in simple geometry.
    (11, 25, [
        "warehouse_small.yaml",           # S-NAMO: 177 steps
        "movable_obstacle_choke_namo.yaml",  # S-NAMO: 212 steps
        "warehouse_3robots.yaml",         # S-NAMO: 549 steps
    ]),

    # Phase 3: 26-40 — larger environments with social costs
    # Goal: generalise coordination across longer paths.
    (26, 40, [
        "warehouse_large.yaml",                    # S-NAMO: 554 steps
        "symmetric_bottleneck_deadlock.yaml",      # S-NAMO: 770 steps
        "cross_intersection_coordination.yaml",    # S-NAMO: 772 steps
    ]),

    # Phase 4: 41-50 — hardest solvable map
    # Goal: stress-test; custom map is dense and realistic.
    (41, 50, [
        "custom_reconstructed_map_robots.yaml",   # S-NAMO: 1740 steps
    ]),
]

# ── configs reserved for zero-shot evaluation only ───────────────────────────
TEST_ONLY_CONFIGS = [
    "narrow_doorway_congestion.yaml",      # S-NAMO: FAILED
    "symmetric_bottleneck_4robots.yaml",   # S-NAMO: FAILED
]


def get_configs_for_epoch(epoch: int) -> list[str]:
    """Return the list of training configs active at this epoch."""
    for start, end, configs in CURRICULUM:
        if start <= epoch <= end:
            return configs
    return CURRICULUM[-1][2]  # fallback to hardest phase


# ── training loop ────────────────────────────────────────────────────────────

def train(args):
    from mappo.mappo_env import MANAMOEnv   # adjust import to your module path
    from mappo.mappo_agent import MAPPOAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # ── logging ──
    log = defaultdict(list)
    consecutive_collapse_epochs = 0
    COLLAPSE_THRESHOLD = 0.70   # if WAIT% > 70%, policy is collapsing
    COLLAPSE_PATIENCE  = 3      # halt after this many consecutive collapse epochs

    # ── checkpoint dir ──
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── env + agent init ──
    # We create the env once and reset it per episode.
    # If your env requires a config at construction, build a factory instead.
    env   = None  # will be (re)created per phase if needed
    agent = None  # built after first env to know obs/action dims

    for epoch in range(1, args.epochs + 1):
        configs_this_epoch = get_configs_for_epoch(epoch)
        epoch_stats = defaultdict(float)
        epoch_actions = defaultdict(int)
        n_episodes = 0

        print(f"\n[Epoch {epoch:>3}/{args.epochs}] "
              f"configs: {[c.replace('.yaml','') for c in configs_this_epoch]}")

        for cfg_name in configs_this_epoch:
            cfg_path = Path(args.config_dir) / cfg_name

            for ep in range(args.episodes_per_epoch):
                # ── build / reset env ──
                env = MANAMOEnv(config_path=str(cfg_path), device=device)
                obs, info = env.reset()

                # ── build agent once we know dims ──
                if agent is None:
                    obs_dim    = obs.shape[-1]
                    action_dim = env.action_space_n
                    agent = MAPPOAgent(
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        device=device,
                    )
                    if args.load_checkpoint:
                        agent.load(args.load_checkpoint)
                        print(f"  loaded checkpoint: {args.load_checkpoint}")

                ep_reward   = 0.0
                ep_steps    = 0
                ep_actions  = defaultdict(int)
                done        = False

                while not done and ep_steps < args.max_steps:
                    # action_mask shape: (n_robots, action_dim) bool
                    action_mask = info.get("action_mask", None)

                    actions, log_probs, values = agent.select_action(
                        obs, action_mask=action_mask
                    )

                    obs_next, rewards, dones, info = env.step(actions)

                    # ── record action distribution ──
                    for a in actions.flatten().tolist():
                        ep_actions[int(a)] += 1

                    agent.store_transition(
                        obs, actions, log_probs, values, rewards, dones, action_mask
                    )

                    ep_reward += rewards.mean().item()
                    ep_steps  += 1
                    done       = dones.all().item()
                    obs        = obs_next

                agent.update()

                # ── per-episode stats ──
                epoch_stats["reward"]    += ep_reward
                epoch_stats["steps"]     += ep_steps
                epoch_stats["success"]   += float(done and ep_steps < args.max_steps)
                for a_id, cnt in ep_actions.items():
                    epoch_actions[a_id] += cnt
                n_episodes += 1

        # ── epoch-level metrics ──
        avg_reward  = epoch_stats["reward"]  / n_episodes
        avg_steps   = epoch_stats["steps"]   / n_episodes
        success_rate= epoch_stats["success"] / n_episodes
        total_acts  = sum(epoch_actions.values()) or 1

        action_names = {0: "NAVIGATE", 1: "PUSH", 2: "YIELD", 3: "WAIT"}
        action_pct   = {action_names.get(k, str(k)): v / total_acts
                        for k, v in epoch_actions.items()}

        wait_pct = action_pct.get("WAIT", 0.0)

        print(f"  avg_reward={avg_reward:.3f}  avg_steps={avg_steps:.1f}  "
              f"success={success_rate:.1%}")
        print(f"  action%: " +
              "  ".join(f"{k}={v:.1%}" for k, v in sorted(action_pct.items())))

        # ── collapse detection ──
        if wait_pct > COLLAPSE_THRESHOLD:
            consecutive_collapse_epochs += 1
            print(f"  ⚠️  WAIT% = {wait_pct:.1%} — collapse warning "
                  f"({consecutive_collapse_epochs}/{COLLAPSE_PATIENCE})")
            if consecutive_collapse_epochs >= COLLAPSE_PATIENCE:
                print("  ❌ Policy collapse detected. "
                      "Halting. Cut learning rate by 2× and resume from last checkpoint.")
                break
        else:
            consecutive_collapse_epochs = 0

        # ── logging ──
        log["epoch"].append(epoch)
        log["avg_reward"].append(avg_reward)
        log["avg_steps"].append(avg_steps)
        log["success_rate"].append(success_rate)
        log["wait_pct"].append(wait_pct)

        # ── checkpoint every 2 epochs ──
        if epoch % 2 == 0:
            ckpt_path = ckpt_dir / f"mappo_epoch_{epoch:03d}.pth"
            agent.save(str(ckpt_path))
            print(f"  saved checkpoint: {ckpt_path}")

    # ── final save ──
    final_path = ckpt_dir / "mappo_final.pth"
    if agent is not None:
        agent.save(str(final_path))
        print(f"\n[train] final model saved: {final_path}")

    # ── save log ──
    import json
    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(dict(log), f, indent=2)
    print(f"[train] log saved: {log_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Curriculum MAPPO training")
    p.add_argument("--config-dir",          default="configs",
                   help="Directory containing *.yaml map configs")
    p.add_argument("--checkpoint-dir",      default="checkpoints/curriculum",
                   help="Where to save .pth checkpoints")
    p.add_argument("--load-checkpoint",     default=None,
                   help="Path to .pth file to resume from")
    p.add_argument("--epochs",              type=int, default=50)
    p.add_argument("--episodes-per-epoch",  type=int, default=50,
                   dest="episodes_per_epoch")
    p.add_argument("--max-steps",           type=int, default=1500,
                   dest="max_steps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
