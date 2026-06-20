"""
train_v4.py — MAPPO v4 fast training.

Key improvements over v3
-------------------------
1. GAE computed fully on CPU (NumPy), eliminating ~80 000 GPU-CPU round-trips.
2. WAIT_PENALTY = -0.02  in mappo_env.py prevents indefinite waiting.
3. Entropy coefficient c_ent = 0.05  (up from 0.02) prevents early collapse.
4. Control interval = 15  (down from 30) — faster physics feedback.
5. Curriculum bridge Stage 2b (2-robot deadlocks) before 4-robot Stage 3.
6. Resume from v3 final checkpoint — skip the already-solved Stage 1 & 2.

Usage
-----
python mappo/train_v4.py \\
    --load-checkpoint checkpoints/v3_maxres/mappo_final.pth \\
    --checkpoint-dir  checkpoints/v4 \\
    --control-interval 15 \\
    --epochs 37

The --epochs 37 flag runs epochs 19-55 (37 new epochs on top of the v3 base).
"""

import os, sys, json, functools, time, argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mappo.mappo_env import NAMOmappoEnv
from mappo.mappo_agent import MAPPOAgent


# ── Collect one episode of experience ───────────────────────────────────────
def collect_rollout(env, agent, max_steps, obs_dim, max_agents, action_dim):
    obs, info = env.reset()
    ep_reward = 0.0
    ep_steps  = 0
    ep_actions = {a: 0 for a in range(action_dim)}
    done = False

    obs_list, joint_obs_list = [], []
    action_list, logprob_list, reward_list, done_list = [], [], [], []

    while not done and ep_steps < max_steps:
        action_mask = info.get("action_mask", None)
        with torch.no_grad():
            actions, log_probs, _ = agent.select_action(obs, action_masks=action_mask)

        next_obs, rewards, dones, info = env.step(actions)

        obs_pad = [obs.get(rid, np.zeros(obs_dim)) for rid in env.robot_ids]
        while len(obs_pad) < max_agents:
            obs_pad.append(np.zeros(obs_dim))

        obs_list.append(obs_pad)
        joint_obs_list.append(np.concatenate(obs_pad))
        action_list.append(
            [actions.get(rid, 3) for rid in env.robot_ids]
            + [3] * (max_agents - len(env.robot_ids))
        )
        logprob_list.append(
            [log_probs.get(rid, 0.0) for rid in env.robot_ids]
            + [0.0] * (max_agents - len(env.robot_ids))
        )
        reward_list.append(
            [rewards.get(rid, 0.0) for rid in env.robot_ids]
            + [0.0] * (max_agents - len(env.robot_ids))
        )
        done_list.append(
            [dones.get(rid, True) for rid in env.robot_ids]
            + [True] * (max_agents - len(env.robot_ids))
        )

        for rid, a in actions.items():
            ep_actions[int(a)] += 1
        ep_reward += np.mean([rewards.get(rid, 0.0) for rid in env.robot_ids])
        ep_steps += 1
        done = dones.get("__all__", False)
        obs  = next_obs

    return (
        obs_list, joint_obs_list, action_list, logprob_list,
        reward_list, done_list,
        ep_reward, ep_steps, info.get("success", False), ep_actions,
    )


# ── v4 curriculum (55 total epochs; resume from epoch 19) ───────────────────
#
# Stage 2b (19-26): Bridge — 2-robot deadlock scenarios before jumping to 4-robot.
#   Prevents the sudden WAIT% spike that killed v3 at epoch 21.
# Stage 3  (27-40): Hard coordination — 3-4 robot warehouse + deadlock/cross.
# Stage 4  (41-55): Congestion specialization — doorway, 4-robot bottleneck.
#
# Epoch offsets below are absolute (matching the v3 log epoch numbers).
CURRICULUM_V4 = [
    # (abs_start, abs_end,  config_files)
    (19, 26, [
        "single_corridor_yielding.yaml",
        "symmetric_bottleneck_deadlock.yaml",
    ]),
    (27, 40, [
        "warehouse_large.yaml",
        "symmetric_bottleneck_deadlock.yaml",
        "cross_intersection_coordination.yaml",
    ]),
    (41, 55, [
        "narrow_doorway_congestion.yaml",
        "symmetric_bottleneck_4robots.yaml",
        "cross_intersection_coordination.yaml",
    ]),
]


def get_configs_for_epoch(epoch):
    for start, end, configs in CURRICULUM_V4:
        if start <= epoch <= end:
            return configs
    return CURRICULUM_V4[-1][2]


# ── Main training loop ───────────────────────────────────────────────────────
def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train_v4] device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = defaultdict(list)

    # Collapse detection — more patient than v3 (patience=8 vs 3)
    consecutive_collapse = 0
    COLLAPSE_THRESHOLD   = 0.85
    COLLAPSE_PATIENCE    = 8   # v4: wait longer before giving up

    # ── Initialise: detect obs_dim from checkpoint ─────────────────────────
    include_congestion = True  # default: use all 49 dims
    if args.load_checkpoint:
        try:
            _chk = torch.load(args.load_checkpoint, map_location="cpu")
            _chk_dim = _chk["actor_state"]["net.0.weight"].shape[1]
            include_congestion = (_chk_dim == 49)   # 46 = v3 without congestion feats
            print(f"[train_v4] checkpoint obs_dim={_chk_dim}  "
                  f"include_congestion_feats={include_congestion}")
        except Exception as _e:
            print(f"[train_v4] WARNING: could not detect obs_dim: {_e}")

    agent = None
    env   = None

    # Absolute epoch numbers start at START_EPOCH (19) to continue the v3 log
    START_EPOCH = 19
    TOTAL_EPOCHS = args.epochs   # number of NEW epochs to run

    for local_ep in range(TOTAL_EPOCHS):
        epoch = START_EPOCH + local_ep       # absolute epoch number for logging
        configs_this_epoch = get_configs_for_epoch(epoch)
        epoch_stats  = defaultdict(float)
        epoch_actions = defaultdict(int)
        n_episodes   = 0
        t0           = time.time()

        print(
            f"\n[Epoch {epoch:>3}/{START_EPOCH + TOTAL_EPOCHS - 1}] "
            f"configs: {[c.replace('.yaml','') for c in configs_this_epoch]}"
        )

        obs_history, joint_obs_history, action_history = [], [], []
        reward_history, log_prob_history, done_history = [], [], []

        for cfg_name in configs_this_epoch:
            cfg_path = Path(args.config_dir) / cfg_name
            if not cfg_path.exists():
                print(f"  [SKIP] Config not found: {cfg_path}")
                continue

            # First config initialises the agent (once)
            if agent is None:
                dummy_env = NAMOmappoEnv(
                    config_path=str(cfg_path), gui=False, max_steps=10,
                    control_interval=args.control_interval,
                    include_congestion_feats=include_congestion,
                )
                d_obs, _ = dummy_env.reset()
                sample_obs = next(iter(d_obs.values()))
                obs_dim    = sample_obs.shape[0] if hasattr(sample_obs, "shape") else len(sample_obs)
                action_dim = dummy_env.action_dim
                dummy_env.close()
                max_agents = 4

                agent = MAPPOAgent(
                    obs_dim=obs_dim,
                    num_agents=max_agents,
                    action_dim=action_dim,
                    device=device,
                    lr=args.lr,
                    c_ent=args.c_ent,    # 0.05 — higher entropy keeps exploration alive
                )
                if args.load_checkpoint:
                    agent.load(args.load_checkpoint)
                    print(f"  loaded checkpoint: {args.load_checkpoint}  obs_dim={obs_dim}")

            if env is not None:
                env.close()
            env = NAMOmappoEnv(
                config_path=str(cfg_path),
                gui=False,
                max_steps=args.max_steps,
                control_interval=args.control_interval,
                randomize_starts=True,
                include_congestion_feats=include_congestion,
            )

            for ep in range(args.episodes_per_epoch):
                o, jo, a, lp, r, d, ep_r, ep_s, success, ep_acts = collect_rollout(
                    env, agent, args.max_steps, obs_dim, max_agents, action_dim
                )
                obs_history.extend(o)
                joint_obs_history.extend(jo)
                action_history.extend(a)
                log_prob_history.extend(lp)
                reward_history.extend(r)
                done_history.extend(d)

                epoch_stats["reward"]  += ep_r
                epoch_stats["steps"]   += ep_s
                epoch_stats["success"] += float(success)
                for aid, cnt in ep_acts.items():
                    epoch_actions[aid] += cnt
                n_episodes += 1

                if (ep + 1) % 10 == 0:
                    print(
                        f"    Ep {ep+1}/{args.episodes_per_epoch} | "
                        f"Steps: {ep_s} | Reward: {ep_r:.2f} | Success: {success}"
                    )

        if env is not None:
            env.close()
            env = None

        elapsed     = time.time() - t0
        avg_reward  = epoch_stats["reward"]  / max(1, n_episodes)
        avg_steps   = epoch_stats["steps"]   / max(1, n_episodes)
        success_rate = epoch_stats["success"] / max(1, n_episodes)
        total_acts  = sum(epoch_actions.values()) or 1

        action_names = {0: "NAV", 1: "PUSH", 2: "YIELD", 3: "WAIT"}
        action_pct   = {action_names.get(k, str(k)): v / total_acts for k, v in epoch_actions.items()}
        wait_pct     = action_pct.get("WAIT", 0.0)

        print(
            f"  avg_reward={avg_reward:.3f}  avg_steps={avg_steps:.1f}  "
            f"success={success_rate:.1%}  time={elapsed:.1f}s"
        )
        print("  action%: " + "  ".join(f"{k}={v:.1%}" for k, v in sorted(action_pct.items())))

        if wait_pct > COLLAPSE_THRESHOLD:
            consecutive_collapse += 1
            print(
                f"  WARNING: WAIT%={wait_pct:.1%} collapse "
                f"({consecutive_collapse}/{COLLAPSE_PATIENCE})"
            )
            if consecutive_collapse >= COLLAPSE_PATIENCE:
                print("  ERROR: Policy collapse persists. Halting training.")
                break
        else:
            consecutive_collapse = 0

        log["epoch"].append(epoch)
        log["avg_reward"].append(avg_reward)
        log["avg_steps"].append(avg_steps)
        log["success_rate"].append(success_rate)
        log["wait_pct"].append(wait_pct)

        if not obs_history:
            print("  [WARN] No experience collected this epoch — skipping PPO update.")
            continue

        print(f"  PPO update with {len(obs_history)} steps...")
        actor_loss, critic_loss = agent.train_step(
            obs_history, joint_obs_history, action_history,
            reward_history, log_prob_history, done_history,
        )
        print(f"  actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}")

        if epoch % 2 == 0:
            ckpt_path = ckpt_dir / f"mappo_epoch_{epoch:03d}.pth"
            agent.save(str(ckpt_path))
            print(f"  saved: {ckpt_path}")

    # ── Finalise ─────────────────────────────────────────────────────────────
    final_path = ckpt_dir / "mappo_final.pth"
    if agent is not None:
        agent.save(str(final_path))
        print(f"\n[train_v4] final model saved: {final_path}")

    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(dict(log), f, indent=2)
    print(f"[train_v4] log saved: {log_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MAPPO v4 — fast curriculum training")
    p.add_argument("--config-dir",       default="configs",
                   help="Directory containing scenario YAML configs")
    p.add_argument("--checkpoint-dir",   default="checkpoints/v4",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--load-checkpoint",  default=None,
                   help="Path to a checkpoint to resume from (e.g. v3 final)")
    p.add_argument("--epochs",           type=int, default=37,
                   help="Number of NEW epochs to run (default=37 → abs epochs 19-55)")
    p.add_argument("--episodes-per-epoch", type=int, default=40,
                   dest="episodes_per_epoch",
                   help="Episodes per scenario per epoch (reduced for speed)")
    p.add_argument("--max-steps",        type=int, default=200,
                   dest="max_steps",
                   help="Max control steps per episode (reduced from 250 for speed)")
    p.add_argument("--control-interval", type=int, default=15,
                   dest="control_interval",
                   help="Physics steps per control action (15 = 2× faster than v3)")
    p.add_argument("--lr",               type=float, default=2e-4,
                   help="Learning rate (slightly lower than v3 for fine-tuning stability)")
    p.add_argument("--c-ent",            type=float, default=0.05,
                   dest="c_ent",
                   help="Entropy coefficient (0.05 prevents WAIT collapse)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
