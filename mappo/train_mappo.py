"""
train_mappo.py — Fast sequential MAPPO training with curriculum.
Fixes: obs indexing (env.robot_ids), success metric (info["success"]), WAIT persistence.
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


def collect_rollout(env, agent, max_steps, obs_dim, max_agents, action_dim):
    obs, info = env.reset()
    ep_reward = 0.0
    ep_steps = 0
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
        action_list.append([actions.get(rid, 3) for rid in env.robot_ids] + [3] * (max_agents - len(env.robot_ids)))
        logprob_list.append([log_probs.get(rid, 0.0) for rid in env.robot_ids] + [0.0] * (max_agents - len(env.robot_ids)))
        reward_list.append([rewards.get(rid, 0.0) for rid in env.robot_ids] + [0.0] * (max_agents - len(env.robot_ids)))
        done_list.append([dones.get(rid, True) for rid in env.robot_ids] + [True] * (max_agents - len(env.robot_ids)))

        for rid, a in actions.items():
            ep_actions[int(a)] += 1
        ep_reward += np.mean([rewards.get(rid, 0.0) for rid in env.robot_ids])
        ep_steps += 1
        done = dones.get("__all__", False)
        obs = next_obs

    return obs_list, joint_obs_list, action_list, logprob_list, reward_list, done_list, ep_reward, ep_steps, info.get("success", False), ep_actions


# ── 25-epoch validation curriculum ──────────────────────────────────────────
# Stage 1 (1–8):   Force PUSH learning — scenarios where push is mandatory
# Stage 2 (9–18):  Scale to multi-robot warehouse + yielding
# Stage 3 (19–25): Hard coordination & deadlock stress test
# If results look good after 25 epochs → scale to 50+ epochs
CURRICULUM = [
    (1,  8,  ["namo_push_only.yaml", "movable_obstacle_choke_namo.yaml"]),
    (9,  18, ["warehouse_small.yaml", "warehouse_3robots.yaml", "single_corridor_yielding.yaml"]),
    (19, 25, ["warehouse_large.yaml", "symmetric_bottleneck_deadlock.yaml", "cross_intersection_coordination.yaml"]),
]

def get_configs_for_epoch(epoch):
    for start, end, configs in CURRICULUM:
        if start <= epoch <= end:
            return configs
    return CURRICULUM[-1][2]


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train] device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = defaultdict(list)
    consecutive_collapse_epochs = 0
    COLLAPSE_THRESHOLD = 0.85
    COLLAPSE_PATIENCE = 5

    agent = None
    env = None

    for epoch in range(1, args.epochs + 1):
        configs_this_epoch = get_configs_for_epoch(epoch)
        epoch_stats = defaultdict(float)
        epoch_actions = defaultdict(int)
        n_episodes = 0
        t0 = time.time()

        print(f"\n[Epoch {epoch:>3}/{args.epochs}] configs: {[c.replace('.yaml','') for c in configs_this_epoch]}")

        obs_history, joint_obs_history, action_history = [], [], []
        reward_history, log_prob_history, done_history = [], [], []

        for cfg_name in configs_this_epoch:
            cfg_path = Path(args.config_dir) / cfg_name

            if agent is None:
                dummy_env = NAMOmappoEnv(config_path=str(cfg_path), gui=False, max_steps=10)
                d_obs, _ = dummy_env.reset()
                sample_obs = next(iter(d_obs.values()))
                obs_dim = sample_obs.shape[0] if hasattr(sample_obs, 'shape') else len(sample_obs)
                action_dim = dummy_env.action_dim
                dummy_env.close()
                max_agents = 4
                agent = MAPPOAgent(obs_dim=obs_dim, num_agents=max_agents, action_dim=action_dim, device=device, lr=3e-4)
                if args.load_checkpoint:
                    agent.load(args.load_checkpoint)
                    print(f"  loaded: {args.load_checkpoint}")

            if env is not None:
                env.close()
            env = NAMOmappoEnv(config_path=str(cfg_path), gui=False, max_steps=args.max_steps)

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

                epoch_stats["reward"] += ep_r
                epoch_stats["steps"] += ep_s
                epoch_stats["success"] += float(success)
                for aid, cnt in ep_acts.items():
                    epoch_actions[aid] += cnt
                n_episodes += 1

                if (ep + 1) % 10 == 0:
                    print(f"    Ep {ep+1}/{args.episodes_per_epoch} | Steps: {ep_s} | Reward: {ep_r:.2f} | Success: {success}")

        if env is not None:
            env.close()
            env = None

        elapsed = time.time() - t0
        avg_reward = epoch_stats["reward"] / max(1, n_episodes)
        avg_steps = epoch_stats["steps"] / max(1, n_episodes)
        success_rate = epoch_stats["success"] / max(1, n_episodes)
        total_acts = sum(epoch_actions.values()) or 1

        action_names = {0: "NAV", 1: "PUSH", 2: "YIELD", 3: "WAIT"}
        action_pct = {action_names.get(k, str(k)): v / total_acts for k, v in epoch_actions.items()}
        wait_pct = action_pct.get("WAIT", 0.0)

        print(f"  avg_reward={avg_reward:.3f}  avg_steps={avg_steps:.1f}  success={success_rate:.1%}  time={elapsed:.1f}s")
        print(f"  action%: " + "  ".join(f"{k}={v:.1%}" for k, v in sorted(action_pct.items())))

        if wait_pct > COLLAPSE_THRESHOLD:
            consecutive_collapse_epochs += 1
            print(f"  WARNING: WAIT%={wait_pct:.1%} collapse ({consecutive_collapse_epochs}/{COLLAPSE_PATIENCE})")
            if consecutive_collapse_epochs >= COLLAPSE_PATIENCE:
                print("  ERROR: Policy collapse. Halting.")
                break
        else:
            consecutive_collapse_epochs = 0

        log["epoch"].append(epoch)
        log["avg_reward"].append(avg_reward)
        log["avg_steps"].append(avg_steps)
        log["success_rate"].append(success_rate)
        log["wait_pct"].append(wait_pct)

        print(f"  PPO update with {len(obs_history)} steps...")
        actor_loss, critic_loss = agent.train_step(
            obs_history, joint_obs_history, action_history, reward_history, log_prob_history, done_history
        )
        print(f"  actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}")

        if epoch % 2 == 0:
            ckpt_path = ckpt_dir / f"mappo_epoch_{epoch:03d}.pth"
            agent.save(str(ckpt_path))
            print(f"  saved: {ckpt_path}")

    final_path = ckpt_dir / "mappo_final.pth"
    if agent is not None:
        agent.save(str(final_path))
        print(f"\n[train] final model saved: {final_path}")

    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(dict(log), f, indent=2)
    print(f"[train] log saved: {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Fast MAPPO training")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--checkpoint-dir", default="checkpoints/curriculum")
    p.add_argument("--load-checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=25,
                   help="Training epochs. Use 25 for validation, 55 for full run.")
    p.add_argument("--episodes-per-epoch", type=int, default=40, dest="episodes_per_epoch",
                   help="Episodes per scenario per epoch. 40 gives ~2k steps/update on M3.")
    p.add_argument("--max-steps", type=int, default=250, dest="max_steps",
                   help="Max env steps per episode.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
