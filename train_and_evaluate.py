"""
train_and_evaluate.py
---------------------
Full MAPPO training pipeline with:
  - Per-epoch CSV logging (reward, success rate, collisions, losses)
  - Auto-generated convergence plots after training
  - Best-model checkpoint saving
  - Multi-scenario evaluation after training

Usage:
  python train_and_evaluate.py --config configs/single_corridor_yielding.yaml --epochs 200
  python train_and_evaluate.py --config configs/symmetric_bottleneck_deadlock.yaml --epochs 200
  python train_and_evaluate.py --all-scenarios --epochs 200
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent))

from mappo.mappo_env import NAMOmappoEnv
from mappo.mappo_agent import MAPPOAgent

# ─────────────────────────────────────────────────────────────────────────────
# Scenarios to train and evaluate
# ─────────────────────────────────────────────────────────────────────────────
ALL_SCENARIOS = [
    "configs/single_corridor_yielding.yaml",
    "configs/symmetric_bottleneck_deadlock.yaml",
    "configs/movable_obstacle_choke_namo.yaml",
    "configs/narrow_doorway_congestion.yaml",
]

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(config_path, epochs=200, episodes_per_epoch=5, max_steps=60, lr=3e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    scenario_name = Path(config_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directories
    run_dir = Path("results/training_runs") / f"{scenario_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("models/checkpoints") / scenario_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  MAPPO Training — {scenario_name}")
    print(f"  Epochs: {epochs} | Episodes/epoch: {episodes_per_epoch} | Seed: {seed}")
    print(f"  Output: {run_dir}")
    print(f"{'='*65}\n")

    # Init environment and agent
    env = NAMOmappoEnv(config_path, gui=False, max_steps=max_steps)
    obs = env.reset()
    obs_dim = next(iter(obs.values())).shape[0]
    num_agents = len(obs)
    print(f"  Agents: {num_agents} | Obs dim: {obs_dim}\n")

    agent = MAPPOAgent(obs_dim, num_agents, lr=lr)

    # CSV log
    csv_path = run_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_reward", "success_rate", "collisions", "actor_loss", "critic_loss"])

    best_success_rate = -1.0
    history = {"reward": [], "success": [], "collisions": [], "actor_loss": [], "critic_loss": []}

    for epoch in range(1, epochs + 1):
        obs_history, joint_obs_history = [], []
        action_history, reward_history = [], []
        log_prob_history, done_history = [], []

        epoch_rewards, epoch_collisions, success_count = [], 0, 0

        for _ in range(episodes_per_epoch):
            obs = env.reset()
            ep_reward = 0

            for _ in range(max_steps):
                obs_list   = [obs[rid] for rid in env.robot_ids]
                joint_obs  = np.concatenate(obs_list)

                actions, log_probs, _ = agent.select_action(obs)
                next_obs, rewards, dones, info = env.step(actions)

                obs_history.append(obs_list)
                joint_obs_history.append(joint_obs)
                action_history.append([actions[rid]   for rid in env.robot_ids])
                reward_history.append([rewards[rid]   for rid in env.robot_ids])
                log_prob_history.append([log_probs[rid] for rid in env.robot_ids])
                done_history.append([dones[rid]       for rid in env.robot_ids])

                ep_reward        += sum(rewards.values())
                epoch_collisions += info["collisions"]
                obs               = next_obs

                if dones["__all__"]:
                    if info["success"]:
                        success_count += 1
                    break

            epoch_rewards.append(ep_reward)

        # Policy update
        actor_loss, critic_loss = agent.train_step(
            obs_history, joint_obs_history, action_history,
            reward_history, log_prob_history, done_history
        )

        avg_reward   = float(np.mean(epoch_rewards))
        success_rate = (success_count / episodes_per_epoch) * 100.0

        history["reward"].append(avg_reward)
        history["success"].append(success_rate)
        history["collisions"].append(epoch_collisions)
        history["actor_loss"].append(actor_loss)
        history["critic_loss"].append(critic_loss)

        # CSV row
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_reward:.3f}", f"{success_rate:.1f}",
                                    epoch_collisions, f"{actor_loss:.5f}", f"{critic_loss:.5f}"])

        # Save best checkpoint
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            torch.save(agent.actor.state_dict(),  str(checkpoint_dir / "best_actor.pth"))
            torch.save(agent.critic.state_dict(), str(checkpoint_dir / "best_critic.pth"))

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Collisions: {epoch_collisions:4d} | "
                  f"Loss A/C: {actor_loss:.4f}/{critic_loss:.4f}")

    env.close()

    # Save final checkpoint
    torch.save(agent.actor.state_dict(),  str(checkpoint_dir / "final_actor.pth"))
    torch.save(agent.critic.state_dict(), str(checkpoint_dir / "final_critic.pth"))
    # Also overwrite the main model files
    torch.save(agent.actor.state_dict(),  "models/mappo_actor_checkpoint.pth")
    torch.save(agent.critic.state_dict(), "models/mappo_critic_checkpoint.pth")

    print(f"\n  Best success rate: {best_success_rate:.1f}%")
    print(f"  Checkpoints saved to: {checkpoint_dir}")

    # Generate plots
    _plot_training_curves(history, run_dir, scenario_name, epochs)

    return history, best_success_rate, agent, obs_dim, num_agents


# ─────────────────────────────────────────────────────────────────────────────
# Plot generator
# ─────────────────────────────────────────────────────────────────────────────
def _plot_training_curves(history, run_dir, scenario_name, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#0f0f1a")

    def _style_ax(ax, title, ylabel):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", color="#aaaacc", fontsize=9)
        ax.set_ylabel(ylabel, color="#aaaacc", fontsize=9)
        ax.tick_params(colors="#aaaacc")
        for spine in ax.spines.values():
            spine.set_color("#333355")
        ax.grid(alpha=0.15, color="white")

    xs = list(range(1, epochs + 1))

    # Reward
    axes[0, 0].plot(xs, history["reward"], color="#51cf66", linewidth=2)
    axes[0, 0].fill_between(xs, history["reward"], alpha=0.15, color="#51cf66")
    _style_ax(axes[0, 0], "Average Episode Reward", "Reward")

    # Success rate
    axes[0, 1].plot(xs, history["success"], color="#74c0fc", linewidth=2)
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].axhline(y=100, color="#ff6b6b", linewidth=0.8, linestyle="--", alpha=0.5)
    _style_ax(axes[0, 1], "Success Rate (%)", "Success Rate (%)")

    # Collisions
    axes[1, 0].plot(xs, history["collisions"], color="#ff6b6b", linewidth=2)
    _style_ax(axes[1, 0], "Collisions per Epoch", "Collision Count")

    # Losses
    axes[1, 1].plot(xs, history["actor_loss"],  color="#ffd43b", linewidth=1.8, label="Actor")
    axes[1, 1].plot(xs, history["critic_loss"], color="#cc5de8", linewidth=1.8, label="Critic")
    axes[1, 1].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    _style_ax(axes[1, 1], "Training Losses", "Loss")

    fig.suptitle(f"MAPPO Training — {scenario_name}", color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    plot_path = run_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Training curves saved → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Post-training evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(config_path, agent, n_episodes=10, max_steps=60):
    env = NAMOmappoEnv(config_path, gui=False, max_steps=max_steps)
    success_count, total_steps, total_collisions, total_pushes = 0, 0, 0, 0

    for _ in range(n_episodes):
        obs = env.reset()
        for t in range(max_steps):
            actions, _, _ = agent.select_action(obs)
            obs, _, dones, info = env.step(actions)
            total_steps      += 1
            total_collisions += info["collisions"]
            if dones["__all__"]:
                if info["success"]:
                    success_count += 1
                break

    env.close()
    return {
        "success_rate":    (success_count / n_episodes) * 100.0,
        "avg_steps":        total_steps / n_episodes,
        "avg_collisions":   total_collisions / n_episodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MAPPO Training + Evaluation Pipeline")
    parser.add_argument("--config",         default="configs/single_corridor_yielding.yaml")
    parser.add_argument("--all-scenarios",  action="store_true",  help="Train on all scenarios")
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--episodes",       type=int,   default=5,    help="Episodes per epoch")
    parser.add_argument("--max-steps",      type=int,   default=60)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--eval-episodes",  type=int,   default=10)
    args = parser.parse_args()

    configs = ALL_SCENARIOS if args.all_scenarios else [args.config]

    summary_rows = []

    for cfg in configs:
        if not Path(cfg).exists():
            print(f"  [SKIP] Config not found: {cfg}")
            continue

        history, best_sr, agent, obs_dim, num_agents = train(
            cfg,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes,
            max_steps=args.max_steps,
            lr=args.lr,
            seed=args.seed,
        )

        print(f"\n  Evaluating {Path(cfg).stem} for {args.eval_episodes} episodes...")
        metrics = evaluate(cfg, agent, n_episodes=args.eval_episodes, max_steps=args.max_steps)
        print(f"  → Success: {metrics['success_rate']:.1f}% | "
              f"Avg Steps: {metrics['avg_steps']:.1f} | "
              f"Avg Collisions: {metrics['avg_collisions']:.1f}")

        summary_rows.append({
            "scenario":       Path(cfg).stem,
            "best_train_sr":  f"{best_sr:.1f}%",
            "eval_sr":        f"{metrics['success_rate']:.1f}%",
            "avg_steps":      f"{metrics['avg_steps']:.1f}",
            "avg_collisions": f"{metrics['avg_collisions']:.1f}",
        })

    # Save summary table
    if summary_rows:
        out_path = Path("results/training_runs/summary_table.md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# MAPPO Training & Evaluation Summary\n",
            "| Scenario | Best Train SR | Eval SR | Avg Steps | Avg Collisions |",
            "|:---|:---:|:---:|:---:|:---:|",
        ]
        for r in summary_rows:
            lines.append(f"| {r['scenario']} | {r['best_train_sr']} | {r['eval_sr']} | {r['avg_steps']} | {r['avg_collisions']} |")
        out_path.write_text("\n".join(lines))
        print(f"\n  Summary table → {out_path}")

    print(f"\n{'='*65}")
    print("  Training complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
