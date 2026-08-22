"""Dedicated Discrete SAC Training Runner for Config Original (Exact Paper Setup).

Executes 50,000 training episodes under Discrete SAC with:
- 60-dim observation (baseline without task-intent mu_a)
- 4-layer MLP [256, 256, 64, 3]
- 150,000 replay buffer capacity, batch size 1024
- gamma=0.90, tau=0.005, actor_lr=0.0003, critic_lr=0.0005
- Automatic checkpoint auto-resuming
- P-core thread optimization (torch.set_num_threads(4)) to prevent hybrid CPU stall
"""
import os
import sys

# Constrain OpenMP/MKL before torch is imported
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import time
import glob
import re
import csv
import torch
import numpy as np

# Set PyTorch thread count explicitly to 4 (P-cores)
torch.set_num_threads(4)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'common'))
sys.path.insert(0, COMMON_DIR)

from config import CONFIG
from discrete_sac import DiscreteSACAgent, ReplayBuffer
from evaluate_exact import build_env


def train():
    exp_tag = CONFIG["exp_tag"]
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    logs_dir = os.path.join(BASE_DIR, "logs")
    log_csv = os.path.join(logs_dir, "training_progress.csv")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Check for existing checkpoints to support automatic resumption
    ckpts = glob.glob(os.path.join(checkpoint_dir, "sac_model_ep_*.pt"))
    ckpt_eps = []
    for c in ckpts:
        m = re.search(r"sac_model_ep_(\d+)\.pt", c)
        if m:
            ckpt_eps.append(int(m.group(1)))

    if ckpt_eps:
        start_ep = max(ckpt_eps)
        latest_ckpt_path = os.path.join(checkpoint_dir, f"sac_model_ep_{start_ep}.pt")
    else:
        start_ep = 0
        latest_ckpt_path = None

    print(f"=== Starting / Resuming Training: {exp_tag} ===")
    print(f"Algorithm: Discrete SAC | State Dim: {CONFIG['state_dim']} (Baseline without mu_a)")
    print(f"Target Total Episodes: {CONFIG['total_episodes']} | Starting from episode: {start_ep}")
    print(f"PyTorch CPU Threads: {torch.get_num_threads()} (Optimized for Intel P-cores)")

    device = "cpu"
    if torch.cuda.is_available():
        try:
            test_lin = torch.nn.Linear(10, 10).cuda()
            test_x = torch.randn(2, 10).cuda()
            _ = test_lin(test_x)
            device = "cuda"
        except Exception:
            device = "cpu"
    print(f"Hardware accelerator: {device}")

    # Initialize environment
    env = build_env(use_task_intent=CONFIG["use_task_intent"], template_net=CONFIG["template_net"])

    # Initialize agent & replay buffer
    agent = DiscreteSACAgent(
        state_dim=CONFIG["state_dim"],
        action_dim=CONFIG["action_dim"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"],
        actor_lr=CONFIG["actor_lr"],
        critic_lr=CONFIG["critic_lr"],
        alpha_lr=CONFIG["alpha_lr"],
        device=device
    )

    if latest_ckpt_path:
        agent.load(latest_ckpt_path)
        print(f"Loaded checkpoint weights & optimizer states from: {latest_ckpt_path}")

    replay_buffer = ReplayBuffer(capacity=CONFIG["replay_capacity"])

    # Open CSV in append mode if resuming, write mode if starting fresh
    if os.path.exists(log_csv) and start_ep > 0:
        csv_file = open(log_csv, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
    else:
        csv_file = open(log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "return", "steps", "task", "outcome", "critic_loss", "actor_loss", "alpha", "buffer_size"])

    recent_returns = []
    if os.path.exists(log_csv):
        try:
            with open(log_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    recent_returns.append(float(row["return"]))
            recent_returns = recent_returns[-1000:]
        except Exception:
            pass

    start_time = time.time()

    for episode in range(start_ep + 1, CONFIG["total_episodes"] + 1):
        obs = env.reset()
        done = False
        episode_return = 0.0
        step = 0
        task = "no_spawn"
        outcome = "timeout"
        last_losses = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": agent.alpha.item()}

        while not done:
            step += 1
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, done, info = env.step(action)

            if env.rl_veh:
                ego_id = env.rl_veh[0]
                if task == "no_spawn":
                    task = env._get_task(ego_id)

            if reward == 30.0:
                outcome = "arrival"
                done = True
            elif reward == -120.0:
                outcome = "collision"
                done = True
            elif step >= 25:
                done = True

            replay_buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            episode_return += reward

            # Update parameters if buffer is warm
            if len(replay_buffer) >= CONFIG["batch_size"]:
                losses = agent.update_parameters(replay_buffer, batch_size=CONFIG["batch_size"])
                if losses is not None:
                    last_losses = losses

        recent_returns.append(episode_return)
        if len(recent_returns) > 1000:
            recent_returns.pop(0)

        csv_writer.writerow([
            episode,
            episode_return,
            step,
            task,
            outcome,
            f"{last_losses['critic_loss']:.4f}",
            f"{last_losses['actor_loss']:.4f}",
            f"{last_losses['alpha']:.4f}",
            len(replay_buffer)
        ])
        csv_file.flush()

        if episode % CONFIG["log_freq"] == 0:
            avg_1000 = np.mean(recent_returns) if recent_returns else episode_return
            elapsed = time.time() - start_time
            print(f"[Ep {episode:6d}/{CONFIG['total_episodes']}] 1000-ep Avg Return: {avg_1000:6.2f} | "
                  f"Last Return: {episode_return:6.1f} | Buffer: {len(replay_buffer):6d} | "
                  f"Alpha: {last_losses['alpha']:.4f} | Time: {elapsed/60:6.1f}m")

        if episode % CONFIG["checkpoint_freq"] == 0 or episode == CONFIG["total_episodes"]:
            ckpt_path = os.path.join(checkpoint_dir, f"sac_model_ep_{episode}.pt")
            agent.save(ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

    env.terminate()
    csv_file.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    train()
