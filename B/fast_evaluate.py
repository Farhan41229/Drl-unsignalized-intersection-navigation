import os
import sys
import argparse
import csv
import numpy as np
import torch

COMMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, COMMON_DIR)

from discrete_sac import DiscreteSACAgent
from evaluate_exact import build_env

def fast_evaluate(checkpoint_path, target_per_task=1000, output_dir="./eval_output"):
    os.makedirs(output_dir, exist_ok=True)
    agent = DiscreteSACAgent(state_dim=61, action_dim=3, device="cpu")
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    env = build_env(use_task_intent=True)

    task_counts = {"left": 0, "straight": 0, "right": 0}
    task_arrivals = {"left": 0, "straight": 0, "right": 0}
    task_collisions = {"left": 0, "straight": 0, "right": 0}
    task_returns = {"left": [], "straight": [], "right": []}
    task_speeds = {"left": [], "straight": [], "right": []}

    ep_csv = os.path.join(output_dir, "episode_metrics.csv")
    st_csv = os.path.join(output_dir, "step_metrics.csv")

    with open(ep_csv, "w", newline="", encoding="utf-8") as f_ep, \
         open(st_csv, "w", newline="", encoding="utf-8") as f_st:
        
        ep_writer = csv.writer(f_ep)
        ep_writer.writerow(["episode_id", "task", "outcome", "total_return", "steps", "avg_speed"])

        st_writer = csv.writer(f_st)
        st_writer.writerow(["episode_id", "step", "ego_x", "ego_y", "ego_speed", "action", "reward"])

        episode_idx = 0
        while any(task_counts[t] < target_per_task for t in ["left", "straight", "right"]):
            episode_idx += 1
            obs = env.reset()
            done = False
            total_reward = 0.0
            step = 0
            task = None
            outcome = "timeout"
            step_speeds = []

            while not done:
                step += 1
                action = agent.select_action(obs, evaluate=True)
                next_obs, reward, done, info = env.step(action)

                if env.rl_veh:
                    ego_id = env.rl_veh[0]
                    if task is None:
                        task = env._get_task(ego_id)
                    if ego_id in env.k.vehicle.get_ids():
                        speed = env.k.vehicle.get_speed(ego_id)
                        x, y, _ = env.k.vehicle.get_orientation(ego_id)
                        step_speeds.append(speed)
                        st_writer.writerow([episode_idx, step, x, y, speed, action, reward])

                if reward == 30.0:
                    outcome = "arrival"
                    done = True
                elif reward == -120.0:
                    outcome = "collision"
                    done = True
                elif step >= 25:
                    done = True

                total_reward += reward
                obs = next_obs

            if task is None:
                continue

            if task_counts[task] < target_per_task:
                task_counts[task] += 1
                avg_speed = float(np.mean(step_speeds)) if step_speeds else 0.0
                task_returns[task].append(total_reward)
                task_speeds[task].append(avg_speed)

                if outcome == "arrival":
                    task_arrivals[task] += 1
                elif outcome == "collision":
                    task_collisions[task] += 1

                ep_writer.writerow([episode_idx, task, outcome, total_reward, step, avg_speed])
                f_ep.flush()
                f_st.flush()

                if task_counts[task] % 100 == 0:
                    print(f"Task progress: Left={task_counts['left']}/{target_per_task}, "
                          f"Straight={task_counts['straight']}/{target_per_task}, "
                          f"Right={task_counts['right']}/{target_per_task}", flush=True)

    env.terminate()

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY (Paper Exact Protocol)")
    print("=" * 60)
    print(f"{'Task':<10} | {'Episodes':<8} | {'Arrival %':<10} | {'Collision %':<12} | {'Avg Return':<12} | {'Avg Speed (m/s)':<15}")
    print("-" * 75)
    for t in ["left", "straight", "right"]:
        n = task_counts[t]
        arr_pct = (task_arrivals[t] / n * 100.0) if n > 0 else 0.0
        col_pct = (task_collisions[t] / n * 100.0) if n > 0 else 0.0
        ret_mean = np.mean(task_returns[t]) if task_returns[t] else 0.0
        spd_mean = np.mean(task_speeds[t]) if task_speeds[t] else 0.0
        print(f"{t:<10} | {n:<8} | {arr_pct:6.2f}%    | {col_pct:6.2f}%      | {ret_mean:8.2f}     | {spd_mean:6.2f}")
    print("=" * 60)

if __name__ == "__main__":
    fast_evaluate("checkpoints/sac_model_ep_50000.pt", target_per_task=1000)
