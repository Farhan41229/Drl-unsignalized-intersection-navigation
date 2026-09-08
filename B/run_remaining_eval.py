import os
import sys
import argparse
import csv
import numpy as np
import torch
import pandas as pd

COMMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, COMMON_DIR)

from discrete_sac import DiscreteSACAgent
from evaluate_exact import build_env

def run_remaining():
    eval_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/eval_output"
    ep_csv = os.path.join(eval_dir, "episode_metrics.csv")
    st_csv = os.path.join(eval_dir, "step_metrics.csv")
    checkpoint_path = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/checkpoints/sac_model_ep_50000.pt"

    target_per_task = 1000
    task_counts = {"left": 0, "straight": 0, "right": 0}
    task_arrivals = {"left": 0, "straight": 0, "right": 0}
    task_collisions = {"left": 0, "straight": 0, "right": 0}
    task_returns = {"left": [], "straight": [], "right": []}
    task_speeds = {"left": [], "straight": [], "right": []}

    episode_idx = 0
    if os.path.exists(ep_csv) and os.path.getsize(ep_csv) > 0:
        df = pd.read_csv(ep_csv)
        for _, row in df.iterrows():
            t = str(row['task'])
            out = str(row['outcome'])
            ret = float(row['total_return'])
            spd = float(row['avg_speed'])
            task_counts[t] += 1
            task_returns[t].append(ret)
            task_speeds[t].append(spd)
            if out == 'arrival':
                task_arrivals[t] += 1
            elif out == 'collision':
                task_collisions[t] += 1
            episode_idx = max(episode_idx, int(row['episode_id']))
        print("Existing counts loaded:", task_counts)

    if all(task_counts[t] >= target_per_task for t in ["left", "straight", "right"]):
        print("All quotas already complete!")
        return

    agent = DiscreteSACAgent(state_dim=61, action_dim=3, device="cpu")
    agent.load(checkpoint_path)
    env = build_env(use_task_intent=True)

    with open(ep_csv, "a", newline="", encoding="utf-8") as f_ep, \
         open(st_csv, "a", newline="", encoding="utf-8") as f_st:
        
        ep_writer = csv.writer(f_ep)
        st_writer = csv.writer(f_st)

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
    print("Full 1,000 episodes per task completed!")

if __name__ == "__main__":
    run_remaining()
