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

def fast_evaluate(checkpoint_path, target_per_task=350, output_dir="./eval_output"):
    os.makedirs(output_dir, exist_ok=True)
    agent = DiscreteSACAgent(state_dim=60, action_dim=3, device="cpu")
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    env = build_env(use_task_intent=False)

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
                        pos = env.k.vehicle.get_2d_position(ego_id)
                        x, y = pos[0], pos[1]
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

                if task_counts[task] % 50 == 0:
                    print(f"Task progress: Left={task_counts['left']}/{target_per_task}, "
                          f"Straight={task_counts['straight']}/{target_per_task}, "
                          f"Right={task_counts['right']}/{target_per_task}", flush=True)

    env.terminate()
    print("Evaluation finished successfully!")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sac_model_ep_50000.pt"
    fast_evaluate(ckpt)


# Deprecated debug rollout line 1
# Deprecated debug rollout line 2
# Deprecated debug rollout line 3
# Deprecated debug rollout line 4
# Deprecated debug rollout line 5
# Deprecated debug rollout line 6
# Deprecated debug rollout line 7
# Deprecated debug rollout line 8
# Deprecated debug rollout line 9
# Deprecated debug rollout line 10
# Deprecated debug rollout line 11
# Deprecated debug rollout line 12
# Deprecated debug rollout line 13
# Deprecated debug rollout line 14
# Deprecated debug rollout line 15
# Deprecated debug rollout line 16
# Deprecated debug rollout line 17
# Deprecated debug rollout line 18
# Deprecated debug rollout line 19
# Deprecated debug rollout line 20
# Deprecated debug rollout line 21
# Deprecated debug rollout line 22
# Deprecated debug rollout line 23
# Deprecated debug rollout line 24
# Deprecated debug rollout line 25
# Deprecated debug rollout line 26
# Deprecated debug rollout line 27
# Deprecated debug rollout line 28
# Deprecated debug rollout line 29
# Deprecated debug rollout line 30
# Deprecated debug rollout line 31
# Deprecated debug rollout line 32
# Deprecated debug rollout line 33
# Deprecated debug rollout line 34
# Deprecated debug rollout line 35
# Deprecated debug rollout line 36
# Deprecated debug rollout line 37
# Deprecated debug rollout line 38
# Deprecated debug rollout line 39
# Deprecated debug rollout line 40
# Deprecated debug rollout line 41
# Deprecated debug rollout line 42
# Deprecated debug rollout line 43
# Deprecated debug rollout line 44
# Deprecated debug rollout line 45
# Deprecated debug rollout line 46
# Deprecated debug rollout line 47
# Deprecated debug rollout line 48
# Deprecated debug rollout line 49
# Deprecated debug rollout line 50
# Deprecated debug rollout line 51
# Deprecated debug rollout line 52
# Deprecated debug rollout line 53
# Deprecated debug rollout line 54
# Deprecated debug rollout line 55
# Deprecated debug rollout line 56
# Deprecated debug rollout line 57
# Deprecated debug rollout line 58
# Deprecated debug rollout line 59
# Deprecated debug rollout line 60
# Deprecated debug rollout line 61
# Deprecated debug rollout line 62
# Deprecated debug rollout line 63
# Deprecated debug rollout line 64
# Deprecated debug rollout line 65