"""Evaluation script for trained checkpoints matching the paper's exact evaluation protocol:
- 1,000 episodes evaluated per task (Left, Straight, Right) -> 3,000 valid episodes total
- Deterministic policy (evaluation mode, no exploration noise)
- Metrics logged: success rate, collision rate, average return, average speed
- Outputs detailed CSV files: episode_metrics.csv and step_metrics.csv
"""
import os
import sys
import argparse
import csv
import numpy as np
import torch

# Ensure common imports resolve
COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMMON_DIR)

from discrete_sac import DiscreteSACAgent
from env_multitask_intersection import MultiTaskIntersectionEnv
from network_all_turning_intersection import AllTurningIntersectionNetwork

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, InFlows
from flow.controllers import RLController, IDMController


def build_env(use_task_intent=False, template_net=None):
    if template_net is None:
        template_net = os.path.expanduser('~/flow/examples/exp_inputs/scenarios/50mIntersection.net.xml')
        if not os.path.exists(template_net):
            # Fallback relative to flow repository
            fallback = os.path.abspath(os.path.join(COMMON_DIR, '..', '..', '..', 'examples', 'exp_inputs', 'scenarios', '50mIntersection.net.xml'))
            if os.path.exists(fallback):
                template_net = fallback

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {"noise": 0.2}),
        car_following_params=SumoCarFollowingParams(min_gap=1, max_speed=8),
        num_vehicles=0)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(min_gap=1, max_speed=8),
        num_vehicles=0)

    inflow = InFlows()
    for edge in ["E#L-X", "E#D-X", "E#R-X", "E#T-X"]:
        inflow.add(
            veh_type="human",
            edge=edge,
            probability=0.06,
            depart_lane="free",
            depart_speed="random",
            begin=1)
    inflow.add(
        veh_type="rl",
        edge="E#L-X",
        probability=1.0,
        depart_lane="free",
        depart_speed="random",
        begin=1,
        end=2)

    sim_params = SumoParams(sim_step=1 / 15, render=False, restart_instance=True)
    env_params = EnvParams(
        horizon=25,
        warmup_steps=0,
        sims_per_step=15,
        clip_actions=False,
        additional_params={
            "num_surrounding": 9,
            "detection_radius": 48,
            "target_velocity": 8,
            "speed_gain": 5,
            "use_task_intent": use_task_intent,
        },
    )
    net_params = NetParams(inflows=inflow, template=template_net)
    network = AllTurningIntersectionNetwork(
        name="eval_intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=InitialConfig())

    env = MultiTaskIntersectionEnv(env_params, sim_params, network)
    return env


def evaluate(checkpoint_path, use_task_intent, target_per_task=1000, output_dir="./eval_output", device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    state_dim = 61 if use_task_intent else 60
    agent = DiscreteSACAgent(state_dim=state_dim, action_dim=3, device=device)
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    env = build_env(use_task_intent=use_task_intent)

    task_counts = {"left": 0, "straight": 0, "right": 0}
    task_arrivals = {"left": 0, "straight": 0, "right": 0}
    task_collisions = {"left": 0, "straight": 0, "right": 0}
    task_returns = {"left": [], "straight": [], "right": []}
    task_speeds = {"left": [], "straight": [], "right": []}

    episode_csv = os.path.join(output_dir, "episode_metrics.csv")
    step_csv = os.path.join(output_dir, "step_metrics.csv")

    with open(episode_csv, "w", newline="", encoding="utf-8") as f_ep, \
         open(step_csv, "w", newline="", encoding="utf-8") as f_st:
        
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
                # Ego did not spawn
                continue

            # Record if this task quota is not yet satisfied
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

                if task_counts[task] % 100 == 0:
                    print(f"Task progress: Left={task_counts['left']}/{target_per_task}, "
                          f"Straight={task_counts['straight']}/{target_per_task}, "
                          f"Right={task_counts['right']}/{target_per_task}")

    env.terminate()

    # Print summary table
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
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--use_task_intent", action="store_true", help="Set for Config B (61-dim state)")
    parser.add_argument("--target_per_task", type=int, default=1000, help="Episodes evaluated per task (default 1000)")
    parser.add_argument("--output_dir", type=str, default="./eval_output", help="Directory for CSV logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint_path,
        use_task_intent=args.use_task_intent,
        target_per_task=args.target_per_task,
        output_dir=args.output_dir,
        device=args.device)