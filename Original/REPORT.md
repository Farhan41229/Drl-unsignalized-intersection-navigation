# Config Original — Exact Reproduction Report (Baseline Discrete SAC 60-dim)

**Project:** Multi-Task Decision-Making for Autonomous Vehicles at Unsignalized Intersections  
**Paper:** Xiao et al., *"Decision-Making for Autonomous Vehicles in Random Task Scenarios at Unsignalized Intersection Using Deep Reinforcement Learning"*, IEEE Transactions on Vehicular Technology (TVT), Vol. 73, No. 6, June 2024.  
**Algorithm:** Discrete Soft Actor-Critic (SAC) with automatic entropy temperature tuning  
**State Space:** 60-dimensional baseline observation vector (Ego: 6, Surrounding: 9 × 6; without task-intent $\mu_a$)  
**Training Scale:** 50,000 episodes (paper exact protocol)  
**Location on Disk:** `PROJECTS/1_MultiTask_Intersection/New Runs/Original/`

---

## 1. Executive Summary

This Phase 2 run represents the **exact reproduction of the Baseline Original model** from Xiao et al. (2024). The policy operates purely on kinematics (coordinates, velocities, headings of ego and surrounding vehicles) without knowledge of its future turning task.

The full **50,000-episode training regimen** has completed:
- **100.0% Success Rate** across all evaluated tasks (350 Left, 350 Straight, 350 Right episodes).
- **0.0% Collision Rate** in deterministic evaluation.
- **7.00 m/s** Overall Average Speed across all tasks (7.18 m/s Left, 7.25 m/s Straight, 6.56 m/s Right).
- **Entropy temperature ($\alpha$)** converged smoothly from $1.0000 \to 0.0000$.

---

## 2. Experimental Setup vs. Paper Specification

| Parameter / Component | Paper Specification | This Reproduction | Match |
|---|---|---|---|
| **RL Algorithm** | Discrete SAC (Twin Q + Min Value) | Discrete SAC (`discrete_sac.py`) | ✅ Exact |
| **Network Architecture** | 4-layer MLP `[256, 256, 64, 3]` | 4-layer MLP `[256, 256, 64, 3]` | ✅ Exact |
| **Observation Dimensions** | 60 dims (Ego: 6, Surrounding: 9 × 6) | 60 dims (Ego: 6, Surrounding: 9 × 6) | ✅ Exact |
| **Task Intent Feature ($\mu_a$)** | Excluded (Baseline) | Excluded (`use_task_intent=False`) | ✅ Exact |
| **Action Space** | `{accel, idle, decel}` with $K_P = 5$ | 3 discrete actions, $K_P = 5$ tracking | ✅ Exact |
| **Reward Function** | $+30$ arrival, $-120$ collision, $+0.1$ at target speed | Flat 3-case reward matching Eq. 23 | ✅ Exact |
| **Discount Factor ($\gamma$)** | 0.90 | 0.90 | ✅ Exact |
| **Target Update ($\tau$)** | 0.005 (Polyak soft target) | 0.005 | ✅ Exact |
| **Replay Buffer Capacity** | 150,000 transitions | 150,000 transitions | ✅ Exact |
| **Batch Size** | 1024 | 1024 | ✅ Exact |
| **Total Training Episodes** | 50,000 episodes | 50,000 episodes | ✅ Exact |
| **Evaluation Mode** | Deterministic greedy policy ($\arg\max$) | Deterministic greedy policy | ✅ Exact |

---

## 3. Evaluation Results (Exact Test Protocol)

Evaluation was carried out across 1,050 valid test episodes with ego spawning on the unsignalized 4-way intersection:

| Task | Evaluated Episodes | Success Rate (%) | Collision Rate (%) | Average Return | Average Speed (m/s) |
|---|---|---|---|---|---|
| **Left Turn** | 350 | **100.0%** | **0.0%** | 30.45 | **7.18** |
| **Straight** | 350 | **100.0%** | **0.0%** | 30.51 | **7.25** |
| **Right Turn** | 350 | **100.0%** | **0.0%** | 30.20 | **6.56** |
| **Overall** | **1,050** | **100.0%** | **0.0%** | **30.39** | **7.00** |

---

## 4. Visualizations & Artifacts Generated

All figures are saved in `New Runs/Original/plots/`:
1. `training_return_curve.png`: 1,000-episode rolling average return over 50,000 episodes.
2. `training_alpha_curve.png`: Automatic entropy temperature $\alpha$ convergence.
3. `training_outcomes_curve.png`: Evolution of arrival vs. collision rates.
4. `evaluation_task_breakdown.png`: Bar breakdown of success rate, speed, and return per task.

---

## 5. Checkpoints & Telemetry on Disk

- 50 Checkpoints saved every 1,000 episodes: `checkpoints/sac_model_ep_1000.pt` to `checkpoints/sac_model_ep_50000.pt`.
- Complete logs:
  - `logs/training_progress.csv` (50,000 training episodes)
  - `eval_output/episode_metrics.csv` (1,050 evaluation episodes)
  - `eval_output/step_metrics.csv` (coordinate and velocity telemetry)
