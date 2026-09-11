# Config B — Exact Reproduction Report (Discrete SAC with Task-Intent $\mu_a$)

**Project:** Multi-Task Decision-Making for Autonomous Vehicles at Unsignalized Intersections  
**Paper:** Xiao et al., *"Decision-Making for Autonomous Vehicles in Random Task Scenarios at Unsignalized Intersection Using Deep Reinforcement Learning"*, IEEE Transactions on Vehicular Technology (TVT), Vol. 73, No. 6, June 2024.  
**Algorithm:** Discrete Soft Actor-Critic (SAC) with automatic entropy temperature tuning  
**State Space:** 61-dimensional observation vector (with ego task-intent $\mu_a$, paper Eq. 27–28)  
**Training Scale:** 50,000 episodes (paper exact protocol)  
**Location on Disk:** `PROJECTS/1_MultiTask_Intersection/New Runs/B/`

---

## 1. Executive Summary

In Phase 1, Config B was explored using PPO with 1.2M episodes. **This Phase 2 run represents the exact algorithmic reproduction of the paper's Discrete SAC formulation** as specified in Sections III, IV-B, and V of Xiao et al. (2024).

The training of **50,000 episodes** has completed with high stability:
- **100.0%** Success Rate in evaluation across all test episodes.
- **0.0%** Collision Rate across all test episodes.
- **7.01 m/s** Average Speed for straight driving, **6.91 m/s** for left turns, and **6.32 m/s** for right turns (against an 8.0 m/s ceiling).
- **Entropy temperature ($\alpha$)** converged smoothly from $1.0 \to 0.0000$, transitioning from exploratory actions to a deterministic, high-reward policy.

---

## 2. Experimental Setup vs. Paper Specification

| Parameter / Component | Paper Specification | This Reproduction | Match |
|---|---|---|---|
| **RL Algorithm** | Discrete SAC (Twin Q + Min Value) | Discrete SAC (`discrete_sac.py`) | ✅ Exact |
| **Network Architecture** | 4-layer MLP `[256, 256, 64, 3]` | 4-layer MLP `[256, 256, 64, 3]` | ✅ Exact |
| **Observation Dimensions** | 61 dims (Ego: 7, Surrounding: 9 × 6) | 61 dims (Ego: 7, Surrounding: 9 × 6) | ✅ Exact |
| **Task Intent Feature ($\mu_a$)** | Turn: $\| \phi_a - \theta_a \|$, Straight: $\arctan(d_a/d_{total})$ | Eq. 28 implemented in env | ✅ Exact |
| **Action Space** | `{accel, idle, decel}` with $K_P = 5$ | 3 discrete actions, $K_P = 5$ tracking | ✅ Exact |
| **Reward Function** | $+30$ arrival, $-120$ collision, $+0.1$ at target speed | Flat 3-case reward matching Eq. 23 | ✅ Exact |
| **Discount Factor ($\gamma$)** | 0.90 | 0.90 | ✅ Exact |
| **Target Update ($\tau$)** | 0.005 (Polyak soft target) | 0.005 | ✅ Exact |
| **Replay Buffer Capacity** | 150,000 transitions | 150,000 transitions | ✅ Exact |
| **Batch Size** | 1024 | 1024 | ✅ Exact |
| **Actor / Critic Learning Rates**| $3 \times 10^{-4}$ / $5 \times 10^{-4}$ | $3 \times 10^{-4}$ / $5 \times 10^{-4}$ | ✅ Exact |
| **Total Training Episodes** | 50,000 episodes | 50,000 episodes | ✅ Exact |
| **Evaluation Mode** | Deterministic greedy policy ($\arg\max$) | Deterministic greedy policy | ✅ Exact |

---

## 3. Evaluation Results (Exact Test Protocol)

Evaluation was carried out across 1,083 valid test episodes with ego spawning on the unsignalized 4-way intersection:

| Task | Evaluated Episodes | Success Rate (%) | Collision Rate (%) | Average Return | Average Speed (m/s) |
|---|---|---|---|---|---|
| **Left Turn** | 390 | **100.0%** | **0.0%** | 30.41 | **6.91** |
| **Straight** | 337 | **100.0%** | **0.0%** | 30.49 | **7.01** |
| **Right Turn** | 356 | **100.0%** | **0.0%** | 30.17 | **6.32** |
| **Overall** | **1,083** | **100.0%** | **0.0%** | **30.36** | **6.75** |

### Key Metrics Comparison (Paper vs. Phase 1 vs. Phase 2)

| Metric | Paper (Fig. 9 / Table II) | Phase 1 PPO (9,360 iters) | **Phase 2 SAC (Exact 50k)** |
|---|---|---|---|
| **Algorithm** | Discrete SAC | PPO (`ray.rllib`) | **Discrete SAC** |
| **Episodes Trained** | 50,000 | ~1,200,000 | **50,000** |
| **Left Turn Success** | High (~90%+) | 100.0% | **100.0%** |
| **Straight Success** | High (~90%+) | 94.0% (1 crash) | **100.0%** |
| **Right Turn Success** | High (~90%+) | 100.0% | **100.0%** |
| **Average Crossing Speed** | ~6.5 – 7.2 m/s | ~6.4 – 7.1 m/s | **6.75 m/s** |

---

## 4. Visualizations & Artifacts Generated

All figures are rendered and saved in `New Runs/B/plots/`:
1. `training_return_curve.png`: 1,000-episode rolling average return over the full 50,000 episodes (matches Paper Figure 5).
2. `training_alpha_curve.png`: Automatic entropy temperature $\alpha$ decay and convergence.
3. `training_outcomes_curve.png`: Longitudinal evolution of arrival rate vs. collision rate.
4. `evaluation_task_breakdown.png`: Bar comparison of success rate, speed, and return per driving task.

---

## 5. Checkpoints & Reproducibility

- Checkpoints saved every 1,000 episodes: `checkpoints/sac_model_ep_1000.pt` to `checkpoints/sac_model_ep_50000.pt`.
- Complete per-step and per-episode metrics logged:
  - `logs/training_progress.csv` (all 50,000 training episodes)
  - `eval_output/episode_metrics.csv` (evaluation episodes)
  - `eval_output/step_metrics.csv` (detailed coordinate and speed trajectories)
