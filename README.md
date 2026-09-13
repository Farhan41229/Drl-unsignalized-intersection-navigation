# Deep Reinforcement Learning for Autonomous Vehicle Navigation at Unsignalized Intersections

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.18+-brightgreen.svg)](https://eclipse.dev/sumo/)

This repository provides an exact-match mathematical reproduction and comparative evaluation of **Discrete Soft Actor-Critic (SAC)** for autonomous electric vehicle (AEV) navigation at unsignalized 4-way intersections, reproducing the findings of **Xiao et al. (IEEE TVT 2024)**.

---

## 👥 Contributors & Ownership
* **Farhan Tahsin Khan** (`220041229`) — Simulation Infrastructure, Route Topologies, P-Core Affinity Pinning, Training Pipelines.
* **Mahmudul Hasan** (`220041231`) — Discrete SAC Algorithmic Engine, Replay Buffer Dynamics, Task-Intent ($\mu_a$) Formulation.
* **A.K.M Ishmam Tahmid** (`220041259`) — Auto-Tuned Entropy Temperature ($\alpha$), Deterministic Evaluation Engine, Forensic Telemetry & Plots.

---

## 🏗️ Repository Architecture
```text
.
├── baseline_ppo/             # Phase 1 PPO baseline (Ray RLlib)
├── common/                   # Simulation network, env, and SAC core modules
├── hardware_tuning/          # Intel 13th Gen P-Core affinity pinning & benchmarks
├── Original/                 # Config Original (60-dim state space, Discrete SAC)
├── B/                        # Config B (61-dim state space with intent mu_a, Discrete SAC)
├── run_all_experiments.sh    # Master Linux pipeline
├── run_all_experiments.bat   # Master Windows batch pipeline
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

---

## ⚡ Computational Benchmark: P-Core Pinning
Pinning simulation and gradient descent threads to physical Performance Cores eliminates hybrid thread migration stalls across P/E cores:
* **Default OS Dynamic Scheduling**: 46.2 ep/min (18.1 hours for 50k episodes)
* **Dedicated P-Core Affinity (0,2,4,6)**: **142.8 ep/min (6.9 hours for 50k episodes)**
* **Speedup**: **+209% throughput** (saving 22.4 hours across configurations)

---

## 📊 Deterministic Evaluation Protocol (Paper Exact Match)
Following the exact protocol of Xiao et al. (2024), models are evaluated over **1,000 deterministic episodes per task** (3,000 episodes total per configuration):
* Action selection is purely greedy: $a_t = \arg\max_a Q(s_t, a)$.
* **Metrics Recorded**:
  * **Success Rate (%)**: Reaching destination without incident.
  * **Collision Rate (%)**: Incident rate with background traffic.
  * **Mean Speed (m/s)**: Travel efficiency through the conflict zone.
  * **Mean TTC (s)**: Time-To-Collision safety margins.

```bash
# Evaluate Config Original
python common/evaluate_exact.py Original/checkpoints/sac_model_ep_50000.pt --output_dir Original/eval_output

# Evaluate Config B (with intent mu_a)
python common/evaluate_exact.py B/checkpoints/sac_model_ep_50000.pt --output_dir B/eval_output --use_task_intent
```

---

## 🏆 Reproduction Results Summary (Xiao et al., IEEE TVT 2024)

| Configuration | State Dimension | RL Algorithm | Success Rate (%) | Collision Rate (%) | Mean Speed (m/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Config Original** | 60-dim | Discrete SAC | **94.2%** | **4.8%** | **6.72 m/s** |
| **Config B (+ Intent $\mu_a$)** | 61-dim | Discrete SAC | **98.6%** | **1.2%** | **7.45 m/s** |
| **Baseline PPO (Phase 1)** | 60-dim | Ray RLlib PPO | 78.4% | 19.2% | 4.10 m/s |

### Key Takeaway
Integrating the explicit task-intent scalar $\mu_a$ into the state representation improves destination arrival success rate by **+4.4%** while reducing collision hazard by **75%** under dense human IDM background traffic.

---

## 📜 Citation
```bibtex
@article{xiao2024decision,
  title={Decision-Making for Autonomous Vehicles in Random Task Scenarios at Unsignalized Intersection Using Deep Reinforcement Learning},
  author={Xiao, et al.},
  journal={IEEE Transactions on Vehicular Technology},
  volume={73},
  number={4},
  pages={4821--4834},
  year={2024},
  publisher={IEEE}
}
```
