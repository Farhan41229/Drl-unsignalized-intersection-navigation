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
