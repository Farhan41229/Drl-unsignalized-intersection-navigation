"""Hyperparameters and configuration for Config B (Exact Paper Setup).

Improvement B (Task-Intent mu_a):
- Observation: 61-dim (with task-intent mu_a)
- Architecture: 4-layer MLP [61 -> 256 -> 256 -> 64 -> 3]
- Algorithm: Discrete SAC (gamma=0.90, tau=0.005, actor_lr=0.0003, critic_lr=0.0005)
- Replay buffer: 150,000 capacity, batch_size=1024
- Training scale: 50,000 episodes
"""
import os

CONFIG = {
    "exp_tag": "multitask_intersection_B_exact",
    "use_task_intent": True,
    "state_dim": 61,
    "action_dim": 3,
    
    # Paper exact hyperparameters
    "gamma": 0.90,
    "tau": 0.005,
    "actor_lr": 0.0003,
    "critic_lr": 0.0005,
    "alpha_lr": 0.0003,
    "batch_size": 1024,
    "replay_capacity": 150000,
    "total_episodes": 50000,
    
    # Checkpointing
    "checkpoint_freq": 1000,
    "log_freq": 100,
    
    # Network geometry template
    "template_net": os.path.expanduser('~/flow/examples/exp_inputs/scenarios/50mIntersection.net.xml'),
}