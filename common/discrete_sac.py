"""Discrete Soft Actor-Critic (SAC) implementation.

Faithful to Xiao et al. (2024), "Decision-Making for Autonomous Vehicles in
Random Task Scenarios at Unsignalized Intersection Using Deep Reinforcement Learning":
- 4-layer MLP: [obs_dim -> 256 -> 256 -> 64 -> 3]
- Actor learning rate: 0.0003
- Critic learning rate: 0.0005
- Replay memory capacity: 150,000
- Batch size: 1024
- Soft update parameter tau: 0.005
- Discount factor gamma: 0.90
- Target entropy auto-tuning for Discrete Action Spaces (Christodoulou, 2019)
"""
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    """Standard experience replay buffer matching paper capacity 150,000."""
    def __init__(self, capacity=150000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=1024):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DiscreteSACActor(nn.Module):
    """Actor network with paper's [256, 256, 64, 3] MLP architecture."""
    def __init__(self, state_dim, action_dim=3):
        super(DiscreteSACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc_out(x)
        action_probs = F.softmax(logits, dim=-1)
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return action_probs, log_action_probs

    def select_action(self, state, evaluate=False):
        """Select action: argmax if evaluate else categorical sample."""
        if not isinstance(state, torch.Tensor):
            device = next(self.parameters()).device
            state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action_probs, _ = self.forward(state)
        if evaluate:
            action = torch.argmax(action_probs, dim=-1)
            return action.item()
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            return action.item()


# Twin critic architecture Q_phi1, Q_phi2 for clipped double-Q estimation
class DiscreteSACCritic(nn.Module):
    """Twin Q-networks with paper's [256, 256, 64, 3] MLP architecture."""
    def __init__(self, state_dim, action_dim=3):
        super(DiscreteSACCritic, self).__init__()
        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_fc3 = nn.Linear(256, 64)
        self.q1_out = nn.Linear(64, action_dim)

        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_fc3 = nn.Linear(256, 64)
        self.q2_out = nn.Linear(64, action_dim)

    def forward(self, state):
        x1 = F.relu(self.q1_fc1(state))
        x1 = F.relu(self.q1_fc2(x1))
        x1 = F.relu(self.q1_fc3(x1))
        q1 = self.q1_out(x1)

        x2 = F.relu(self.q2_fc1(state))
        x2 = F.relu(self.q2_fc2(x2))
        x2 = F.relu(self.q2_fc3(x2))
        q2 = self.q2_out(x2)

        return q1, q2


class DiscreteSACAgent:
    """Discrete SAC agent coordinating Actor, Twin Critics, and Entropy."""
    def __init__(
        self,
        state_dim,
        action_dim=3,
        gamma=0.90,
        tau=0.005,
        actor_lr=0.0003,
        critic_lr=0.0005,
        alpha_lr=0.0003,
        target_entropy_ratio=0.98,
        device="cpu"
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # Actor
        self.actor = DiscreteSACActor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin Critics
        self.critic = DiscreteSACCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DiscreteSACCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Automatic Entropy Temperature tuning
        # Target entropy: -log(1/|A|) * ratio
        self.target_entropy = -np.log(1.0 / action_dim) * target_entropy_ratio
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            return self.actor.select_action(state_tensor, evaluate=evaluate)

    def update_parameters(self, replay_buffer, batch_size=1024):
        if len(replay_buffer) < batch_size:
            return None

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action_probs, next_log_probs = self.actor(next_state)
            next_q1_target, next_q2_target = self.critic_target(next_state)
            min_next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # Soft state value V(s') = \sum_a \pi(a|s') * [min Q(s', a) - \alpha \log \pi(a|s')]
            next_v = (next_action_probs * (min_next_q_target - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            target_q = reward + (1.0 - done) * self.gamma * next_v

        # Critic update
        current_q1, current_q2 = self.critic(state)
        q1_val = current_q1.gather(1, action)
        q2_val = current_q2.gather(1, action)

        critic_loss = F.mse_loss(q1_val, target_q) + F.mse_loss(q2_val, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        action_probs, log_probs = self.actor(state)
        with torch.no_grad():
            q1, q2 = self.critic(state)
            min_q = torch.min(q1, q2)

        # Policy objective: \sum_a \pi(a|s) * [\alpha \log \pi(a|s) - min Q(s, a)]
        actor_loss = (action_probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (temperature) update
        # Loss: - \alpha * \sum_a \pi(a|s) * [\log \pi(a|s) + \bar{\mathcal{H}}]
        entropy_diff = -(log_probs + self.target_entropy).detach()
        alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * entropy_diff)).sum(dim=-1).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft target update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item()
        }

    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'alpha_opt': self.alpha_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_opt'])