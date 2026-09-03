import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

logs_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/logs"
csv_path = os.path.join(logs_dir, "training_progress.csv")
output_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/plots"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# 1. Training Return Curve (Rolling 1,000-episode window matching Paper Fig 5)
fig, ax = plt.subplots(figsize=(9, 5))
df['rolling_return'] = df['return'].rolling(window=1000, min_periods=100).mean()
ax.plot(df['episode'], df['rolling_return'], color='#ff7f0e', linewidth=2, label='Config B (Discrete SAC with $\mu_a$)')
ax.set_xlabel('Training Episodes', fontweight='bold')
ax.set_ylabel('Average Return (1,000-ep rolling mean)', fontweight='bold')
ax.set_title('Config B: Training Progress Average Return (Xiao et al., 2024)', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_return_curve.png'))
plt.close(fig)

# 2. Entropy Temperature (alpha) curve
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(df['episode'], df['alpha'], color='#2ca02c', linewidth=1.5, label=r'Temperature $\alpha$')
ax.set_xlabel('Training Episodes', fontweight='bold')
ax.set_ylabel(r'Entropy Temperature $\alpha$', fontweight='bold')
ax.set_title(r'Config B: Adaptive Entropy Temperature $\alpha$ Convergence', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_alpha_curve.png'))
plt.close(fig)

# 3. Rolling Success & Collision Rates
fig, ax = plt.subplots(figsize=(9, 5))
df['is_arrival'] = (df['outcome'] == 'arrival').astype(float)
df['is_collision'] = (df['outcome'] == 'collision').astype(float)
df['roll_arrival'] = df['is_arrival'].rolling(window=1000, min_periods=100).mean() * 100.0
df['roll_collision'] = df['is_collision'].rolling(window=1000, min_periods=100).mean() * 100.0

ax.plot(df['episode'], df['roll_arrival'], color='#1f77b4', linewidth=2, label='Arrival Rate (%)')
ax.plot(df['episode'], df['roll_collision'], color='#d62728', linewidth=2, label='Collision Rate (%)')
ax.set_xlabel('Training Episodes', fontweight='bold')
ax.set_ylabel('Rate (%) [1,000-ep rolling]', fontweight='bold')
ax.set_title('Config B: Training Success and Collision Rates', fontsize=13, fontweight='bold')
ax.set_ylim(-2, 102)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='center right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_outcomes_curve.png'))
plt.close(fig)

print("Generated training plots in:", output_dir)
