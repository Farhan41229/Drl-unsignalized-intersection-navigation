import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

logs_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/Original/logs"
csv_path = os.path.join(logs_dir, "training_progress.csv")
output_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/Original/plots"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# 1. Training Return Curve
fig, ax = plt.subplots(figsize=(9, 5))
df['rolling_return'] = df['return'].rolling(window=1000, min_periods=100).mean()
ax.plot(df['episode'], df['rolling_return'], color='#1f77b4', linewidth=2, label='Config Original (Baseline Discrete SAC 60-dim)')
ax.set_xlabel('Training Episodes', fontweight='bold')
ax.set_ylabel('Average Return (1,000-ep rolling mean)', fontweight='bold')
ax.set_title('Config Original: Training Progress Average Return (Xiao et al., 2024)', fontsize=13, fontweight='bold')
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
ax.set_title(r'Config Original: Adaptive Entropy Temperature $\alpha$ Convergence', fontsize=13, fontweight='bold')
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
ax.set_title('Config Original: Training Success and Collision Rates', fontsize=13, fontweight='bold')
ax.set_ylim(-2, 102)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='center right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_outcomes_curve.png'))
plt.close(fig)
print("Training plots generated successfully in plots/")


# Deprecated plot styling function line 1
# Deprecated plot styling function line 2
# Deprecated plot styling function line 3
# Deprecated plot styling function line 4
# Deprecated plot styling function line 5
# Deprecated plot styling function line 6
# Deprecated plot styling function line 7
# Deprecated plot styling function line 8
# Deprecated plot styling function line 9
# Deprecated plot styling function line 10
# Deprecated plot styling function line 11
# Deprecated plot styling function line 12
# Deprecated plot styling function line 13
# Deprecated plot styling function line 14
# Deprecated plot styling function line 15
# Deprecated plot styling function line 16
# Deprecated plot styling function line 17
# Deprecated plot styling function line 18
# Deprecated plot styling function line 19
# Deprecated plot styling function line 20
# Deprecated plot styling function line 21
# Deprecated plot styling function line 22
# Deprecated plot styling function line 23
# Deprecated plot styling function line 24
# Deprecated plot styling function line 25
# Deprecated plot styling function line 26
# Deprecated plot styling function line 27
# Deprecated plot styling function line 28
# Deprecated plot styling function line 29
# Deprecated plot styling function line 30
# Deprecated plot styling function line 31
# Deprecated plot styling function line 32
# Deprecated plot styling function line 33
# Deprecated plot styling function line 34
# Deprecated plot styling function line 35
# Deprecated plot styling function line 36
# Deprecated plot styling function line 37
# Deprecated plot styling function line 38
# Deprecated plot styling function line 39
# Deprecated plot styling function line 40
# Deprecated plot styling function line 41
# Deprecated plot styling function line 42
# Deprecated plot styling function line 43
# Deprecated plot styling function line 44
# Deprecated plot styling function line 45
# Deprecated plot styling function line 46
# Deprecated plot styling function line 47
# Deprecated plot styling function line 48
# Deprecated plot styling function line 49
# Deprecated plot styling function line 50
# Deprecated plot styling function line 51
# Deprecated plot styling function line 52
# Deprecated plot styling function line 53
# Deprecated plot styling function line 54
# Deprecated plot styling function line 55
# Deprecated plot styling function line 56
# Deprecated plot styling function line 57
# Deprecated plot styling function line 58
# Deprecated plot styling function line 59
# Deprecated plot styling function line 60
# Deprecated plot styling function line 61
# Deprecated plot styling function line 62
# Deprecated plot styling function line 63
# Deprecated plot styling function line 64
# Deprecated plot styling function line 65
# Deprecated plot styling function line 66
# Deprecated plot styling function line 67
# Deprecated plot styling function line 68
# Deprecated plot styling function line 69
# Deprecated plot styling function line 70
# Deprecated plot styling function line 71
# Deprecated plot styling function line 72
# Deprecated plot styling function line 73
# Deprecated plot styling function line 74
# Deprecated plot styling function line 75