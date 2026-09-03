import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

eval_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/eval_output"
ep_csv = os.path.join(eval_dir, "episode_metrics.csv")
output_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/plots"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(ep_csv) or os.path.getsize(ep_csv) == 0:
    print("episode_metrics.csv is not yet ready or empty.")
    sys.exit(0)

df = pd.read_csv(ep_csv)
tasks = ['left', 'straight', 'right']

stats = {}
for t in tasks:
    sub = df[df['task'] == t]
    n = len(sub)
    if n == 0:
        stats[t] = {'success_rate': 0.0, 'collision_rate': 0.0, 'avg_return': 0.0, 'avg_speed': 0.0, 'n': 0}
    else:
        stats[t] = {
            'success_rate': (sub['outcome'] == 'arrival').mean() * 100.0,
            'collision_rate': (sub['outcome'] == 'collision').mean() * 100.0,
            'avg_return': sub['total_return'].mean(),
            'avg_speed': sub['avg_speed'].mean(),
            'n': n
        }

# Plot Evaluation Metrics across Tasks
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 1. Success Rate
ax = axes[0]
vals = [stats[t]['success_rate'] for t in tasks]
bars = ax.bar([t.capitalize() for t in tasks], vals, color=['#1f77b4', '#2ca02c', '#ff7f0e'], width=0.55)
ax.set_ylabel('Success / Arrival Rate (%)', fontweight='bold')
ax.set_title('Evaluation Success Rate by Driving Task', fontweight='bold')
ax.set_ylim(0, 105)
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 2, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

# 2. Average Speed
ax = axes[1]
speeds = [stats[t]['avg_speed'] for t in tasks]
bars = ax.bar([t.capitalize() for t in tasks], speeds, color=['#1f77b4', '#2ca02c', '#ff7f0e'], width=0.55)
ax.axhline(8.0, color='red', linestyle='--', label='Target Speed (8 m/s)')
ax.set_ylabel('Average Speed (m/s)', fontweight='bold')
ax.set_title('Evaluation Speed by Driving Task', fontweight='bold')
ax.set_ylim(0, 9.5)
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 0.15, f"{y:.2f}", ha='center', va='bottom', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right')

# 3. Average Return
ax = axes[2]
returns = [stats[t]['avg_return'] for t in tasks]
bars = ax.bar([t.capitalize() for t in tasks], returns, color=['#1f77b4', '#2ca02c', '#ff7f0e'], width=0.55)
ax.set_ylabel('Average Return', fontweight='bold')
ax.set_title('Evaluation Return by Driving Task', fontweight='bold')
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 0.5, f"{y:.2f}", ha='center', va='bottom', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(output_dir, 'evaluation_task_breakdown.png')
fig.savefig(out_path)
plt.close(fig)
print(f"Generated evaluation plots at: {out_path}")
