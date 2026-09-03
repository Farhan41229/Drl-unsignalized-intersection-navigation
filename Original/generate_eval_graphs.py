import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

eval_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/Original/eval_output"
plots_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/Original/plots"
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(os.path.join(eval_dir, "episode_metrics.csv"))
tasks = ['left', 'straight', 'right']

rates = []
speeds = []
returns = []

for t in tasks:
    sub = df[df['task'] == t]
    rates.append((sub['outcome'] == 'arrival').mean() * 100.0)
    speeds.append(sub['avg_speed'].mean())
    returns.append(sub['total_return'].mean())

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Success Rate
axes[0].bar(tasks, rates, color=['#1f77b4', '#2ca02c', '#ff7f0e'], width=0.55, edgecolor='black', linewidth=0.8)
axes[0].set_ylabel('Success Rate (%)', fontweight='bold')
axes[0].set_ylim(0, 105)
axes[0].set_title('Success Rate by Task', fontweight='bold')
for i, v in enumerate(rates):
    axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

# Avg Speed
axes[1].bar(tasks, speeds, color=['#1f77b4', '#2ca02c', '#ff7f0e'], width=0.55, edgecolor='black', linewidth=0.8)
axes[1].set_ylabel('Average Speed (m/s)', fontweight='bold')
axes[1].set_ylim(0, 9)
axes[1].axhline(y=8.0, color='r', linestyle='--', label='Speed Limit (8 m/s)')
axes[1].set_title('Average Speed by Task', fontweight='bold')
axes[1].legend(loc='lower right')
for i, v in enumerate(speeds):
    axes[1].text(i, v + 0.2, f"{v:.2f} m/s", ha='center', fontweight='bold')

# Avg Return
axes[2].bar(tasks, returns, color=['#1f77b4', '#2ca02c', '#ff7f0e'], width=0.55, edgecolor='black', linewidth=0.8)
axes[2].set_ylabel('Average Return', fontweight='bold')
axes[2].set_ylim(0, 35)
axes[2].set_title('Average Return by Task', fontweight='bold')
for i, v in enumerate(returns):
    axes[2].text(i, v + 0.8, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(plots_dir, 'evaluation_task_breakdown.png'))
plt.close(fig)
print("Evaluation breakdown plot generated successfully in plots/evaluation_task_breakdown.png")
