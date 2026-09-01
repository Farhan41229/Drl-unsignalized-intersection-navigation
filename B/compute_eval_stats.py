import os
import sys
import pandas as pd
import numpy as np

eval_dir = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/eval_output"
ep_csv = os.path.join(eval_dir, "episode_metrics.csv")

if not os.path.exists(ep_csv) or os.path.getsize(ep_csv) == 0:
    print("episode_metrics.csv is not yet flushed/written.")
    sys.exit(0)

df = pd.read_csv(ep_csv)
print(f"Total Evaluated Episodes in CSV: {len(df)}")
tasks = ['left', 'straight', 'right']
print("\n" + "=" * 75)
print(f"{'Task':<10} | {'Count':<6} | {'Arrival %':<10} | {'Collision %':<12} | {'Avg Return':<12} | {'Avg Speed (m/s)':<15}")
print("-" * 75)
for t in tasks:
    sub = df[df['task'] == t]
    n = len(sub)
    if n > 0:
        arr_pct = (sub['outcome'] == 'arrival').mean() * 100.0
        col_pct = (sub['outcome'] == 'collision').mean() * 100.0
        ret_mean = sub['total_return'].mean()
        spd_mean = sub['avg_speed'].mean()
        print(f"{t:<10} | {n:<6} | {arr_pct:6.2f}%    | {col_pct:6.2f}%      | {ret_mean:8.2f}     | {spd_mean:6.2f}")
    else:
        print(f"{t:<10} | {0:<6} | N/A        | N/A          | N/A          | N/A")
print("=" * 75)
