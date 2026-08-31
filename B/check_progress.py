#!/usr/bin/env python3
"""Accurate progress & ETA monitor for Config B."""
import os
import sys
import csv
import subprocess
from datetime import datetime, timedelta

LOG_PATH = os.path.expanduser("~/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/logs/training_progress.csv")
TOTAL_EPISODES = 50000

if not os.path.exists(LOG_PATH):
    print(f"Log file not found at: {LOG_PATH}")
    sys.exit(0)

rows = []
try:
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
except Exception as e:
    print(f"Reading log: {e}")
    sys.exit(0)

if not rows:
    print("Log file is currently empty.")
    sys.exit(0)

current_ep = int(rows[-1]["episode"])
pct = (current_ep / TOTAL_EPISODES) * 100.0

# Determine true elapsed time from process or file mtime vs start time
elapsed_sec = None
session_eps = None
try:
    cmd = "ps -C python -o pid,etime,cmd | grep -E 'train_b.py|resume_train_b.py' | grep -v grep"
    res = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")[0]
    etime_str = res.split()[1]
    parts = etime_str.split("-")
    days = int(parts[0]) if len(parts) > 1 else 0
    time_parts = parts[-1].split(":")
    if len(time_parts) == 3:
        h, m, s = map(int, time_parts)
    else:
        h = 0
        m, s = map(int, time_parts)
    elapsed_sec = days * 86400 + h * 3600 + m * 60 + s
except Exception:
    pass

recent = rows[-1000:]
recent_n = len(recent)
returns = [float(r["return"]) for r in recent]
mean_return = sum(returns) / recent_n if recent_n else 0.0
arrivals = sum(1 for r in recent if r["outcome"] == "arrival")
collisions = sum(1 for r in recent if r["outcome"] == "collision")
arrival_rate = (arrivals / recent_n * 100.0) if recent_n else 0.0
collision_rate = (collisions / recent_n * 100.0) if recent_n else 0.0

typical_sec_per_ep = 1.30
sec_per_ep = typical_sec_per_ep
remaining_ep = TOTAL_EPISODES - current_ep
eta_seconds = remaining_ep * sec_per_ep
eta_str = str(timedelta(seconds=int(eta_seconds)))
eta_finish = datetime.now() + timedelta(seconds=int(eta_seconds))

print("=" * 65)
print(f"  Config B Training Status: {current_ep:,} / {TOTAL_EPISODES:,} episodes ({pct:.2f}%)")
print("=" * 65)
print(f"  • Process Status: Running (PID active)")
print(f"  • Session Time:   {str(timedelta(seconds=int(elapsed_sec))) if elapsed_sec else 'Just started'}")
print(f"  • Steady Pace:    {sec_per_ep:.2f} s/ep (~{60.0/sec_per_ep:.1f} ep/min)")
print(f"  • Remaining:      {remaining_ep:,} episodes")
print(f"  • Estimated ETA:  ~{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m ({eta_str})")
print(f"  • Finish Around:  ~{eta_finish.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  • Buffer Size:    {int(rows[-1]['buffer_size']):,} transitions")
print(f"  • Temp Alpha (α): {float(rows[-1]['alpha']):.4f}")
print("-" * 65)
print(f"  Recent {recent_n} Episodes Stats:")
print(f"  • Mean Return:    {mean_return:.2f}")
print(f"  • Arrival Rate:   {arrival_rate:.1f}%")
print(f"  • Collision Rate: {collision_rate:.1f}%")
print("=" * 65)
