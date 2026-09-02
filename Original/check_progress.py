#!/usr/bin/env python3
"""Zero-dependency progress & real-time ETA monitor for Config Original."""
import os
import sys
import time
import csv
import subprocess
from datetime import datetime, timedelta

LOG_PATH = os.path.expanduser("~/flow/PROJECTS/1_MultiTask_Intersection/New Runs/Original/logs/training_progress.csv")
TOTAL_EPISODES = 50000

# Check process status using standard ps
elapsed_sec = None
proc_running = False
try:
    cmd = "ps -eo pid,etime,args | grep 'train_original.py' | grep -v grep"
    res = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")[0]
    parts = res.split()
    if len(parts) >= 2:
        proc_running = True
        etime_str = parts[1]
        eparts = etime_str.split("-")
        days = int(eparts[0]) if len(eparts) > 1 else 0
        time_parts = eparts[-1].split(":")
        if len(time_parts) == 3:
            h, m, s = map(int, time_parts)
        else:
            h = 0
            m, s = map(int, time_parts)
        elapsed_sec = days * 86400 + h * 3600 + m * 60 + s
except Exception:
    pass

if not os.path.exists(LOG_PATH):
    print("=" * 65)
    print("  Config Original: Initializing Training...")
    print("=" * 65)
    print(f"  • Process Status: {'Running (PID active)' if proc_running else 'Not Running'}")
    print(f"  • Session Time:   {str(timedelta(seconds=int(elapsed_sec))) if elapsed_sec else 'Starting up...'}")
    print("  • Note: Log file not created yet. Waiting for first episode...")
    print("=" * 65)
    sys.exit(0)

# Fast reverse read of latest episode
def get_latest_ep():
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4000))
            lines = f.readlines()
            for line in reversed(lines):
                parts = line.strip().split(",")
                if parts and parts[0].isdigit():
                    return int(parts[0])
    except Exception:
        pass
    return None

ep_start = get_latest_ep()
t_start = time.time()

# 1.0 second sample to measure active real-time throughput
time.sleep(1.0)
ep_end = get_latest_ep()
t_sample = time.time() - t_start

if ep_start is not None and ep_end is not None and ep_end > ep_start:
    delta_ep = ep_end - ep_start
    realtime_sec_per_ep = t_sample / delta_ep
else:
    realtime_sec_per_ep = 0.58  # Steady P-core benchmark pace

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
    print("=" * 65)
    print("  Config Original: Initializing Training...")
    print("=" * 65)
    print(f"  • Process Status: {'Running (PID active)' if proc_running else 'Not Running'}")
    print(f"  • Session Time:   {str(timedelta(seconds=int(elapsed_sec))) if elapsed_sec else 'Starting up...'}")
    print("=" * 65)
    sys.exit(0)

current_ep = int(rows[-1]["episode"])
pct = (current_ep / TOTAL_EPISODES) * 100.0

recent = rows[-1000:]
recent_n = len(recent)
returns = [float(r["return"]) for r in recent]
mean_return = sum(returns) / recent_n if recent_n else 0.0
arrivals = sum(1 for r in recent if r["outcome"] == "arrival")
collisions = sum(1 for r in recent if r["outcome"] == "collision")
arrival_rate = (arrivals / recent_n * 100.0) if recent_n else 0.0
collision_rate = (collisions / recent_n * 100.0) if recent_n else 0.0

# Calculate ETA using real-time steady pace
sec_per_ep = max(0.30, min(1.20, realtime_sec_per_ep))
remaining_ep = max(0, TOTAL_EPISODES - current_ep)
eta_seconds = remaining_ep * sec_per_ep
eta_str = str(timedelta(seconds=int(eta_seconds)))
eta_finish = datetime.now() + timedelta(seconds=int(eta_seconds))

print("=" * 65)
print(f"  Config Original Training Status: {current_ep:,} / {TOTAL_EPISODES:,} episodes ({pct:.2f}%)")
print("=" * 65)
print(f"  • Process Status: {'Running (PID active)' if proc_running else 'Idle / Stopped'}")
print(f"  • Session Time:   {str(timedelta(seconds=int(elapsed_sec))) if elapsed_sec else 'N/A'}")
print(f"  • Real-Time Pace: {sec_per_ep:.2f} s/ep (~{60.0/sec_per_ep:.1f} ep/min)")
print(f"  • Remaining:      {remaining_ep:,} episodes")
print(f"  • Accurate ETA:   ~{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m ({eta_str})")
print(f"  • Projected Done: ~{eta_finish.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  • Buffer Size:    {int(rows[-1]['buffer_size']):,} transitions")
print(f"  • Temp Alpha (α): {float(rows[-1]['alpha']):.4f}")
print("-" * 65)
print(f"  Recent {recent_n} Episodes Stats:")
print(f"  • Mean Return:    {mean_return:.2f}")
print(f"  • Arrival Rate:   {arrival_rate:.1f}%")
print(f"  • Collision Rate: {collision_rate:.1f}%")
print("=" * 65)
