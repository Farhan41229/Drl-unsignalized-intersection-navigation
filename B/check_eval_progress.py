import os

ep_csv = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/eval_output/episode_metrics.csv"
st_csv = "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B/eval_output/step_metrics.csv"

def get_last_ep(path):
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        try:
            f.seek(-1000, os.SEEK_END)
        except IOError:
            pass
        lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
        for l in reversed(lines):
            parts = l.split(",")
            if len(parts) >= 2 and parts[0].isdigit():
                return int(parts[0])
    return 0

ep = get_last_ep(st_csv)
print(f"Current Evaluated Episode: ~{ep} / ~4,500 total rollouts (approx {ep/4500*100:.1f}%)")
