"""
Generates all chart figures used in the presentation from the project's
own reported results (README.md master guide, Projects/2 and Projects/3
milestone reports) and the Xiao et al. (2024) baseline numbers.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 13,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 200,
})

NAVY = "#1f3a5f"
TEAL = "#1f9e8f"
ORANGE = "#e8813a"
GRAY = "#8a8f98"
RED = "#c0392b"

# ---------------------------------------------------------------------
# 1. Success rate & collision rate: our Phase-2 reproduction vs paper
# ---------------------------------------------------------------------
labels = ["Project 2\n(PPO, Orig)", "Project 2\n(PPO, Config B)",
          "Project 3\n(Discrete SAC, ours)", "Xiao et al.\n(2024) — paper"]
success = [99.52, 94.9, 99.52, 97.4]
collision = [0.0, 0.0, 0.0, 2.6]

x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, success, w, label="Success rate (%)", color=NAVY)
b2 = ax.bar(x + w/2, collision, w, label="Collision rate (%)", color=RED)
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 112)
ax.set_title("Success & Collision Rate: Ours vs. Paper")
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        val_str = f"{h:.2f}" if (h > 0 and h != int(h) and round(h, 1) != h) else f"{h:.1f}"
        ax.annotate(val_str, (rect.get_x() + rect.get_width()/2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11)
ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.18))
fig.tight_layout()
fig.savefig("success_collision_chart.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# 2. Mean speed comparison across all milestones and the paper
# ---------------------------------------------------------------------
labels2 = ["Project 1\nRing (PPO)", "Project 2\nOrig (PPO)", "Project 2\nConfig B (PPO)",
           "Project 3\nOrig (SAC)", "Project 3\nConfig B (SAC)", "Paper\nOrig", "Paper\nConfig B"]
speeds = [5.2, 4.88, 4.83, 7.00, 6.75, 3.50, 3.80]
colors2 = [GRAY, TEAL, TEAL, NAVY, NAVY, ORANGE, ORANGE]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels2, speeds, color=colors2)
ax.set_ylabel("Mean speed (m/s)")
ax.set_title("Mean Crossing Speed Across Milestones vs. Paper")
for rect, v in zip(bars, speeds):
    ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width()/2, v),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11)
fig.tight_layout()
fig.savefig("speed_comparison_chart.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# 3. Intent-driven speed modulation: Config Original vs Config B (mu_a)
# ---------------------------------------------------------------------
tasks = ["Straight", "Left turn", "Right turn"]
orig_speed = [7.00, 7.00, 7.00]
cfgB_speed = [7.01, 6.91, 6.32]

x = np.arange(len(tasks))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, orig_speed, w, label=r"Config Original (no $\mu_a$)", color=GRAY)
ax.bar(x + w/2, cfgB_speed, w, label=r"Config B (with $\mu_a$)", color=TEAL)
ax.set_ylabel("Mean speed (m/s)")
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylim(0, 8.5)
ax.set_title("Task-Intent Feature Enables Maneuver-Aware Speed Modulation")
ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.15))
fig.tight_layout()
fig.savefig("intent_speed_modulation.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# 4. P-core thread affinity systems optimization
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

cats = ["Default\n(all cores)", "P-core affinity\n(0,2,4,6)"]
throughput = [46, 160]
axes[0].bar(cats, throughput, color=[GRAY, NAVY])
axes[0].set_ylabel("Episodes / minute")
axes[0].set_title("Training Throughput")
for i, v in enumerate(throughput):
    axes[0].annotate(f"{v}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center")

traintime = [18.0, 6.9]
axes[1].bar(cats, traintime, color=[GRAY, TEAL])
axes[1].set_ylabel("Wall-clock training time (hours)")
axes[1].set_title("50k-Episode Training Duration")
for i, v in enumerate(traintime):
    axes[1].annotate(f"{v}h", (i, v), textcoords="offset points", xytext=(0, 4), ha="center")

fig.suptitle("Hardware Scaling Fix: Intel Hybrid-CPU Thread Affinity", y=1.03)
fig.tight_layout()
fig.savefig("pcore_affinity_chart.png", bbox_inches="tight")
plt.close(fig)

print("All charts generated.")
