import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
})

fig = plt.figure(figsize=(12.2, 5.2))
gs = fig.add_gridspec(1, 2, wspace=0.12)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

NAVY = '#003366'
ACCENT = '#0066cc'
ROAD_GRAY = '#333b47'
MARKING = '#ffffff'
EGO_GREEN = '#2ecc71'
OTHER_RED = '#e74c3c'
FAIL_RED = '#c0392b'
SUCCESS_GREEN = '#15803d'

def draw_4way(ax, cx, cy, size=1.8):
    hw = size / 2.0
    rw = size * 0.38
    ax.add_patch(patches.Rectangle((cx - hw, cy - rw/2), size, rw, color=ROAD_GRAY, zorder=1))
    ax.add_patch(patches.Rectangle((cx - rw/2, cy - hw), rw, size, color=ROAD_GRAY, zorder=1))
    ax.plot([cx - hw, cx - rw/4], [cy, cy], color=MARKING, linestyle='--', lw=1.2, zorder=2)
    ax.plot([cx + rw/4, cx + hw], [cy, cy], color=MARKING, linestyle='--', lw=1.2, zorder=2)
    ax.plot([cx, cx], [cy - hw, cy - rw/4], color=MARKING, linestyle='--', lw=1.2, zorder=2)
    ax.plot([cx, cx], [cy + rw/4, cy + hw], color=MARKING, linestyle='--', lw=1.2, zorder=2)
    ax.plot(cx, cy - rw/4, marker='s', color=EGO_GREEN, markersize=5.5, zorder=3)
    ax.plot(cx + rw/4, cy, marker='o', color=OTHER_RED, markersize=5, zorder=3)

def draw_tjunction(ax, cx, cy, size=1.5):
    hw = size / 2.0
    rw = size * 0.38
    ax.add_patch(patches.Rectangle((cx - hw, cy - rw/2), size, rw, color=ROAD_GRAY, zorder=1))
    ax.add_patch(patches.Rectangle((cx - rw/2, cy - hw), rw, hw, color=ROAD_GRAY, zorder=1))
    ax.plot([cx - hw, cx + hw], [cy, cy], color=MARKING, linestyle='--', lw=1.2, zorder=2)
    ax.plot([cx, cx], [cy - hw, cy - rw/4], color=MARKING, linestyle='--', lw=1.2, zorder=2)
    ax.plot(cx, cy - rw/4, marker='s', color=EGO_GREEN, markersize=5.5, zorder=3)
    ax.plot(cx - rw/3, cy, marker='o', color=OTHER_RED, markersize=5, zorder=3)

def draw_roundabout(ax, cx, cy, size=1.5):
    hw = size / 2.0
    rw = size * 0.26
    r_outer = size * 0.40
    r_inner = size * 0.20
    ax.add_patch(patches.Rectangle((cx - hw, cy - rw/2), size, rw, color=ROAD_GRAY, zorder=1))
    ax.add_patch(patches.Rectangle((cx - rw/2, cy - hw), rw, size, color=ROAD_GRAY, zorder=1))
    ax.add_patch(patches.Circle((cx, cy), r_outer, color=ROAD_GRAY, zorder=2))
    ax.add_patch(patches.Circle((cx, cy), r_inner, color='white', zorder=3))
    ax.add_patch(patches.Circle((cx, cy), (r_outer+r_inner)/2, fill=False, edgecolor=MARKING, linestyle='--', lw=1.2, zorder=4))
    ax.plot(cx + (r_outer+r_inner)/2 * np.cos(np.pi*0.25), cy + (r_outer+r_inner)/2 * np.sin(np.pi*0.25), marker='s', color=EGO_GREEN, markersize=5.5, zorder=5)

# ----------------- PANEL 1: Traditional RL -----------------
ax1.set_xlim(-0.3, 10.3)
ax1.set_ylim(-0.3, 10.3)
ax1.axis('off')

ax1.add_patch(patches.FancyBboxPatch((0.0, 0.0), 10.0, 9.7, boxstyle='round,pad=0.0,rounding_size=0.4',
                                     facecolor='#fafbfc', edgecolor='#94a3b8', lw=1.5))

ax1.text(5.0, 9.15, 'Traditional RL Baseline', ha='center', va='center',
         fontsize=11.5, fontweight='bold', color=NAVY)
ax1.text(5.0, 8.68, '(Current Semester Reproduction)', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color='#475569')
ax1.text(5.0, 8.22, 'Fixed Coordinate State: S = [x, y, v_x, v_y, ...]', ha='center', va='center',
         fontsize=9, color='#64748b', style='italic')

draw_4way(ax1, 2.1, 5.2, size=2.2)
ax1.text(2.1, 3.65, 'Training Environment\nFixed 4-Way Junction', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color=NAVY)
ax1.text(2.1, 2.65, '99.52% Success\n(Single Geometry Only)', ha='center', va='center',
         fontsize=8.5, color=NAVY, bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f0fe', edgecolor='#0066cc', lw=0.8))

ax1.annotate('', xy=(4.8, 6.5), xytext=(3.4, 5.7),
             arrowprops=dict(arrowstyle='-|>', color=FAIL_RED, lw=2.2, ls='--'))
ax1.annotate('', xy=(4.8, 3.5), xytext=(3.4, 4.7),
             arrowprops=dict(arrowstyle='-|>', color=FAIL_RED, lw=2.2, ls='--'))

draw_tjunction(ax1, 5.7, 6.5, size=1.4)
ax1.text(5.7, 5.4, 'T-Junction', ha='center', va='center', fontsize=9, fontweight='bold', color='#333333')
ax1.text(6.8, 6.5, 'FAIL ✗\nCoord / Dimension\nMismatch', ha='left', va='center', fontsize=8, fontweight='bold', color=FAIL_RED,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#fee2e2', edgecolor='#ef4444', lw=0.6))

draw_roundabout(ax1, 5.7, 3.5, size=1.4)
ax1.text(5.7, 2.4, 'Roundabout', ha='center', va='center', fontsize=9, fontweight='bold', color='#333333')
ax1.text(6.8, 3.5, 'FAIL ✗\nTopology\nCollapse', ha='left', va='center', fontsize=8, fontweight='bold', color=FAIL_RED,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#fee2e2', edgecolor='#ef4444', lw=0.6))

ax1.text(5.0, 1.15, 'Current Scope: Traditional baseline reproduced.\nModel is non-generalizable to unseen road topologies.',
         ha='center', va='center', fontsize=8.2, color='#991b1b', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#ef4444', lw=0.9))

# ----------------- PANEL 2: Generalizable Decision-Making -----------------
ax2.set_xlim(-0.3, 10.3)
ax2.set_ylim(-0.3, 10.3)
ax2.axis('off')

ax2.add_patch(patches.FancyBboxPatch((0.0, 0.0), 10.0, 9.7, boxstyle='round,pad=0.0,rounding_size=0.4',
                                     facecolor='#f0fdf4', edgecolor='#16a34a', lw=1.6))

ax2.text(5.0, 9.15, 'Generalizable Decision-Making', ha='center', va='center',
         fontsize=11.5, fontweight='bold', color='#15803d')
ax2.text(5.0, 8.68, '(Next Semester Thesis Target)', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color='#166534')
ax2.text(5.0, 8.22, 'Invariant Relational / Conflict-Graph Representation', ha='center', va='center',
         fontsize=9, color='#166534', style='italic')

ax2.add_patch(patches.FancyBboxPatch((0.5, 3.1), 3.5, 3.6, boxstyle='round,pad=0.1,rounding_size=0.25',
                                     facecolor='white', edgecolor=NAVY, lw=1.3))
ax2.text(2.25, 5.95, 'Unified Policy', ha='center', va='center', fontsize=11, fontweight='bold', color=NAVY)
ax2.text(2.25, 5.05, '$\\pi_\\theta(a \\mid s_{\\mathrm{rel}})$', ha='center', va='center', fontsize=12, fontweight='bold', color=ACCENT)
ax2.text(2.25, 3.95, 'Relative TTC / Headway\nConflict Point Graph\nTopology-Agnostic', ha='center', va='center',
         fontsize=8.5, color='#475569')

ax2.annotate('', xy=(5.35, 6.8), xytext=(4.15, 5.65),
             arrowprops=dict(arrowstyle='-|>', color=SUCCESS_GREEN, lw=2.2))
ax2.annotate('', xy=(5.35, 4.9), xytext=(4.15, 4.9),
             arrowprops=dict(arrowstyle='-|>', color=SUCCESS_GREEN, lw=2.2))
ax2.annotate('', xy=(5.35, 3.0), xytext=(4.15, 4.15),
             arrowprops=dict(arrowstyle='-|>', color=SUCCESS_GREEN, lw=2.2))

draw_4way(ax2, 6.3, 6.8, size=1.4)
ax2.text(7.4, 6.8, '4-Way Cross\nTransfer ✓', ha='left', va='center', fontsize=9, fontweight='bold', color='#166534')

draw_tjunction(ax2, 6.3, 4.9, size=1.4)
ax2.text(7.4, 4.9, 'T-Junction\nTransfer ✓', ha='left', va='center', fontsize=9, fontweight='bold', color='#166534')

draw_roundabout(ax2, 6.3, 3.0, size=1.4)
ax2.text(7.4, 3.0, 'Roundabout\nTransfer ✓', ha='left', va='center', fontsize=9, fontweight='bold', color='#166534')

ax2.text(5.0, 1.15, 'Next Semester Goal: Zero-shot & few-shot policy transfer\nacross unseen road topologies without retraining.',
         ha='center', va='center', fontsize=8.2, color='#14532d', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#dcfce7', edgecolor='#22c55e', lw=0.9))

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generalizable_decision_concept.png')
fig.savefig(out_path, bbox_inches='tight')
plt.close(fig)
print('SUCCESS: generalizable_decision_concept.png generated')