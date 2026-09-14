import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

NAVY = '#003366'
ACCENT = '#0066cc'
ROAD_GRAY = '#3a404a'
MARKING = '#ffffff'
EGO_GREEN = '#27ae60'
OTHER_RED = '#e74c3c'
FAIL_RED = '#c0392b'
SUCCESS_GREEN = '#2e7d32'

def draw_4way(ax, cx, cy, size=0.8):
    hw = size / 2.0
    rw = size * 0.36
    ax.add_patch(patches.Rectangle((cx - hw, cy - rw/2), size, rw, color=ROAD_GRAY, zorder=1))
    ax.add_patch(patches.Rectangle((cx - rw/2, cy - hw), rw, size, color=ROAD_GRAY, zorder=1))
    ax.plot([cx - hw, cx - rw/4], [cy, cy], color=MARKING, linestyle='--', lw=1, zorder=2)
    ax.plot([cx + rw/4, cx + hw], [cy, cy], color=MARKING, linestyle='--', lw=1, zorder=2)
    ax.plot([cx, cx], [cy - hw, cy - rw/4], color=MARKING, linestyle='--', lw=1, zorder=2)
    ax.plot([cx, cx], [cy + rw/4, cy + hw], color=MARKING, linestyle='--', lw=1, zorder=2)
    ax.plot(cx, cy - rw/4, marker='s', color=EGO_GREEN, markersize=5, zorder=3)
    ax.plot(cx + rw/4, cy, marker='o', color=OTHER_RED, markersize=5, zorder=3)

def draw_tjunction(ax, cx, cy, size=0.8):
    hw = size / 2.0
    rw = size * 0.36
    ax.add_patch(patches.Rectangle((cx - hw, cy - rw/2), size, rw, color=ROAD_GRAY, zorder=1))
    ax.add_patch(patches.Rectangle((cx - rw/2, cy - hw), rw, hw, color=ROAD_GRAY, zorder=1))
    ax.plot([cx - hw, cx + hw], [cy, cy], color=MARKING, linestyle='--', lw=1, zorder=2)
    ax.plot([cx, cx], [cy - hw, cy - rw/4], color=MARKING, linestyle='--', lw=1, zorder=2)
    ax.plot(cx, cy - rw/4, marker='s', color=EGO_GREEN, markersize=5, zorder=3)

def draw_roundabout(ax, cx, cy, size=0.8):
    hw = size / 2.0
    rw = size * 0.25
    r_outer = size * 0.38
    r_inner = size * 0.18
    ax.add_patch(patches.Rectangle((cx - hw, cy - rw/2), size, rw, color=ROAD_GRAY, zorder=1))
    ax.add_patch(patches.Rectangle((cx - rw/2, cy - hw), rw, size, color=ROAD_GRAY, zorder=1))
    ax.add_patch(patches.Circle((cx, cy), r_outer, color=ROAD_GRAY, zorder=2))
    ax.add_patch(patches.Circle((cx, cy), r_inner, color='white', zorder=3))
    ax.add_patch(patches.Circle((cx, cy), (r_outer+r_inner)/2, fill=False, edgecolor=MARKING, linestyle='--', lw=1, zorder=4))
    ax.plot(cx + (r_outer+r_inner)/2 * np.cos(np.pi*0.25), cy + (r_outer+r_inner)/2 * np.sin(np.pi*0.25), marker='s', color=EGO_GREEN, markersize=5, zorder=5)

# PANEL 1: Traditional RL
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.add_patch(patches.FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle='round,pad=0.1,rounding_size=0.3',
                                     facecolor='#fafbfc', edgecolor='#cbd5e1', lw=1.2))

ax1.text(5.0, 9.1, 'Traditional RL (Current Reproduction)', ha='center', va='center',
         fontsize=11.5, fontweight='bold', color=NAVY)
ax1.text(5.0, 8.45, 'Fixed Coordinate State: S = [x, y, v_x, v_y, ...]', ha='center', va='center',
         fontsize=9.5, color='#64748b', style='italic')

draw_4way(ax1, 2.5, 5.5, size=2.6)
ax1.text(2.5, 3.6, 'Training Environment\nFixed 4-Way Cross', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color=NAVY)
ax1.text(2.5, 2.4, 'High Success: 99.52%\n(Only on this exact layout)', ha='center', va='center',
         fontsize=8.5, color=NAVY, bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f0fe', edgecolor='#0066cc', lw=0.8))

ax1.annotate('', xy=(5.8, 6.7), xytext=(3.9, 6.0),
             arrowprops=dict(arrowstyle='-|>', color=FAIL_RED, lw=2.2, ls='--'))
ax1.annotate('', xy=(5.8, 4.3), xytext=(3.9, 5.0),
             arrowprops=dict(arrowstyle='-|>', color=FAIL_RED, lw=2.2, ls='--'))

draw_tjunction(ax1, 7.5, 6.8, size=1.8)
ax1.text(7.5, 5.4, 'T-Junction', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#333333')
ax1.text(8.8, 6.8, 'FAIL ✗', ha='center', va='center', fontsize=10, fontweight='bold', color=FAIL_RED)

draw_roundabout(ax1, 7.5, 3.6, size=1.8)
ax1.text(7.5, 2.2, 'Roundabout', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#333333')
ax1.text(8.8, 3.6, 'FAIL ✗', ha='center', va='center', fontsize=10, fontweight='bold', color=FAIL_RED)

ax1.text(5.0, 0.85, 'Current Semester Scope: Traditional baseline reproduced.\nModel is non-generalizable to new junction topologies.',
         ha='center', va='center', fontsize=8.5, color='#991b1b', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#ef4444', lw=0.8))

# PANEL 2: Generalizable Decision-Making
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.add_patch(patches.FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle='round,pad=0.1,rounding_size=0.3',
                                     facecolor='#f0fdf4', edgecolor='#16a34a', lw=1.5))

ax2.text(5.0, 9.1, 'Generalizable Decision-Making (Thesis Target)', ha='center', va='center',
         fontsize=11.5, fontweight='bold', color='#15803d')
ax2.text(5.0, 8.45, 'Invariant Relational / Conflict-Graph State Representation', ha='center', va='center',
         fontsize=9.5, color='#166534', style='italic')

ax2.add_patch(patches.FancyBboxPatch((0.6, 3.5), 3.4, 3.4, boxstyle='round,pad=0.1,rounding_size=0.2',
                                     facecolor='white', edgecolor=NAVY, lw=1.2))
ax2.text(2.3, 6.0, 'Unified Policy', ha='center', va='center', fontsize=10, fontweight='bold', color=NAVY)
ax2.text(2.3, 5.2, r'$\pi_	heta(a \mid s_{rel})$', ha='center', va='center', fontsize=10.5, fontweight='bold', color=ACCENT)
ax2.text(2.3, 4.3, 'Relative TTC\nConflict Graph\nTopology-Agnostic', ha='center', va='center',
         fontsize=8, color='#444444')

ax2.annotate('', xy=(5.3, 7.5), xytext=(4.1, 5.8),
             arrowprops=dict(arrowstyle='-|>', color=SUCCESS_GREEN, lw=2.0))
ax2.annotate('', xy=(5.3, 5.3), xytext=(4.1, 5.2),
             arrowprops=dict(arrowstyle='-|>', color=SUCCESS_GREEN, lw=2.0))
ax2.annotate('', xy=(5.3, 3.1), xytext=(4.1, 4.6),
             arrowprops=dict(arrowstyle='-|>', color=SUCCESS_GREEN, lw=2.0))

draw_4way(ax2, 6.7, 7.5, size=1.5)
ax2.text(8.2, 7.5, '4-Way Cross\nTransfer ✓', ha='left', va='center', fontsize=8.5, fontweight='bold', color='#166534')

draw_tjunction(ax2, 6.7, 5.3, size=1.5)
ax2.text(8.2, 5.3, 'T-Junction\nTransfer ✓', ha='left', va='center', fontsize=8.5, fontweight='bold', color='#166534')

draw_roundabout(ax2, 6.7, 3.1, size=1.5)
ax2.text(8.2, 3.1, 'Roundabout\nTransfer ✓', ha='left', va='center', fontsize=8.5, fontweight='bold', color='#166534')

ax2.text(5.0, 0.85, 'Next Semester Goal: Zero-shot & few-shot transfer\nacross unseen road topologies without retraining.',
         ha='center', va='center', fontsize=8.5, color='#14532d', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#dcfce7', edgecolor='#22c55e', lw=0.8))

plt.tight_layout()
fig.savefig('PRESENTATION/img/generalizable_decision_concept.png', bbox_inches='tight')
plt.close(fig)
print('Saved PRESENTATION/img/generalizable_decision_concept.png')
