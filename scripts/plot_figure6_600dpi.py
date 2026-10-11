"""
Generate a 100% mathematically exact, perfectly symmetrical 600 DPI publication heatmap
of the ISO/IEC 25010 Pairwise Coupling Matrix J_ij for Figure 6 without any top/bottom caption text.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set font and style for publication
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 600
})

classes = ["F", "SE", "US", "O", "PE", "LF", "A", "MN", "SC", "FT", "L", "PO"]
n = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}

# Initialize symmetrical matrix
J = np.zeros((n, n), dtype=float)

# Exact trade-off penalties from AlphaReq ConflictDetector
penalties = {
    ('PE', 'SE'): -0.75,
    ('SE', 'US'): -0.65,
    ('PE', 'FT'): -0.45,
    ('SC', 'PE'): -0.40,
    ('SE', 'O'):  -0.35,
    ('MN', 'PE'): -0.30,
    ('LF', 'PE'): -0.25,
    ('FT', 'US'): -0.20,
}

# Exact cooperation bonuses from AlphaReq ConflictDetector
bonuses = {
    ('A', 'FT'):  0.60,
    ('LF', 'US'): 0.55,
    ('SE', 'L'):  0.55,
    ('SC', 'A'):  0.50,
    ('F', 'O'):   0.45,
}

for (t1, t2), val in penalties.items():
    i, j = class_to_idx[t1], class_to_idx[t2]
    J[i, j] = val
    J[j, i] = val

for (t1, t2), val in bonuses.items():
    i, j = class_to_idx[t1], class_to_idx[t2]
    J[i, j] = val
    J[j, i] = val

# Create figure
fig, ax = plt.subplots(figsize=(10, 8.5), dpi=600)

# Custom Diverging Colormap (Red - White - Green)
cmap = sns.diverging_palette(10, 135, s=90, l=60, as_cmap=True)

# Heatmap
sns.heatmap(
    J,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-0.80,
    vmax=0.80,
    center=0.00,
    square=True,
    linewidths=1.0,
    linecolor='#e0e0e0',
    cbar_kws={
        'label': 'Coupling Strength ($J_{ij}$)',
        'shrink': 0.85,
        'ticks': [-0.80, -0.60, -0.40, -0.20, 0.00, 0.20, 0.40, 0.60, 0.80]
    },
    xticklabels=classes,
    yticklabels=classes,
    ax=ax
)

# Axis labels
ax.set_xlabel('ISO/IEC 25010 Requirement Quality Attribute Class', fontweight='bold', labelpad=10)
ax.set_ylabel('ISO/IEC 25010 Requirement Quality Attribute Class', fontweight='bold', labelpad=10)

plt.tight_layout()

# Save at 600 DPI to results/plots
out_dir = r"C:\Users\umert\.gemini\antigravity\scratch\AlphaReq\results\plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "figure6_conflict_heatmap.png")
plt.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"SUCCESS: Saved 600 DPI perfectly symmetrical Figure 6 to {out_path}")
