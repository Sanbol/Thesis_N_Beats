"""
Generate HP Tuning Visualizations for Thesis Presentation
==========================================================
N-HiTS-S hyperparameter tuning on M4 validation set.
N-BEATS-S uses Van Belle et al. (2023) values directly (256/20).

Creates:
1. Heatmap: val_sMAPE for each (hidden_units, blocks) combo for N-HiTS-S
2. Scatter plot: Parameter count vs val_sMAPE for N-HiTS-S

Outputs to thesis_figures_v2/ folder.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "nhits_hp_tuning_m4_results.csv"
OUTPUT_DIR = BASE_DIR / "thesis_figures_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

# Read data (skip FAILED rows)
with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

rows = [r for r in all_rows if r.get('val_sMAPE') not in ('FAILED', '')]
failed_rows = [r for r in all_rows if r.get('val_sMAPE') == 'FAILED']

hidden_vals = [32, 128, 256, 512]
blocks_vals = [3, 5, 10]

# FIGURE 1: Heatmap (N-HiTS-S on M4)
fig, ax = plt.subplots(figsize=(8, 5.5))

# Build matrix (NaN for missing/failed)
smape_matrix = np.full((len(hidden_vals), len(blocks_vals)), np.nan)
for r in rows:
    h = int(r['hidden_layer_units'])
    b = int(r['n_blocks'])
    smape = float(r['val_sMAPE'])
    hi = hidden_vals.index(h)
    bi = blocks_vals.index(b)
    smape_matrix[hi, bi] = smape

# Find best (lowest sMAPE), ignoring NaN
valid_mask = ~np.isnan(smape_matrix)
if valid_mask.any():
    temp = smape_matrix.copy()
    temp[~valid_mask] = np.inf
    best_idx = np.unravel_index(np.argmin(temp), temp.shape)
else:
    best_idx = (0, 0)

vmin = np.nanmin(smape_matrix) - 0.05
vmax = np.nanmax(smape_matrix) + 0.05

im = ax.imshow(smape_matrix, cmap='RdYlGn_r', aspect='auto', vmin=vmin, vmax=vmax)

# Annotate cells
for i in range(len(hidden_vals)):
    for j in range(len(blocks_vals)):
        val = smape_matrix[i, j]
        if np.isnan(val):
            ax.text(j, i, 'FAILED', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='red')
        elif (i, j) == best_idx:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=13, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#2196F3', alpha=0.8))
        else:
            mid = (vmin + vmax) / 2
            color = 'white' if val > mid else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=12, fontweight='normal', color=color)

ax.set_xticks(range(len(blocks_vals)))
ax.set_xticklabels([str(b) for b in blocks_vals], fontsize=11)
ax.set_yticks(range(len(hidden_vals)))
ax.set_yticklabels([str(h) for h in hidden_vals], fontsize=11)
ax.set_xlabel('Number of Blocks', fontsize=12)
ax.set_ylabel('Hidden Units', fontsize=12)

best_h = hidden_vals[best_idx[0]]
best_b = blocks_vals[best_idx[1]]
ax.set_title(f'N-HiTS-S Hyperparameter Tuning (M4 Validation)\nBest: {best_h}/{best_b}',
            fontsize=13, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Validation sMAPE (%)', fontsize=11)

plt.savefig(OUTPUT_DIR / "hp_tuning_heatmap.png", dpi=200, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'hp_tuning_heatmap.png'}")


# FIGURE 2: Scatter plot (Params vs sMAPE, N-HiTS-S)
fig, ax = plt.subplots(figsize=(10, 6))

params = [int(r['n_parameters']) for r in rows]
smapes = [float(r['val_sMAPE']) for r in rows]

ax.scatter(params, smapes, c='#E64A19', marker='s',
          s=120, label='N-HiTS-S', alpha=0.8, edgecolors='black', linewidths=0.5, zorder=3)

# Annotate each point with h/b
for r in rows:
    p = int(r['n_parameters'])
    s = float(r['val_sMAPE'])
    h = r['hidden_layer_units']
    b = r['n_blocks']
    ax.annotate(f'{h}/{b}', (p, s),
               textcoords="offset points", xytext=(8, 4),
               fontsize=8, color='#E64A19', alpha=0.85)

# Highlight best
best_smape_idx = np.argmin(smapes)
best_p = params[best_smape_idx]
best_s = smapes[best_smape_idx]
best_r = rows[best_smape_idx]

ax.scatter([best_p], [best_s], c='#E64A19', marker='*',
          s=400, edgecolors='black', linewidths=1.5, zorder=5)
ax.annotate(f'BEST: {best_r["hidden_layer_units"]}/{best_r["n_blocks"]}',
           (best_p, best_s),
           textcoords="offset points", xytext=(12, -12),
           fontsize=10, fontweight='bold', color='#E64A19',
           arrowprops=dict(arrowstyle='->', color='#E64A19', lw=1.5))

ax.set_xscale('log')
ax.set_xlabel('Number of Parameters (log scale)', fontsize=12)
ax.set_ylabel('Validation sMAPE (%)', fontsize=12)
ax.set_title('N-HiTS-S Grid Search: Validation sMAPE as a Function of Model Capacity\n'
             '(M4 Monthly, single seed per configuration)',
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Add footnote about N-BEATS-S
fig.text(0.5, -0.04,
         'Note: N-BEATS-S hyperparameters (hidden = 256, blocks = 20) were adopted from '
         'Van Belle et al. (2023, Table 3)\nand therefore not subject to tuning.',
         ha='center', va='top', fontsize=9, fontstyle='italic', color='#555555')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hp_tuning_scatter.png", dpi=200, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'hp_tuning_scatter.png'}")


# Summary
print("\n" + "="*60)
print("N-HiTS-S HP TUNING ON M4 — SUMMARY:")
print("="*60)

baseline = next((r for r in rows if r['hidden_layer_units'] == '32' and r['n_blocks'] == '3'), None)

print(f"\n  Best config: {best_r['hidden_layer_units']} hidden / {best_r['n_blocks']} blocks")
print(f"  Parameters: {int(best_r['n_parameters']):,}")
print(f"  val_sMAPE: {float(best_r['val_sMAPE']):.2f}")
if baseline:
    print(f"  Baseline 32/3: {float(baseline['val_sMAPE']):.2f}")

if failed_rows:
    print(f"\n  Failed configs: {len(failed_rows)}")
    for r in failed_rows:
        print(f"    h={r['hidden_layer_units']} b={r['n_blocks']}")

print(f"\n  Note: N-BEATS-S uses Van Belle et al. (2023) values: hidden=256, blocks=20")
