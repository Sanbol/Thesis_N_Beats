"""Generate N-HiTS-S Final Test Results table as a presentation-ready PNG image."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

# Title
fig.text(0.5, 0.95, 'Final Test Results — N-HiTS-S', ha='center', va='top',
         fontsize=20, fontweight='bold', color='#1a2a3a')
fig.text(0.5, 0.90, 'Architecture: 512 Hidden Units / 10 Blocks (M4 grid search)  •  3 seeds per condition',
         ha='center', va='top', fontsize=11, color='#555555')

# Table data
col_labels = ['Paradigm', 'Variant', 'sMAPE', 'RMSSE', 'RMSSC', 'sMAPC']

rows = [
    ['Scratch',            'Standard',        '13.05 ± 0.08', '1.33 ± 0.04', '0.46 ± 0.04', '5.74 ± 0.62'],
    ['',                   'Stabilized',       '13.26 ± 0.32', '1.33 ± 0.06', '0.34 ± 0.03', '4.48 ± 0.51'],
    ['',                   'Δ (Stab − Std)',   '+0.21 (+1.6%)', '−0.00 (−0.2%)', '−0.12 (−25.9%)', '−1.25 (−21.8%)'],
    ['Transfer Learning',  'Standard',        '13.30 ± 0.19', '1.21 ± 0.01', '0.52 ± 0.02', '7.78 ± 0.51'],
    ['',                   'Stabilized',       '13.12 ± 0.17', '1.22 ± 0.01', '0.48 ± 0.02', '6.89 ± 0.36'],
    ['',                   'Δ (Stab − Std)',   '−0.18 (−1.3%)', '+0.01 (+0.8%)', '−0.05 (−8.7%)', '−0.89 (−11.4%)'],
    ['Zero-Shot',          'Standard',        '13.99 ± 0.12', '1.33 ± 0.06', '0.54 ± 0.02', '7.52 ± 0.47'],
    ['',                   'Stabilized',       '13.41 ± 0.09', '1.32 ± 0.02', '0.42 ± 0.03', '5.84 ± 0.64'],
    ['',                   'Δ (Stab − Std)',   '−0.58 (−4.1%)', '−0.00 (−0.4%)', '−0.12 (−22.2%)', '−1.68 (−22.4%)'],
]

# Colors
header_color = '#2c5f7c'
std_row_color = '#fdf2e9'
stab_row_color = '#eafaf1'
delta_row_color = '#fef9e7'

row_colors = [
    std_row_color, stab_row_color, delta_row_color,
    std_row_color, stab_row_color, delta_row_color,
    std_row_color, stab_row_color, delta_row_color,
]

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.02, 0.05, 0.96, 0.78]
)

table.auto_set_font_size(False)
table.set_fontsize(11)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor(header_color)
    cell.set_text_props(color='white', fontweight='bold', fontsize=12)
    cell.set_edgecolor('#1a2a3a')
    cell.set_linewidth(1.2)

# Style data rows
for i in range(len(rows)):
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(row_colors[i])
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.8)

        text = rows[i][j]

        # Bold the paradigm names
        if j == 0 and text:
            cell.set_text_props(fontweight='bold', fontsize=11)

        # Italic delta rows
        if 'Δ' in rows[i][1]:
            cell.set_text_props(fontstyle='italic', fontsize=10, color='#444444')

        # Bold best RMSSC values (stabilized rows)
        if j == 4 and rows[i][1] == 'Stabilized':
            cell.set_text_props(fontweight='bold', fontsize=11)

        cell.set_height(0.085)

# Set column widths
col_widths = [0.16, 0.14, 0.14, 0.14, 0.14, 0.14]
for j, w in enumerate(col_widths):
    for i in range(len(rows) + 1):
        table[i, j].set_width(w)

plt.savefig('thesis_figures_v2/nhits_results_table.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.3)
plt.close()
print("Saved: thesis_figures_v2/nhits_results_table.png")
