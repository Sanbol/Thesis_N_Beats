"""Generate N-BEATS-S Final Test Results table as a presentation-ready PNG image."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

# Title
fig.text(0.5, 0.95, 'Final Test Results — N-BEATS-S', ha='center', va='top',
         fontsize=20, fontweight='bold', color='#1a2a3a')
fig.text(0.5, 0.90, 'Architecture: 256 Hidden Units / 20 Blocks (Van Belle et al., 2023)  •  3 seeds per condition',
         ha='center', va='top', fontsize=11, color='#555555')

# Table data
col_labels = ['Paradigm', 'Variant', 'sMAPE', 'RMSSE', 'RMSSC', 'sMAPC']

rows = [
    ['Scratch',            'Standard',        '12.80 ± 0.49', '1.38 ± 0.11', '0.51 ± 0.04', '5.59 ± 0.32'],
    ['',                   'Stabilized',       '12.49 ± 0.16', '1.24 ± 0.01', '0.30 ± 0.01', '3.80 ± 0.04'],
    ['',                   'Δ (Stab − Std)',   '−0.30 (−2.4%)', '−0.14 (−9.9%)', '−0.21 (−40.7%)', '−1.79 (−32.0%)'],
    ['Transfer Learning',  'Standard',        '12.35 ± 0.08', '1.17 ± 0.01', '0.40 ± 0.01', '5.55 ± 0.13'],
    ['',                   'Stabilized',       '12.30 ± 0.04', '1.17 ± 0.01', '0.38 ± 0.01', '5.30 ± 0.16'],
    ['',                   'Δ (Stab − Std)',   '−0.05 (−0.4%)', '+0.00 (+0.1%)', '−0.02 (−4.7%)', '−0.25 (−4.6%)'],
    ['Zero-Shot',          'Standard',        '12.77 ± 0.24', '1.25 ± 0.02', '0.47 ± 0.04', '6.02 ± 0.39'],
    ['',                   'Stabilized',       '12.44 ± 0.04', '1.27 ± 0.01', '0.33 ± 0.01', '4.60 ± 0.10'],
    ['',                   'Δ (Stab − Std)',   '−0.33 (−2.6%)', '+0.01 (+1.2%)', '−0.13 (−28.5%)', '−1.43 (−23.7%)'],
]

# Colors
header_color = '#2c5f7c'
std_row_color = '#fdf2e9'      # light peach for standard
stab_row_color = '#eafaf1'     # light green for stabilized
delta_row_color = '#fef9e7'    # light yellow for delta
white = '#ffffff'

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

        # Set min height
        cell.set_height(0.085)

# Set column widths
col_widths = [0.16, 0.14, 0.14, 0.14, 0.14, 0.14]
for j, w in enumerate(col_widths):
    for i in range(len(rows) + 1):
        table[i, j].set_width(w)

plt.savefig('thesis_figures_v2/nbeats_results_table.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.3)
plt.close()
print("Saved: thesis_figures_v2/nbeats_results_table.png")
