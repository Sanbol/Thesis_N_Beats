"""
Generate thesis-quality figures from the TUNED experiment results.
Reads: experiment_results_tuned.csv (N-BEATS-S 256/20)
       nhits_experiment_results_tuned.csv (N-HiTS-S 512/10)
       lambda_sensitivity_results.csv (lambda sensitivity)

Outputs to thesis_figures_tuned/:
  1. results_table.png              — Main results table (mean +/- std)
  2. smape_bar_chart.png            — sMAPE grouped bar chart
  3. rmssc_bar_chart.png            — RMSSC grouped bar chart
  4. stabilization_delta_smape.png  — sMAPE improvement from stabilization
  5. stabilization_delta_rmssc.png  — RMSSC improvement from stabilization
  6. lambda_smapc.png               — Lambda vs sMAPC convergence plot
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "thesis_figures_tuned")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load experiment data ──────────────────────────────────────────────────────
df_nb = pd.read_csv(os.path.join(BASE, "experiment_results_tuned.csv"))
df_nh = pd.read_csv(os.path.join(BASE, "nhits_experiment_results_tuned.csv"))

# Filter to final test conditions (remove pretrain rows)
df_nb = df_nb[~df_nb["condition"].str.contains("pretrain")].copy()
df_nh = df_nh[~df_nh["condition"].str.contains("pretrain")].copy()

# Normalize condition names
df_nb["condition"] = df_nb["condition"].str.replace("NHITS_", "", regex=False)
df_nh["condition"] = df_nh["condition"].str.replace("NHITS_", "", regex=False)

df_nb["model"] = "N-BEATS-S"
df_nh["model"] = "N-HiTS-S"

df = pd.concat([df_nb, df_nh], ignore_index=True)

# Define display order
CONDITIONS = [
    "Scratch_Standard", "Scratch_Stabilized",
    "TL_Standard", "TL_Stabilized",
    "ZeroShot_Standard", "ZeroShot_Stabilized",
]
PARADIGMS = ["Scratch", "TL", "ZeroShot"]
PARADIGM_LABELS = {"Scratch": "Scratch", "TL": "Transfer Learning", "ZeroShot": "Zero-Shot"}
MODELS = ["N-BEATS-S", "N-HiTS-S"]

# Compute mean and std per (model, condition)
stats = df.groupby(["model", "condition"]).agg(
    sMAPE_mean=("sMAPE", "mean"),
    sMAPE_std=("sMAPE", "std"),
    RMSSC_mean=("RMSSC", "mean"),
    RMSSC_std=("RMSSC", "std"),
    RMSSE_mean=("RMSSE", "mean"),
    RMSSE_std=("RMSSE", "std"),
    sMAPC_mean=("sMAPC", "mean"),
    sMAPC_std=("sMAPC", "std"),
).reset_index()

# ── Load lambda sensitivity data ─────────────────────────────────────────────
df_lambda = pd.read_csv(os.path.join(BASE, "lambda_sensitivity_results.csv"))
lambda_stats = df_lambda.groupby("lambda").agg(
    sMAPC_mean=("sMAPC", "mean"),
    sMAPC_std=("sMAPC", "std"),
    sMAPE_mean=("sMAPE", "mean"),
    sMAPE_std=("sMAPE", "std"),
    RMSSC_mean=("RMSSC", "mean"),
    RMSSC_std=("RMSSC", "std"),
).reset_index().sort_values("lambda")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main Results Table (improved colors + readable title)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Results table...")

fig, ax = plt.subplots(figsize=(15, 7))
ax.axis('off')

# Build table data
col_headers = ["", "sMAPE (\u2193)", "RMSSE (\u2193)", "RMSSC (\u2193)", "sMAPC (\u2193)"]
table_data = []

for model in MODELS:
    table_data.append([model, "", "", "", ""])
    for paradigm in PARADIGMS:
        table_data.append([f"  {PARADIGM_LABELS[paradigm]}", "", "", "", ""])
        for variant in ["Standard", "Stabilized"]:
            cond = f"{paradigm}_{variant}"
            row = stats[(stats["model"] == model) & (stats["condition"] == cond)]
            if len(row) == 0:
                table_data.append([f"    {variant}", "\u2014", "\u2014", "\u2014", "\u2014"])
            else:
                r = row.iloc[0]
                table_data.append([
                    f"    {variant}",
                    f"{r['sMAPE_mean']:.2f} \u00b1 {r['sMAPE_std']:.2f}",
                    f"{r['RMSSE_mean']:.2f} \u00b1 {r['RMSSE_std']:.2f}",
                    f"{r['RMSSC_mean']:.2f} \u00b1 {r['RMSSC_std']:.2f}",
                    f"{r['sMAPC_mean']:.2f} \u00b1 {r['sMAPC_std']:.2f}",
                ])

table = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1.05, 1.55)

# Style: column header row
for j in range(len(col_headers)):
    cell = table[0, j]
    cell.set_facecolor('#1A5276')
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    cell.set_edgecolor('#D5D8DC')

# Style data rows
for i, row_data in enumerate(table_data, start=1):
    label = row_data[0]
    for j in range(len(col_headers)):
        cell = table[i, j]
        cell.set_edgecolor('#D5D8DC')

    if label in MODELS:
        for j in range(len(col_headers)):
            cell = table[i, j]
            cell.set_facecolor('#2E86C1')
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    elif label.strip() in PARADIGM_LABELS.values():
        for j in range(len(col_headers)):
            cell = table[i, j]
            cell.set_facecolor('#AED6F1')
            cell.set_text_props(fontweight='bold', color='#1A5276', fontsize=10)
    elif "Stabilized" in label:
        for j in range(len(col_headers)):
            table[i, j].set_facecolor('#EAF2F8')
    else:
        for j in range(len(col_headers)):
            table[i, j].set_facecolor('#FDFEFE')

# Title — larger, with more padding, black text on white background
fig.text(0.5, 0.96,
         "Experiment Results: Mean \u00b1 Std across 3 Seeds",
         ha='center', fontsize=14, fontweight='bold', color='#1A5276')
fig.text(0.5, 0.925,
         "N-BEATS-S (256/20, Van Belle et al.)  |  N-HiTS-S (512/10, M4-tuned)",
         ha='center', fontsize=11, color='#5D6D7E')

plt.savefig(os.path.join(OUT, "results_table.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  Saved: {os.path.join(OUT, 'results_table.png')}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: sMAPE Grouped Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: sMAPE bar chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

colors = {"Standard": "#E74C3C", "Stabilized": "#2980B9"}
x = np.arange(len(PARADIGMS))
width = 0.32

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    for vi, variant in enumerate(["Standard", "Stabilized"]):
        means = []
        stds = []
        for paradigm in PARADIGMS:
            cond = f"{paradigm}_{variant}"
            row = stats[(stats["model"] == model) & (stats["condition"] == cond)]
            means.append(row["sMAPE_mean"].values[0])
            stds.append(row["sMAPE_std"].values[0])

        offset = -width/2 + vi * width
        bars = ax.bar(x + offset, means, width * 0.9, yerr=stds,
                      label=variant, color=colors[variant], alpha=0.85,
                      edgecolor='white', linewidth=0.8,
                      capsize=4, error_kw=dict(lw=1.2))

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                    f'{m:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([PARADIGM_LABELS[p] for p in PARADIGMS], fontsize=11)
    ax.set_title(model, fontsize=13, fontweight='bold')
    ax.set_ylabel('sMAPE (%) \u2193' if idx == 0 else '', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    all_vals = [stats[(stats["model"] == model)]["sMAPE_mean"].min(),
                stats[(stats["model"] == model)]["sMAPE_mean"].max()]
    ax.set_ylim(min(all_vals) - 0.6, max(all_vals) + 0.8)

fig.suptitle("Forecast Accuracy (sMAPE) \u2014 Standard vs Stabilized",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "smape_bar_chart.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  Saved: {os.path.join(OUT, 'smape_bar_chart.png')}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: RMSSC Grouped Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: RMSSC bar chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    for vi, variant in enumerate(["Standard", "Stabilized"]):
        means = []
        stds = []
        for paradigm in PARADIGMS:
            cond = f"{paradigm}_{variant}"
            row = stats[(stats["model"] == model) & (stats["condition"] == cond)]
            means.append(row["RMSSC_mean"].values[0])
            stds.append(row["RMSSC_std"].values[0])

        offset = -width/2 + vi * width
        bars = ax.bar(x + offset, means, width * 0.9, yerr=stds,
                      label=variant, color=colors[variant], alpha=0.85,
                      edgecolor='white', linewidth=0.8,
                      capsize=4, error_kw=dict(lw=1.2))

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([PARADIGM_LABELS[p] for p in PARADIGMS], fontsize=11)
    ax.set_title(model, fontsize=13, fontweight='bold')
    ax.set_ylabel('RMSSC \u2193' if idx == 0 else '', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle("Forecast Stability (RMSSC) \u2014 Standard vs Stabilized",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "rmssc_bar_chart.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  Saved: {os.path.join(OUT, 'rmssc_bar_chart.png')}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4a: Stabilization Delta — sMAPE (separate, legend below)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4a: Stabilization delta (sMAPE)...")

model_colors = {"N-BEATS-S": "#2980B9", "N-HiTS-S": "#E67E22"}

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(PARADIGMS))
w = 0.32

for midx, model in enumerate(MODELS):
    deltas = []
    for paradigm in PARADIGMS:
        std_val = stats[(stats["model"] == model) & (stats["condition"] == f"{paradigm}_Standard")]["sMAPE_mean"].values[0]
        stab_val = stats[(stats["model"] == model) & (stats["condition"] == f"{paradigm}_Stabilized")]["sMAPE_mean"].values[0]
        deltas.append(std_val - stab_val)

    offset = -w/2 + midx * w
    bars = ax.bar(x_pos + offset, deltas, w * 0.9,
                  label=model, color=model_colors[model], alpha=0.85,
                  edgecolor='white', linewidth=0.8)

    for bar, d in zip(bars, deltas):
        va = 'bottom' if d >= 0 else 'top'
        yoff = 0.015 if d >= 0 else -0.015
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yoff,
                f'{d:+.2f}', ha='center', va=va, fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([PARADIGM_LABELS[p] for p in PARADIGMS], fontsize=11)
ax.set_ylabel('sMAPE Improvement (pp)', fontsize=12)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.grid(axis='y', alpha=0.3)
ax.set_title('sMAPE: Standard \u2212 Stabilized\n(positive = stabilization helps)',
             fontsize=12, fontweight='bold')

# Legend below the chart
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=2, framealpha=0.9, fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "stabilization_delta_smape.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  Saved: {os.path.join(OUT, 'stabilization_delta_smape.png')}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4b: Stabilization Delta — RMSSC (separate, legend below)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4b: Stabilization delta (RMSSC)...")

fig, ax = plt.subplots(figsize=(8, 5))

for midx, model in enumerate(MODELS):
    deltas = []
    for paradigm in PARADIGMS:
        std_val = stats[(stats["model"] == model) & (stats["condition"] == f"{paradigm}_Standard")]["RMSSC_mean"].values[0]
        stab_val = stats[(stats["model"] == model) & (stats["condition"] == f"{paradigm}_Stabilized")]["RMSSC_mean"].values[0]
        deltas.append(std_val - stab_val)

    offset = -w/2 + midx * w
    bars = ax.bar(x_pos + offset, deltas, w * 0.9,
                  label=model, color=model_colors[model], alpha=0.85,
                  edgecolor='white', linewidth=0.8)

    for bar, d in zip(bars, deltas):
        va = 'bottom' if d >= 0 else 'top'
        yoff = 0.003 if d >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yoff,
                f'{d:+.3f}', ha='center', va=va, fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([PARADIGM_LABELS[p] for p in PARADIGMS], fontsize=11)
ax.set_ylabel('RMSSC Improvement', fontsize=12)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.grid(axis='y', alpha=0.3)
ax.set_title('RMSSC: Standard \u2212 Stabilized\n(positive = stabilization helps)',
             fontsize=12, fontweight='bold')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=2, framealpha=0.9, fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "stabilization_delta_rmssc.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  Saved: {os.path.join(OUT, 'stabilization_delta_rmssc.png')}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Lambda vs sMAPC (convergence plot)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Lambda vs sMAPC...")

fig, ax1 = plt.subplots(figsize=(10, 6))

lambdas = lambda_stats["lambda"].values
smapc_mean = lambda_stats["sMAPC_mean"].values
smapc_std = lambda_stats["sMAPC_std"].values
smape_mean = lambda_stats["sMAPE_mean"].values
smape_std = lambda_stats["sMAPE_std"].values
rmssc_mean = lambda_stats["RMSSC_mean"].values
rmssc_std = lambda_stats["RMSSC_std"].values

# Primary axis: sMAPC
color_smapc = '#2980B9'
ax1.errorbar(lambdas, smapc_mean, yerr=smapc_std,
             marker='o', markersize=8, linewidth=2.2, color=color_smapc,
             capsize=5, capthick=1.5, label='sMAPC \u2193', zorder=3)
ax1.fill_between(lambdas, smapc_mean - smapc_std, smapc_mean + smapc_std,
                 alpha=0.15, color=color_smapc)
ax1.set_xlabel('\u03bb (Stabilization Strength)', fontsize=13)
ax1.set_ylabel('sMAPC (\u2193 = more stable changes)', fontsize=12, color=color_smapc)
ax1.tick_params(axis='y', labelcolor=color_smapc)

# Secondary axis: sMAPE
ax2 = ax1.twinx()
color_smape = '#E74C3C'
ax2.errorbar(lambdas, smape_mean, yerr=smape_std,
             marker='s', markersize=7, linewidth=2.2, color=color_smape,
             capsize=5, capthick=1.5, label='sMAPE \u2193', linestyle='--', zorder=2)
ax2.fill_between(lambdas, smape_mean - smape_std, smape_mean + smape_std,
                 alpha=0.1, color=color_smape)
ax2.set_ylabel('sMAPE (%) (\u2193 = more accurate)', fontsize=12, color=color_smape)
ax2.tick_params(axis='y', labelcolor=color_smape)

# Tertiary: RMSSC as scatter with size
for i, lam in enumerate(lambdas):
    ax1.scatter(lam, smapc_mean[i], s=rmssc_mean[i]*400, color=color_smapc,
                alpha=0.15, zorder=1)

# Mark the chosen lambda (0.02)
chosen_idx = np.where(np.isclose(lambdas, 0.02))[0]
if len(chosen_idx) > 0:
    ci = chosen_idx[0]
    ax1.axvline(x=0.02, color='#27AE60', linewidth=1.5, linestyle=':', alpha=0.7)
    ax1.annotate('\u03bb = 0.02\n(chosen)', xy=(0.02, smapc_mean[ci]),
                 xytext=(0.06, smapc_mean[ci] + 0.3),
                 fontsize=10, fontweight='bold', color='#27AE60',
                 arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))

ax1.set_title('Effect of \u03bb on Forecast Stability & Accuracy\n'
              'N-BEATS-S, Scratch, M3 Monthly (mean \u00b1 std, 3 seeds)',
              fontsize=13, fontweight='bold')

# Combined legend below
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper center', bbox_to_anchor=(0.5, -0.10),
           ncol=2, framealpha=0.9, fontsize=11)

ax1.grid(axis='both', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "lambda_smapc.png"), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  Saved: {os.path.join(OUT, 'lambda_smapc.png')}")


print("\n" + "="*60)
print("All figures generated in thesis_figures_tuned/")
print("="*60)
