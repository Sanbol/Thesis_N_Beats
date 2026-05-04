"""
Statistical significance tests for thesis results.

Reads per_series_results.csv (one row per series × condition, seed-averaged).

Outputs:
  friedman_results.csv    — Friedman χ² test across 4 models per (scenario, metric)
  wilcoxon_results.csv    — Paired Wilcoxon signed-rank Standard vs Stabilized
                            per (scenario, arch, metric), with Holm correction
  nemenyi_results.csv     — Nemenyi post-hoc pairwise p-values per (scenario, metric)
  cd_diagram_*.png        — Critical difference diagrams

Usage:
  python run_statistical_tests.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

df = pd.read_csv("per_series_results.csv")
print(f"Loaded per_series_results.csv: {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Scenarios: {sorted(df['scenario'].unique())}")
print(f"Models:    {sorted(df['model'].unique())}")
print(f"Series:    {df['series_id'].nunique():,} unique\n")

METRICS  = ["sMAPE", "RMSSE", "sMAPC", "RMSSC"]
SCENARIOS = ["Scratch", "TL", "ZeroShot"]
MODELS    = ["NBEATS_Standard", "NBEATS_Stabilized", "NHITS_Standard", "NHITS_Stabilized"]
ALPHA     = 0.05

os.makedirs("stat_test_figures", exist_ok=True)

def make_pivot(scenario, metric):
    """Returns a DataFrame with series_id as index, one column per model."""
    sub = df[df["scenario"] == scenario][["series_id", "model", metric]]
    pivot = sub.pivot(index="series_id", columns="model", values=metric)
    # Keep only models that are present
    present = [m for m in MODELS if m in pivot.columns]
    pivot = pivot[present].dropna()
    return pivot


print("=" * 60)
print("FRIEDMAN TEST (4 models, all scenarios)")
print("=" * 60)

friedman_rows = []
for scenario in SCENARIOS:
    for metric in METRICS:
        pivot = make_pivot(scenario, metric)
        if pivot.shape[1] < 2:
            print(f"  SKIP {scenario}/{metric}: only {pivot.shape[1]} models present")
            continue
        # Friedman needs at least 3 groups; use all present columns
        groups = [pivot[col].values for col in pivot.columns]
        stat, p = stats.friedmanchisquare(*groups)
        sig = "**" if p < 0.01 else ("*" if p < ALPHA else "ns")
        print(f"  {scenario:10s} {metric:6s}  χ²={stat:.3f}  p={p:.4f}  {sig}")
        friedman_rows.append({
            "scenario": scenario, "metric": metric,
            "n_series": pivot.shape[0], "n_models": pivot.shape[1],
            "chi2": round(stat, 4), "p_value": round(p, 6),
            "significant": p < ALPHA
        })

pd.DataFrame(friedman_rows).to_csv("friedman_results.csv", index=False)
print("\nSaved friedman_results.csv\n")


print("=" * 60)
print("WILCOXON SIGNED-RANK: Standard vs Stabilized (paired)")
print("=" * 60)

wilcoxon_rows = []
for scenario in SCENARIOS:
    for arch in ["NBEATS", "NHITS"]:
        std_col  = f"{arch}_Standard"
        stab_col = f"{arch}_Stabilized"
        for metric in METRICS:
            pivot = make_pivot(scenario, metric)
            if std_col not in pivot.columns or stab_col not in pivot.columns:
                continue
            a = pivot[std_col].values
            b = pivot[stab_col].values
            # Remove any pairs with NaN
            mask = ~(np.isnan(a) | np.isnan(b))
            a, b = a[mask], b[mask]
            if len(a) < 10:
                print(f"  SKIP {scenario}/{arch}/{metric}: only {len(a)} pairs")
                continue
            stat, p = stats.wilcoxon(a, b, alternative="two-sided")
            # Median difference (Stabilized - Standard): negative = improvement
            med_diff = float(np.median(b - a))
            wilcoxon_rows.append({
                "scenario": scenario, "arch": arch, "metric": metric,
                "n_pairs": int(len(a)),
                "w_stat": round(stat, 4),
                "p_raw": round(p, 6),
                "median_diff_stab_minus_std": round(med_diff, 4),
                "direction": "better" if med_diff < 0 else "worse"
            })

# Holm–Bonferroni correction
wdf = pd.DataFrame(wilcoxon_rows)
if len(wdf) > 0:
    sorted_idx = wdf["p_raw"].argsort().values
    n = len(wdf)
    holm_p = np.ones(n)
    for rank, idx in enumerate(sorted_idx):
        holm_p[idx] = min(1.0, wdf.iloc[idx]["p_raw"] * (n - rank))
    # Enforce monotonicity (step-down)
    holm_p_sorted = holm_p[sorted_idx]
    for i in range(1, len(holm_p_sorted)):
        holm_p_sorted[i] = max(holm_p_sorted[i], holm_p_sorted[i-1])
    holm_p[sorted_idx] = holm_p_sorted
    wdf["p_holm"] = np.round(holm_p, 6)
    wdf["significant_holm"] = wdf["p_holm"] < ALPHA

    print(f"\n{'Scenario':<10} {'Arch':<7} {'Metric':<7} {'N':>5}  {'p_raw':>8}  {'p_holm':>8}  {'Med_Δ':>8}  {'Sig':>4}")
    print("-" * 72)
    for _, row in wdf.sort_values(["scenario","arch","metric"]).iterrows():
        sig = "✓" if row["significant_holm"] else "-"
        print(f"  {row['scenario']:<10} {row['arch']:<7} {row['metric']:<7} {int(row['n_pairs']):>5}  "
              f"{row['p_raw']:>8.4f}  {row['p_holm']:>8.4f}  {row['median_diff_stab_minus_std']:>+8.4f}  {sig:>4}")

    wdf.to_csv("wilcoxon_results.csv", index=False)
    print("\nSaved wilcoxon_results.csv\n")


print("=" * 60)
print("NEMENYI POST-HOC (where Friedman p < 0.05)")
print("=" * 60)

nemenyi_rows = []
for scenario in SCENARIOS:
    for metric in METRICS:
        # Only run post-hoc if Friedman was significant
        fr = pd.DataFrame(friedman_rows)
        fr_row = fr[(fr.scenario == scenario) & (fr.metric == metric)]
        if fr_row.empty or not fr_row.iloc[0]["significant"]:
            continue

        pivot = make_pivot(scenario, metric)
        if pivot.shape[1] < 3:
            continue

        nem = sp.posthoc_nemenyi_friedman(pivot.values)
        nem.index   = pivot.columns
        nem.columns = pivot.columns

        print(f"\n  {scenario} / {metric}  (Friedman p={fr_row.iloc[0]['p_value']:.4f})")
        for i, m1 in enumerate(pivot.columns):
            for j, m2 in enumerate(pivot.columns):
                if j <= i:
                    continue
                p_val = nem.loc[m1, m2]
                sig   = "**" if p_val < 0.01 else ("*" if p_val < ALPHA else "ns")
                print(f"    {m1} vs {m2}: p={p_val:.4f}  {sig}")
                nemenyi_rows.append({
                    "scenario": scenario, "metric": metric,
                    "model_a": m1, "model_b": m2,
                    "p_nemenyi": round(p_val, 6),
                    "significant": p_val < ALPHA
                })

if nemenyi_rows:
    pd.DataFrame(nemenyi_rows).to_csv("nemenyi_results.csv", index=False)
    print("\nSaved nemenyi_results.csv")
else:
    print("  No significant Friedman results → no Nemenyi output.")


print("\n" + "=" * 60)
print("CRITICAL DIFFERENCE DIAGRAMS")
print("=" * 60)

for scenario in SCENARIOS:
    for metric in METRICS:
        pivot = make_pivot(scenario, metric)
        if pivot.shape[1] < 3:
            continue

        # Average ranks across series (lower rank = better)
        ranks = pivot.rank(axis=1)
        avg_ranks = ranks.mean().sort_values()

        n  = pivot.shape[0]
        k  = pivot.shape[1]

        # Critical Difference (Nemenyi, α=0.05) per Demšar (2006)
        # q_alpha values for k=2..10 at alpha=0.05:
        q_table = {2:1.960, 3:2.343, 4:2.569, 5:2.728, 6:2.850,
                   7:2.949, 8:3.031, 9:3.102, 10:3.164}
        q = q_table.get(k, 3.164)
        CD = q * np.sqrt(k * (k + 1) / (6 * n))

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.set_title(f"Critical Difference Diagram\n{scenario} / {metric}  (N={n}, CD={CD:.3f})",
                     fontsize=11)
        ax.set_xlim(0.5, k + 0.5)
        ax.set_ylim(-1.5, 2)
        ax.axis("off")

        ax.plot([1, k], [0, 0], "k-", lw=1.5)
        for i in range(1, k + 1):
            ax.plot([i, i], [-0.1, 0.1], "k-", lw=1)
            ax.text(i, -0.35, str(i), ha="center", va="top", fontsize=8)
        ax.text((1 + k) / 2, -0.85, "Average Rank", ha="center", va="top", fontsize=9)

        colors = ["#4472C4", "#548235", "#7030A0", "#C00000"]
        for idx, (model, rank) in enumerate(avg_ranks.items()):
            y_pos = 1.0 + (idx % 2) * 0.5
            ax.plot([rank, rank], [0, y_pos], "k-", lw=0.8)
            ax.plot(rank, y_pos, "o", color=colors[idx % 4], markersize=8)
            ax.text(rank, y_pos + 0.15, f"{model}\n({rank:.2f})",
                    ha="center", va="bottom", fontsize=7.5)

        best_rank = avg_ranks.iloc[0]
        ax.annotate("", xy=(best_rank + CD, 1.5), xytext=(best_rank, 1.5),
                    arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
        ax.text(best_rank + CD / 2, 1.65, f"CD={CD:.2f}", ha="center", color="red", fontsize=8)

        fname = f"stat_test_figures/cd_diagram_{scenario}_{metric}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved {fname}")

print("\n" + "=" * 60)
print("SUMMARY: Wilcoxon results (Standard vs Stabilized, Holm-corrected)")
print("=" * 60)
if len(wdf) > 0:
    for scenario in SCENARIOS:
        print(f"\n  {scenario}:")
        sub = wdf[wdf.scenario == scenario].sort_values(["arch","metric"])
        for _, r in sub.iterrows():
            direction = "↓ better" if r["median_diff_stab_minus_std"] < 0 else "↑ worse"
            sig = "SIGNIFICANT" if r["significant_holm"] else "not sig."
            print(f"    {r['arch']}-S {r['metric']:6s}: Δ={r['median_diff_stab_minus_std']:+.4f} ({direction})"
                  f"  p_holm={r['p_holm']:.4f}  {sig}")

print("\nAll done.")
