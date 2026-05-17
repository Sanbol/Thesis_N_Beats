"""
Reads mcb_results.csv (committed) and produces fig_mcb_critical_difference.png:
2x2 grid of MCB Critical Difference plots for sMAPE, RMSSE, sMAPC, RMSSC.
Methods sorted by mean_rank (best = lowest rank, plotted at top).
Red: tied with best (within CD). Blue: not tied.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path(__file__).parent
OUT = BASE / "fig_mcb_critical_difference.png"
MCB_CSV = BASE / "mcb_results.csv"

METRICS = ["sMAPE", "RMSSE", "sMAPC", "RMSSC"]
GRID = [(0, 0), (0, 1), (1, 0), (1, 1)]


def load_mcb(metric):
    rows = []
    with open(MCB_CSV) as f:
        for r in csv.DictReader(f):
            if r["metric"] == metric:
                rows.append({
                    "method": r["method"],
                    "mean_rank": float(r["mean_rank"]),
                    "diff_from_best": float(r["diff_from_best"]),
                    "tied_with_best": r["tied_with_best"].strip().lower() == "true",
                    "CD": float(r["CD"]),
                    "N": int(r["N"]),
                    "k": int(r["k"]),
                })
    rows.sort(key=lambda x: x["mean_rank"])
    return rows


def plot_cd(ax, rows, metric):
    cd = rows[0]["CD"]
    n = rows[0]["N"]
    k = rows[0]["k"]
    best_rank = rows[0]["mean_rank"]

    ax.set_title(f"{metric}  (CD = {cd:.4f}, α = 0.05, N = {n:,}, k = {k})", fontsize=8.5)
    ax.set_xlabel("Mean rank (lower is better)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    color_tied = "#d62728"
    color_other = "#1f77b4"

    y_positions = list(range(len(rows) - 1, -1, -1))
    for i, row in enumerate(rows):
        y = y_positions[i]
        color = color_tied if row["tied_with_best"] else color_other
        ax.plot(row["mean_rank"], y, "o", color=color, markersize=6, zorder=3)
        ax.plot([row["mean_rank"] - cd / 2, row["mean_rank"] + cd / 2],
                [y, y], "-", color=color, linewidth=1.5, alpha=0.7)
        ax.text(row["mean_rank"], y + 0.35, row["method"],
                ha="center", va="bottom", fontsize=6.5)

    ax.axvline(best_rank, color="green", linewidth=1, linestyle="--", alpha=0.6,
               label=f"Best = {best_rank:.2f}")
    ax.axvline(best_rank + cd, color="orange", linewidth=1, linestyle="--", alpha=0.6,
               label=f"Best + CD = {best_rank + cd:.2f}")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    margin = cd * 1.5
    all_ranks = [r["mean_rank"] for r in rows]
    ax.set_xlim(min(all_ranks) - margin, max(all_ranks) + margin)
    ax.set_ylim(-1, len(rows) + 0.5)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MCB Critical Difference Plots — All Methods (12 conditions)", fontsize=12)

for metric, (r, c) in zip(METRICS, GRID):
    rows = load_mcb(metric)
    plot_cd(axes[r][c], rows, metric)

plt.tight_layout()
plt.savefig(str(OUT), dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
