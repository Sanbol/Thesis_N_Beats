"""
Reads loss_curve_*_Stabilized*.csv (committed W&B exports) and produces
fig_loss_decomposition.png:
2x3 grid — Stabilized variants only: Accuracy loss, Stability loss, Total loss per epoch.
Row 0: NBEATS (Scratch, TL Pretrain, TL Finetune).
Row 1: NHITS  (Scratch, TL Pretrain, TL Finetune).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
OUT = BASE / "fig_loss_decomposition.png"

PANELS = [
    # (row, col, title, csv_name)
    (0, 0, "NBEATS Scratch Stabilized",         "loss_curve_NBEATS_Scratch_Stabilized.csv"),
    (0, 1, "NBEATS TL Stabilized Pretrain",      "loss_curve_NBEATS_TL_Stabilized_Pretrain.csv"),
    (0, 2, "NBEATS TL Stabilized Finetune",      "loss_curve_NBEATS_TL_Stabilized_Finetune.csv"),
    (1, 0, "NHITS Scratch Stabilized",           "loss_curve_NHITS_Scratch_Stabilized.csv"),
    (1, 1, "NHITS TL Stabilized Pretrain",       "loss_curve_NHITS_TL_Stabilized_Pretrain.csv"),
    (1, 2, "NHITS TL Stabilized Finetune",       "loss_curve_NHITS_TL_Stabilized_Finetune.csv"),
]


def load_decomposition(csv_path):
    epochs, acc, stab, total = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row.get("tloss_epoch", "").strip():
                continue
            epochs.append(int(float(row["epoch"])))
            acc.append(float(row["tloss_a_epoch"]))
            stab.append(float(row["tloss_s_epoch"]))
            total.append(float(row["tloss_epoch"]))
    return epochs, acc, stab, total


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Stabilized loss decomposition (accuracy + stability)", fontsize=12, y=1.01)

for row, col, title, csv_name in PANELS:
    ax = axes[row][col]
    epochs, acc, stab, total = load_decomposition(BASE / csv_name)
    ax.plot(epochs, acc,   color="#1f77b4", linewidth=1.2, label="Accuracy loss")
    ax.plot(epochs, stab,  color="#d62728", linewidth=1.2, label="Stability loss")
    ax.plot(epochs, total, color="black",   linewidth=1.2, linestyle="--", label="Total loss")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT), dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
