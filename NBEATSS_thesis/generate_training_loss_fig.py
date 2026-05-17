"""
Reads loss_curve_*.csv (committed W&B exports) and produces fig_training_loss.png:
2x3 grid — Standard (lambda=0) vs Stabilized (lambda=0.15) training loss per epoch.
Rows: Scratch (M3), TL Pretrain (M4), TL Finetune (M4->M3).
Columns: N-BEATS (left), N-HiTS (right).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
OUT = BASE / "fig_training_loss.png"

PANELS = [
    # (row, col, title, nbeats_std_csv, nbeats_stab_csv, nhits_std_csv, nhits_stab_csv)
    (0, 0, "N-BEATS — Scratch (M3)",
     "loss_curve_NBEATS_Scratch_Standard.csv",
     "loss_curve_NBEATS_Scratch_Stabilized.csv"),
    (0, 1, "N-HiTS — Scratch (M3)",
     "loss_curve_NHITS_Scratch_Standard.csv",
     "loss_curve_NHITS_Scratch_Stabilized.csv"),
    (1, 0, "N-BEATS — TL Pretrain (M4)",
     "loss_curve_NBEATS_TL_Standard_Pretrain.csv",
     "loss_curve_NBEATS_TL_Stabilized_Pretrain.csv"),
    (1, 1, "N-HiTS — TL Pretrain (M4)",
     "loss_curve_NHITS_TL_Standard_Pretrain.csv",
     "loss_curve_NHITS_TL_Stabilized_Pretrain.csv"),
    (2, 0, "N-BEATS — TL Finetune (M4→M3)",
     "loss_curve_NBEATS_TL_Standard_Finetune.csv",
     "loss_curve_NBEATS_TL_Stabilized_Finetune.csv"),
    (2, 1, "N-HiTS — TL Finetune (M4→M3)",
     "loss_curve_NHITS_TL_Standard_Finetune.csv",
     "loss_curve_NHITS_TL_Stabilized_Finetune.csv"),
]


def load_epoch_loss(csv_path, col="tloss_epoch"):
    epochs, losses = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            v = row.get(col, "").strip()
            if v:
                epochs.append(int(float(row["epoch"])))
                losses.append(float(v))
    return epochs, losses


fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Training loss curves — Standard vs Stabilized", fontsize=13, y=1.01)

for row, col, title, std_csv, stab_csv in PANELS:
    ax = axes[row][col]
    ep_std, lo_std = load_epoch_loss(BASE / std_csv)
    ep_stab, lo_stab = load_epoch_loss(BASE / stab_csv)
    ax.plot(ep_std, lo_std, color="#1f77b4", linewidth=1.2, label=r"Standard ($\lambda$=0)")
    ax.plot(ep_stab, lo_stab, color="#d62728", linewidth=1.2, label=r"Stabilized ($\lambda$=0.15)")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Training loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT), dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
