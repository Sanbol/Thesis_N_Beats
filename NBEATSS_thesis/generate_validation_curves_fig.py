"""
Reads val_curve_*_validation.csv (committed W&B exports) and produces
fig_validation_curves.png:
2x2 grid — N-BEATS Scratch Stabilized (left) and N-HiTS Scratch Stabilized (right).
Top row:    Training loss vs Validation loss (total).
Bottom row: Validation loss decomposition (accuracy, stability, total).
Title: "Training vs Validation Loss — Scratch Stabilized (155 epochs, eval_mode=validation)"
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
OUT = BASE / "fig_validation_curves.png"

MODELS = [
    ("N-BEATS Scratch Stabilized", "val_curve_NBEATS_Scratch_Stabilized_validation.csv"),
    ("N-HiTS Scratch Stabilized",  "val_curve_NHITS_Scratch_Stabilized_validation.csv"),
]


def load_val_curve(csv_path):
    train_ep, train_loss = [], []
    val_ep, val_total, val_acc, val_stab = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("tloss_epoch", "").strip():
                train_ep.append(int(float(row["epoch"])))
                train_loss.append(float(row["tloss_epoch"]))
            if row.get("vloss", "").strip():
                val_ep.append(int(float(row["epoch"])))
                val_total.append(float(row["vloss"]))
                val_acc.append(float(row["vloss_a"]))
                val_stab.append(float(row["vloss_s"]))
    return train_ep, train_loss, val_ep, val_total, val_acc, val_stab


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Training vs Validation Loss — Scratch Stabilized (155 epochs, eval_mode=validation)",
    fontsize=11, y=1.01
)

for col, (model_name, csv_name) in enumerate(MODELS):
    train_ep, train_loss, val_ep, val_total, val_acc, val_stab = load_val_curve(BASE / csv_name)

    # Top: training vs validation total loss
    ax = axes[0][col]
    ax.set_title(f"{model_name}\nTraining vs Validation", fontsize=9)
    ax.plot(train_ep, train_loss, color="#1f77b4", linewidth=1.2, label="Training loss")
    ax.plot(val_ep,   val_total,  color="#d62728", linewidth=1.2, label="Validation loss")
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Total loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom: validation decomposition
    ax = axes[1][col]
    ax.set_title(f"{model_name}\nValidation loss decomposition", fontsize=9)
    ax.plot(val_ep, val_acc,   color="#1f77b4", linewidth=1.2, label="Validation accuracy loss")
    ax.plot(val_ep, val_stab,  color="#d62728", linewidth=1.2, label="Validation stability loss")
    ax.plot(val_ep, val_total, color="black",   linewidth=1.2, linestyle="--", label="Validation total loss")
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Validation loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT), dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")
