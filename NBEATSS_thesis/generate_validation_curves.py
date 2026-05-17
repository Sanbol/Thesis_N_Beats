"""
Reads Validation_Loss_Curves.xlsx and produces two thesis figures:
  fig_validation_nbeats.png  — N-BEATS-S Scratch Stabilized
  fig_validation_nhits.png   — N-HiTS-S Scratch Stabilized

Each figure: Training loss (total) vs Validation loss (total) per epoch.
Source: sheets "N-BEATS curves" and "N-HiTS curves", header row 2, data rows 3–157.
"""
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
XLSX = BASE / "Validation_Loss_Curves.xlsx"

MODELS = [
    ("N-BEATS curves", "N-BEATS-S Scratch Stabilized", "fig_validation_nbeats.png"),
    ("N-HiTS curves",  "N-HiTS-S Scratch Stabilized",  "fig_validation_nhits.png"),
]


def load_sheet(sheet_name):
    df = pd.read_excel(XLSX, sheet_name=sheet_name, header=1, usecols=[0, 1, 2])
    df.columns = ["epoch", "train_loss", "val_loss"]
    df = df.dropna(subset=["epoch", "train_loss", "val_loss"])
    df["epoch"] = df["epoch"].astype(int)
    return df


for sheet, title, out_name in MODELS:
    df = load_sheet(sheet)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], color="#1f77b4", linewidth=1.5, label="Training loss")
    ax.plot(df["epoch"], df["val_loss"],   color="#d62728", linewidth=1.5, label="Validation loss")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Total loss", fontsize=10)
    ax.set_title(f"Training vs Validation Loss\n{title}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = BASE / out_name
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")
