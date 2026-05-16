"""
Training and Validation Loss Curve Generator for N-BEATS-S
Re-runs a few representative lambda values in VALIDATION mode so that
both training loss (per-step) and validation loss (per-epoch) are logged.

Uses a custom PyTorch Lightning callback to save per-step/per-epoch losses
to a CSV, then generates a 2-panel figure similar to Fig. 9 from the
original N-BEATS-S paper.

Panel 1: Training and Validation RMSSE for different lambdas
Panel 2: Validation RMSSE (zoomed) for different lambdas
"""

import subprocess
import os
import re
import csv
import time
import shutil
import sys
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
PYTHON_EXE = str(BASE_DIR.parent / ".venv" / "Scripts" / "python.exe")
MAIN_PY = BASE_DIR / "main_loss_curves.py"
MAIN_PY_TEMPLATE = BASE_DIR / "main.py"
LOSS_DATA_DIR = BASE_DIR / "loss_curve_data"
OUTPUT_PLOT = BASE_DIR / "training_validation_loss_curves.png"


LAMBDA_CONFIGS = [
    {"lambda": 0.0,  "ema_decay": 0.0,  "label": r"$\lambda$ = 0 (Standard N-BEATS)"},
    {"lambda": 0.02, "ema_decay": 0.99, "label": r"$\lambda$ = 0.02"},
    {"lambda": 0.10, "ema_decay": 0.99, "label": r"$\lambda$ = 0.10"},
    {"lambda": 0.20, "ema_decay": 0.99, "label": r"$\lambda$ = 0.20"},
]

SEED = 1
MAX_EPOCHS = 10
BATCHES_PER_EPOCH = 50

def build_main_script(config, seed, loss_csv_path):
    loss_csv_escaped = str(loss_csv_path).replace("\\", "\\\\")

    content = f'''
# Modified main.py for loss curve extraction
# Adds a custom callback that saves per-step training and per-epoch validation losses

import os
import sys

from src.methods.NBEATSS import LitNBEATSS
from src.utils.callbacks import LoadModelWarning, WriteForecastsToCSV, PlotTestPredictions

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
import torch
import csv

class LossCurveLogger(Callback):
    """Saves per-step training loss and per-epoch validation loss to CSV."""

    def __init__(self, output_path):
        super().__init__()
        self.output_path = output_path
        self.train_steps = []
        self.val_epochs = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        metrics = trainer.callback_metrics
        tloss_a = metrics.get("tloss_a_step", None)
        tloss_s = metrics.get("tloss_s_step", None)
        tloss = metrics.get("tloss_step", None)

        if tloss_a is not None:
            self.train_steps.append([
                step, trainer.current_epoch, batch_idx, "train",
                float(tloss_a),
                float(tloss_s) if tloss_s is not None else 0.0,
                float(tloss) if tloss is not None else float(tloss_a),
            ])

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        vloss_a = metrics.get("vloss_a", None)
        vloss_s = metrics.get("vloss_s", None)
        vloss = metrics.get("vloss", None)

        if vloss_a is not None:
            step = trainer.global_step
            self.val_epochs.append([
                step, trainer.current_epoch, "", "val",
                float(vloss_a),
                float(vloss_s) if vloss_s is not None else 0.0,
                float(vloss) if vloss is not None else float(vloss_a),
            ])

    def on_fit_end(self, trainer, pl_module):
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "global_step", "epoch", "batch_idx",
                             "loss_accuracy", "loss_stability", "loss_total"])
            for row in self.train_steps:
                writer.writerow(["train", row[0], row[1], row[2], row[4], row[5], row[6]])
            for row in self.val_epochs:
                writer.writerow(["val", row[0], row[1], row[2], row[4], row[5], row[6]])
        print(f"LOSS_CSV_SAVED: " + self.output_path)
        print(f"  Training steps logged: " + str(len(self.train_steps)))
        print(f"  Validation epochs logged: " + str(len(self.val_epochs)))


def main():
    wandb.login()
    project_name = "NBEATSS_thesis"

    dataset = "M3"
    subset = "Monthly"
    dataset_id = "M3M"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    load_model = False
    update_loaded_model_specific_training_and_eval_hparams = False

    if load_model:
        model_id = ""
        checkpoint = "last"
    else:
        backcast_length_multiplier = 8
        forecast_length = 6
        hidden_layer_units = 32
        n_blocks = 3
        n_blocks_shared = 1
        ensemble_size = 1
        zero_mean = True
        unit_variance = True

    eval_mode = "validation"
    random_seed = {seed}
    forecasting_origin_range_multiplier = 1e6
    batch_size = 32
    num_workers = 0
    if not load_model or update_loaded_model_specific_training_and_eval_hparams:
        lambda_stability = {config["lambda"]}
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = 1e-3
        explr_gamma = 1.0
        ema_decay = {config["ema_decay"]}
    max_norm = 1.0
    batches_per_epoch = {BATCHES_PER_EPOCH}
    patience = 1e6
    max_epochs = {MAX_EPOCHS}

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    save_forecasts = False
    plot_forecasts = False

    L.seed_everything(random_seed, workers=True)

    if load_model:
        path_to_checkpoint = project_name + "/" + model_id + "/checkpoints/" + checkpoint + ".ckpt"
        NBEATSS = LitNBEATSS.load_from_checkpoint(path_to_checkpoint)
        forecast_length = NBEATSS.hparams["forecast_length"]
        backcast_length_multiplier = NBEATSS.hparams["backcast_length_multiplier"]
        zero_mean = NBEATSS.hparams["zero_mean"]
        unit_variance = NBEATSS.hparams["unit_variance"]
        if update_loaded_model_specific_training_and_eval_hparams:
            NBEATSS.hparams["lambda_stability"] = lambda_stability
            NBEATSS.hparams["enforce_nonnegative_forecast_metric_calculation"] = enforce_nonnegative_forecast_metric_calculation
            NBEATSS.hparams["learning_rate"] = learning_rate
            NBEATSS.hparams["explr_gamma"] = explr_gamma
            NBEATSS.hparams["ema_decay"] = ema_decay
    else:
        NBEATSS = LitNBEATSS(
            backcast_length_multiplier=backcast_length_multiplier,
            forecast_length=forecast_length,
            hidden_layer_units=hidden_layer_units,
            n_blocks=n_blocks,
            n_blocks_shared=n_blocks_shared,
            ensemble_size=ensemble_size,
            zero_mean=zero_mean,
            unit_variance=unit_variance,
            lambda_stability=lambda_stability,
            enforce_nonnegative_forecast_metric_calculation=enforce_nonnegative_forecast_metric_calculation,
            learning_rate=learning_rate,
            explr_gamma=explr_gamma,
            ema_decay=ema_decay)

    if dataset == "M3":
        from src.data.M3 import load_data
    if dataset == "M4":
        from src.data.M4 import load_data

    train_dataloader, validation_dataloader, validation_dataloader_target, _, _ = load_data(
        subset=subset,
        test_mode_nrows=test_mode_nrows,
        backcast_length_multiplier=backcast_length_multiplier,
        forecast_length=forecast_length,
        validation_periods=validation_periods,
        test_periods=test_periods,
        zero_mean=zero_mean,
        unit_variance=unit_variance,
        forecasting_origin_range_multiplier=int(forecasting_origin_range_multiplier),
        batch_size=batch_size,
        num_workers=num_workers)

    wandb_logger = WandbLogger(project=project_name, log_model=False)
    modelsummary_callback = ModelSummary(max_depth=3)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="vloss", mode="min", save_last=True)
    early_stop_callback = EarlyStopping(monitor="vloss", mode="min", patience=int(patience))

    loss_csv_path = r"{loss_csv_escaped}"
    loss_curve_callback = LossCurveLogger(output_path=loss_csv_path)

    callbacks_list = [
        modelsummary_callback,
        checkpoint_callback,
        early_stop_callback,
        loss_curve_callback,
    ]

    trainer = Trainer(
        callbacks=callbacks_list,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=max_norm,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        max_epochs=max_epochs,
        limit_train_batches=batches_per_epoch,
    )

    print("Start model training and evaluation.")
    trainer.fit(NBEATSS, train_dataloader, validation_dataloader)

    if max_epochs == 0:
        trainer.test(NBEATSS, validation_dataloader_target, verbose=True)
    else:
        trainer.test(NBEATSS, validation_dataloader_target, "best", verbose=True)

    wandb_logger.experiment.config["dataset"] = dataset_id
    wandb_logger.experiment.config["random_seed"] = random_seed
    wandb_logger.experiment.config["origin_range"] = forecasting_origin_range_multiplier
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["patience"] = patience
    wandb_logger.experiment.config["max_epochs"] = max_epochs
    wandb_logger.experiment.config["max_norm"] = max_norm
    wandb_logger.experiment.config["eval_mode"] = eval_mode
    wandb_logger.experiment.config["n_batches"] = batches_per_epoch

    wandb.finish()


if __name__ == "__main__":
    main()
'''
    with open(MAIN_PY, 'w', encoding='utf-8') as f:
        f.write(content)


def run_experiment(config, seed):
    lam = config["lambda"]
    label_safe = f"lambda_{lam:.3f}_seed{seed}"
    loss_csv_path = LOSS_DATA_DIR / f"{label_safe}.csv"

    if loss_csv_path.exists():
        print(f"  [SKIP] {label_safe} - CSV already exists")
        return loss_csv_path

    build_main_script(config, seed, loss_csv_path)

    print(f"\n{'='*70}")
    print(f"  RUNNING: {label_safe}")
    print(f"  lambda={lam}, ema_decay={config['ema_decay']}, seed={seed}")
    print(f"{'='*70}")

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    result = subprocess.run(
        [PYTHON_EXE, str(MAIN_PY)],
        capture_output=True, text=True,
        cwd=str(BASE_DIR),
        env=env,
        encoding='utf-8', errors='replace',
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] Exit code {result.returncode}")
        print(f"  stderr: {(result.stderr or '')[:500]}")
        return None

    print(f"  Completed in {int(elapsed//60):02d}:{int(elapsed%60):02d}")

    if loss_csv_path.exists():
        print(f"  Loss CSV saved: {loss_csv_path}")
        return loss_csv_path
    else:
        print(f"  [WARNING] Loss CSV not found at {loss_csv_path}")
        return None


def load_loss_data(csv_path):
    train_steps = []
    val_epochs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "train":
                train_steps.append({
                    "step": int(row["global_step"]),
                    "epoch": int(row["epoch"]),
                    "loss_a": float(row["loss_accuracy"]),
                    "loss_s": float(row["loss_stability"]),
                    "loss": float(row["loss_total"]),
                })
            elif row["type"] == "val":
                val_epochs.append({
                    "step": int(row["global_step"]),
                    "epoch": int(row["epoch"]),
                    "loss_a": float(row["loss_accuracy"]),
                    "loss_s": float(row["loss_stability"]),
                    "loss": float(row["loss_total"]),
                })
    return train_steps, val_epochs


def moving_average(data, window=5):
    """Simple moving average for smoothing."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def generate_loss_curve_plots(results):
    """
    Generate 2-panel figure matching the style of Fig. 9 from the N-BEATS-S paper:
    Panel 1: Training and Validation RMSSE for different lambdas
    Panel 2: Validation stability loss for different lambdas
    """

    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes


    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("RMSSE", fontsize=12)

    for i, (config, train_data, val_data) in enumerate(results):
        color = colors[i % len(colors)]
        lam = config["lambda"]

        if train_data:
            steps = [d["step"] for d in train_data]
            losses = [d["loss_a"] for d in train_data]


            ax1.plot(steps, losses, color=color, alpha=0.12, linewidth=0.5)


            smoothed = moving_average(np.array(losses), window=15)
            offset = 14
            smoothed_steps = steps[offset:offset+len(smoothed)]
            ax1.plot(smoothed_steps, smoothed, color=color,
                     linewidth=2.0, linestyle='-',
                     label='Training RMSSE $\\lambda$ = %.2g' % lam)

        if val_data:
            v_steps = [d["step"] for d in val_data]
            v_losses = [d["loss_a"] for d in val_data]
            ax1.plot(v_steps, v_losses, color=color,
                     linewidth=2.0, linestyle='--', marker='o', markersize=5,
                     alpha=0.9,
                     label='Validation RMSSE $\\lambda$ = %.2g' % lam)

    ax1.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelsize=10)


    ax2.set_title("Validation Loss", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Iterations", fontsize=12)
    ax2.set_ylabel("RMSSE / Stability", fontsize=12)

    for i, (config, train_data, val_data) in enumerate(results):
        color = colors[i % len(colors)]
        lam = config["lambda"]

        if val_data:
            v_steps = [d["step"] for d in val_data]
            v_losses_a = [d["loss_a"] for d in val_data]
            v_losses_s = [d["loss_s"] for d in val_data]


            ax2.plot(v_steps, v_losses_a, color=color,
                     linewidth=2.5, linestyle='-', marker='s', markersize=5,
                     label='RMSSE $\\lambda$ = %.2g' % lam)


            ax2.plot(v_steps, v_losses_s, color=color,
                     linewidth=1.5, linestyle=':', marker='d', markersize=4,
                     alpha=0.7,
                     label='Stability $\\lambda$ = %.2g' % lam)

    ax2.legend(fontsize=7.5, loc='upper right', framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=10)

    plt.tight_layout(w_pad=3)
    plt.savefig(str(OUTPUT_PLOT), dpi=300, bbox_inches='tight')
    plt.close()


    output_val_only = str(OUTPUT_PLOT).replace('.png', '_validation_only.png')
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5))


    ax3.set_title("Validation RMSSE", fontsize=14, fontweight='bold', pad=12)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("RMSSE", fontsize=12)

    for i, (config, train_data, val_data) in enumerate(results):
        color = colors[i % len(colors)]
        lam = config["lambda"]

        if val_data:
            epochs = [d["epoch"] for d in val_data]
            v_losses_a = [d["loss_a"] for d in val_data]
            ax3.plot(epochs, v_losses_a, color=color,
                     linewidth=2.5, linestyle='-', marker='o', markersize=6,
                     label='$\\lambda$ = %.2g' % lam)

    ax3.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(10))
    ax3.tick_params(labelsize=10)


    ax4.set_title("Validation Stability Loss", fontsize=14, fontweight='bold', pad=12)
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Stability Loss", fontsize=12)

    for i, (config, train_data, val_data) in enumerate(results):
        color = colors[i % len(colors)]
        lam = config["lambda"]

        if val_data:
            epochs = [d["epoch"] for d in val_data]
            v_losses_s = [d["loss_s"] for d in val_data]
            ax4.plot(epochs, v_losses_s, color=color,
                     linewidth=2.5, linestyle='-', marker='s', markersize=6,
                     label='$\\lambda$ = %.2g' % lam)

    ax4.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(10))
    ax4.tick_params(labelsize=10)

    plt.tight_layout(w_pad=3)
    plt.savefig(output_val_only, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Loss curve plot saved: {OUTPUT_PLOT}")
    print(f"  Validation-only plot saved: {output_val_only}")


def main():
    print("=" * 80)
    print("  TRAINING & VALIDATION LOSS CURVES - Lambda Comparison")
    print("=" * 80)
    print(f"  Lambda values: {[c['lambda'] for c in LAMBDA_CONFIGS]}")
    print(f"  Seed: {SEED}")
    print(f"  Epochs: {MAX_EPOCHS}, Steps/epoch: {BATCHES_PER_EPOCH}")
    print(f"  Total steps per run: {MAX_EPOCHS * BATCHES_PER_EPOCH}")
    print(f"  Output: {OUTPUT_PLOT}")
    print()

    os.makedirs(LOSS_DATA_DIR, exist_ok=True)

    t_start = time.time()


    csv_paths = []
    for config in LAMBDA_CONFIGS:
        csv_path = run_experiment(config, SEED)
        csv_paths.append((config, csv_path))


    if MAIN_PY.exists():
        os.remove(MAIN_PY)

    elapsed_total = time.time() - t_start
    print(f"\n  Total experiment time: {int(elapsed_total//60):02d}:{int(elapsed_total%60):02d}")


    print("\n  Loading loss data and generating plots...")
    results = []
    for config, csv_path in csv_paths:
        if csv_path and csv_path.exists():
            train_data, val_data = load_loss_data(csv_path)
            results.append((config, train_data, val_data))
            print(f"  {config['label']}: {len(train_data)} train steps, {len(val_data)} val epochs")
        else:
            print(f"  [SKIP] {config['label']} - no data")

    if results:
        generate_loss_curve_plots(results)
    else:
        print("  [ERROR] No data to plot!")

    print("\n" + "#" * 70)
    print("  DONE - Loss Curve Generation Complete")
    print("#" * 70)


if __name__ == "__main__":
    main()
