"""
N-HiTS-S Hyperparameter Tuning on M4 (Validation-Based)
=========================================================
Tunes architecture hyperparameters for N-HiTS-S using the M4
VALIDATION set, following professor feedback.

N-BEATS-S hyperparameters are taken directly from Van Belle et al. (2023):
  hidden_layer_units = 256, n_blocks = 20

This script tunes N-HiTS-S only:
  - hidden_layer_units: [32, 128, 256, 512]
  - n_blocks: [3, 5, 10]

Conditions: Standard only (lambda=0.0, ema=0.0, lr=1e-3, max_epochs=10)
Uses eval_mode='validation' on M4 dataset.
Single seed (seed=1) for grid search.

Results saved to nhits_hp_tuning_m4_results.csv.
"""

import subprocess
import os
import re
import csv
import time
from pathlib import Path
from datetime import datetime
from itertools import product

# PATHS
BASE_DIR = Path(__file__).parent
PYTHON_EXE = str(BASE_DIR.parent / ".venv" / "Scripts" / "python.exe")
WANDB_DIR = BASE_DIR / "wandb"
RESULTS_FILE = BASE_DIR / "nhits_hp_tuning_m4_results.csv"

# TUNING GRID
HIDDEN_UNITS_GRID = [32, 128, 256, 512]
N_BLOCKS_GRID = [3, 5, 10]
SEED = 1  # Single seed for tuning

# N-HiTS-S TEMPLATE (M4 dataset)

NHITS_M4_TEMPLATE = '''# N-HiTS-S Hyperparameter Tuning Run on M4 (auto-generated)
import os
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
import torch

from src.methods.NHITSS import LitNHITSS

def main():
    wandb.login()
    project_name = "NBEATSS_thesis"

    # Dataset — M4 Monthly
    dataset = "M4"
    subset = "Monthly"
    dataset_id = "M4M"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    # Model architecture
    backcast_length_multiplier = 8
    forecast_length = 6
    hidden_layer_units = {hidden_layer_units}
    n_blocks = {n_blocks}
    n_blocks_shared = 1
    ensemble_size = 1
    zero_mean = True
    unit_variance = True

    # Training and evaluation
    eval_mode = 'validation'  # Use validation set for HP tuning
    random_seed = {seed}
    forecasting_origin_range_multiplier = 1e6
    batch_size = 32
    num_workers = 0
    lambda_stability = 0.0
    enforce_nonnegative_forecast_metric_calculation = True
    learning_rate = 1e-3
    explr_gamma = 1.0
    ema_decay = 0.0
    max_norm = 1.0
    batches_per_epoch = 50
    patience = 1e6
    max_epochs = 10

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    L.seed_everything(random_seed, workers=True)

    model = LitNHITSS(
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

    wandb_logger = WandbLogger(project=project_name, log_model=True)
    modelsummary_callback = ModelSummary(max_depth=3)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="vloss", mode="min", save_last=True)
    early_stop_callback = EarlyStopping(monitor="vloss", mode="min", patience=int(patience))
    
    trainer = Trainer(
        callbacks=[modelsummary_callback, checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices="auto",
        gradient_clip_val=max_norm,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        max_epochs=max_epochs,
        limit_train_batches=batches_per_epoch,
    )
    trainer.fit(model, train_dataloader, validation_dataloader)
    trainer.test(model, validation_dataloader_target, "best", verbose=True)

    wandb_logger.experiment.config["dataset"] = dataset_id
    wandb_logger.experiment.config["random_seed"] = random_seed
    wandb_logger.experiment.config["hidden_layer_units"] = hidden_layer_units
    wandb_logger.experiment.config["n_blocks"] = n_blocks
    wandb_logger.experiment.config["eval_mode"] = eval_mode
    wandb_logger.experiment.config["tuning_run"] = True

    wandb.finish()

if __name__ == '__main__':
    main()
'''


# HELPER FUNCTIONS

def get_existing_wandb_runs():
    if not WANDB_DIR.exists():
        return set()
    return set(d.name for d in WANDB_DIR.iterdir()
               if d.is_dir() and d.name.startswith("offline-run-"))


def find_new_run(existing_runs):
    current_runs = get_existing_wandb_runs()
    new_runs = current_runs - existing_runs
    if len(new_runs) == 1:
        return new_runs.pop()
    elif len(new_runs) > 1:
        return sorted(new_runs)[-1]
    return None


def parse_metrics_from_output(text):
    """Parse test metrics from trainer output."""
    metrics = {}
    for metric in ["RMSSE", "sMAPE", "RMSSC", "sMAPC"]:
        match = re.search(rf'{metric}\s+([0-9.]+)', text)
        if not match:
            match = re.search(rf'test_{metric}[\'"]?\s*[:\|]\s*([0-9.]+)', text)
        if match:
            metrics[metric] = float(match.group(1))
    return metrics


def count_parameters(hidden_units, n_blocks):
    """Calculate parameter count for N-HiTS-S."""
    from src.methods.NHITSS import LitNHITSS, num_parameters
    model = LitNHITSS(
        backcast_length_multiplier=8, forecast_length=6,
        hidden_layer_units=hidden_units, n_blocks=n_blocks,
        n_blocks_shared=1, ensemble_size=1)
    n_params = num_parameters(model.model[0])
    del model
    return n_params


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_single_config(hidden_units, n_blocks, seed):
    """Run a single N-HiTS-S configuration on M4 and return metrics."""
    
    script_content = NHITS_M4_TEMPLATE.format(
        hidden_layer_units=hidden_units,
        n_blocks=n_blocks,
        seed=seed,
    )
    
    temp_script = BASE_DIR / "_hp_tune_temp_nhits_m4.py"
    temp_script.write_text(script_content, encoding='utf-8')
    
    existing_runs = get_existing_wandb_runs()
    
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONIOENCODING"] = "utf-8"
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [PYTHON_EXE, str(temp_script)],
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=14400,  # 4 hours (M4 is larger than M3)
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 4 hours")
        return None, None
    
    elapsed = time.time() - start_time
    
    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        # Print last 30 lines for diagnostics
        lines = combined_output.strip().split('\n')
        for line in lines[-30:]:
            print(f"    {line}")
        return None, elapsed
    
    # Parse metrics
    metrics = parse_metrics_from_output(combined_output)
    
    # Find wandb run
    new_run = find_new_run(existing_runs)
    if new_run:
        metrics['wandb_run_id'] = new_run
    
    if not metrics.get('sMAPE'):
        print(f"  WARNING: Could not parse sMAPE from output")
        # Print last 20 lines
        lines = combined_output.strip().split('\n')
        for line in lines[-20:]:
            print(f"    {line}")
        return None, elapsed
    
    return metrics, elapsed


# MAIN

def main():
    total_configs = len(HIDDEN_UNITS_GRID) * len(N_BLOCKS_GRID)
    
    print(f"{'='*70}")
    print(f"  N-HiTS-S HYPERPARAMETER TUNING ON M4")
    print(f"  Grid: hidden={HIDDEN_UNITS_GRID} x blocks={N_BLOCKS_GRID}")
    print(f"  Total configs: {total_configs}")
    print(f"  Seed: {SEED}")
    print(f"  Dataset: M4 Monthly (validation set)")
    print(f"{'='*70}")
    
    fieldnames = [
        'architecture', 'hidden_layer_units', 'n_blocks', 'n_parameters',
        'seed', 'val_sMAPE', 'val_RMSSE', 'val_RMSSC', 'val_sMAPC',
        'time_seconds', 'wandb_run_id', 'timestamp'
    ]
    
    # Check if results file exists and has rows (for resuming)
    existing_configs = set()
    failed_configs = set()
    write_header = True
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (int(r['hidden_layer_units']), int(r['n_blocks']))
                if r.get('val_sMAPE') == 'FAILED':
                    failed_configs.add(key)
                else:
                    existing_configs.add(key)
        write_header = len(existing_configs) == 0 and len(failed_configs) == 0
        if existing_configs:
            print(f"  Resuming: {len(existing_configs)} configs already done")
        if failed_configs:
            print(f"  Will retry: {len(failed_configs)} previously failed configs")
    
    start_total = time.time()
    run_num = 0

    for hidden_units, n_blocks_val in product(HIDDEN_UNITS_GRID, N_BLOCKS_GRID):
        
        # Skip already completed
        if (hidden_units, n_blocks_val) in existing_configs:
            print(f"  SKIP (already done): hidden={hidden_units}, blocks={n_blocks_val}")
            continue
        
        run_num += 1
        remaining = total_configs - len(existing_configs) - run_num + 1
        
        print(f"\n{'='*70}")
        print(f"  [{run_num}/{total_configs - len(existing_configs)}] "
              f"N-HiTS-S: hidden={hidden_units}, blocks={n_blocks_val}")
        print(f"  Remaining after this: {remaining - 1}")
        print(f"{'='*70}")
        
        # Count parameters
        try:
            n_params = count_parameters(hidden_units, n_blocks_val)
        except Exception:
            n_params = -1
        
        # Run experiment
        metrics, elapsed = run_single_config(hidden_units, n_blocks_val, SEED)
        
        if metrics is None:
            print(f"  SKIPPED (failed)")
            row = {
                'architecture': 'NHITS',
                'hidden_layer_units': hidden_units,
                'n_blocks': n_blocks_val,
                'n_parameters': n_params,
                'seed': SEED,
                'val_sMAPE': 'FAILED',
                'val_RMSSE': 'FAILED',
                'val_RMSSC': 'FAILED',
                'val_sMAPC': 'FAILED',
                'time_seconds': elapsed or 0,
                'wandb_run_id': '',
                'timestamp': datetime.now().isoformat(),
            }
        else:
            elapsed_str = format_time(elapsed) if elapsed else "?"
            smape = metrics.get('sMAPE', '?')
            rmssc = metrics.get('RMSSC', '?')
            print(f"  Result: sMAPE={smape}, RMSSC={rmssc} ({elapsed_str})")
            
            row = {
                'architecture': 'NHITS',
                'hidden_layer_units': hidden_units,
                'n_blocks': n_blocks_val,
                'n_parameters': n_params,
                'seed': SEED,
                'val_sMAPE': metrics.get('sMAPE', ''),
                'val_RMSSE': metrics.get('RMSSE', ''),
                'val_RMSSC': metrics.get('RMSSC', ''),
                'val_sMAPC': metrics.get('sMAPC', ''),
                'time_seconds': round(elapsed, 1) if elapsed else 0,
                'wandb_run_id': metrics.get('wandb_run_id', ''),
                'timestamp': datetime.now().isoformat(),
            }
        
        # Append to CSV
        with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)
        
        print(f"  Saved to {RESULTS_FILE}")
    
    total_elapsed = time.time() - start_total
    print(f"\n{'='*70}")
    print(f"  ALL {total_configs} TUNING RUNS COMPLETE")
    print(f"  Total time: {format_time(total_elapsed)}")
    print(f"  Results: {RESULTS_FILE}")
    print(f"{'='*70}")
    
    # RETRY FAILED CONFIGS
    retry_failed_configs(fieldnames)
    
    # Print summary table
    print_summary()


def retry_failed_configs(fieldnames):
    """Re-run any configs that previously FAILED and update the CSV."""
    if not RESULTS_FILE.exists():
        return
    
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    failed_indices = [
        i for i, r in enumerate(all_rows)
        if r.get('val_sMAPE') == 'FAILED'
    ]
    
    if not failed_indices:
        print("\n  No failed configs to retry.")
        return
    
    print(f"\n{'='*70}")
    print(f"  RETRYING {len(failed_indices)} FAILED CONFIG(S)")
    print(f"{'='*70}")
    
    for idx in failed_indices:
        row = all_rows[idx]
        hidden_units = int(row['hidden_layer_units'])
        n_blocks_val = int(row['n_blocks'])
        
        print(f"\n  RETRY: hidden={hidden_units}, blocks={n_blocks_val}")
        
        try:
            n_params = count_parameters(hidden_units, n_blocks_val)
        except Exception:
            n_params = -1
        
        metrics, elapsed = run_single_config(hidden_units, n_blocks_val, SEED)
        
        if metrics is None:
            print(f"  RETRY FAILED AGAIN: hidden={hidden_units}, blocks={n_blocks_val}")
            continue
        
        elapsed_str = format_time(elapsed) if elapsed else "?"
        smape = metrics.get('sMAPE', '?')
        print(f"  RETRY SUCCESS: sMAPE={smape} ({elapsed_str})")
        
        # Update the row in-place
        all_rows[idx] = {
            'architecture': 'NHITS',
            'hidden_layer_units': hidden_units,
            'n_blocks': n_blocks_val,
            'n_parameters': n_params,
            'seed': SEED,
            'val_sMAPE': metrics.get('sMAPE', ''),
            'val_RMSSE': metrics.get('RMSSE', ''),
            'val_RMSSC': metrics.get('RMSSC', ''),
            'val_sMAPC': metrics.get('sMAPC', ''),
            'time_seconds': round(elapsed, 1) if elapsed else 0,
            'wandb_run_id': metrics.get('wandb_run_id', ''),
            'timestamp': datetime.now().isoformat(),
        }
    
    # Rewrite the entire CSV with updated rows
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"  Updated {RESULTS_FILE}")


def print_summary():
    """Print a summary of all tuning results."""
    if not RESULTS_FILE.exists():
        print("No results file found.")
        return
    
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("No results found.")
        return
    
    print(f"\n{'='*85}")
    print(f"  N-HiTS-S HP TUNING ON M4 — RESULTS SUMMARY")
    print(f"{'='*85}")
    
    print(f"\n  {'Hidden':<8} {'Blocks':<8} {'Params':>10} {'val_sMAPE':>12} {'val_RMSSC':>12} {'Time':>10}")
    print(f"  {'-'*62}")
    
    valid_smapes = []
    for r in rows:
        try:
            smape_val = float(r['val_sMAPE'])
            valid_smapes.append((smape_val, r))
        except (ValueError, KeyError):
            pass
        
        h = r['hidden_layer_units']
        b = r['n_blocks']
        p = r.get('n_parameters', '?')
        try:
            smape = f"{float(r['val_sMAPE']):.2f}"
        except (ValueError, KeyError):
            smape = r.get('val_sMAPE', '?')
        try:
            rmssc = f"{float(r['val_RMSSC']):.4f}"
        except (ValueError, KeyError):
            rmssc = r.get('val_RMSSC', '?')
        t = r.get('time_seconds', '?')
        try:
            t = format_time(float(t))
        except (ValueError, TypeError):
            pass
        
        print(f"  {h:<8} {b:<8} {p:>10} {smape:>12} {rmssc:>12} {t:>10}")
    
    if valid_smapes:
        best_smape, best_row = min(valid_smapes, key=lambda x: x[0])
        print(f"\n  >>> BEST: hidden={best_row['hidden_layer_units']}, "
              f"blocks={best_row['n_blocks']}, val_sMAPE={best_smape:.2f}")
    
    print(f"\n  Note: N-BEATS-S uses Van Belle et al. (2023) values: hidden=256, blocks=20")


if __name__ == '__main__':
    main()
