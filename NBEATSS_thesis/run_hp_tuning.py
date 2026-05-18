"""
Hyperparameter Tuning Script (Validation-Based)
=================================================
Tunes architecture hyperparameters using the VALIDATION set following
best practices: select hyperparameters on validation, evaluate final
model on test set.

This script evaluates different configurations of:
  - hidden_layer_units: [32, 128, 256, 512]
  - n_blocks: [3, 5, 10]

For BOTH architectures:
  - N-BEATS-S (via main.py)
  - N-HiTS-S (via main_nhits.py)

Conditions evaluated (Standard only, to isolate architecture effect):
  - lambda=0.0, ema=0.0, lr=1e-3, max_epochs=10

Uses eval_mode='validation' so that model performance is measured
on the held-out validation set (NOT the test set).

Single seed (seed=1) for the grid search. Best config should then
be re-run on the test set with 3 seeds.

Results saved to hp_tuning_results.csv.

Literature references for hyperparameter ranges:
  - Oreshkin et al. (2020) N-BEATS: 512 hidden, 30 blocks
  - Challu et al. (2023) N-HiTS: 512 hidden, varying blocks
  - Van Belle et al. (2023) N-BEATS-S: 256 hidden (code default)
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
RESULTS_FILE = BASE_DIR / "hp_tuning_results.csv"

# TUNING GRID
HIDDEN_UNITS_GRID = [32, 128, 256, 512]
N_BLOCKS_GRID = [3, 5, 10]
SEED = 1  # Single seed for tuning; best config re-run with 3 seeds later

# Which architectures to tune
ARCHITECTURES = ["NBEATS", "NHITS"]  # Both

# MAIN PY TEMPLATES

NBEATS_MAIN_TEMPLATE = '''# N-BEATS-S Hyperparameter Tuning Run (auto-generated)
import os
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
import torch

from src.methods.NBEATSS import LitNBEATSS
from src.utils.callbacks import LoadModelWarning, WriteForecastsToCSV, PlotTestPredictions

def main():
    wandb.login()
    project_name = "NBEATSS_thesis"

    # Dataset
    dataset = "M3"
    subset = "Monthly"
    dataset_id = "M3M"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    # Model architecture
    load_model = False
    update_loaded_model_specific_training_and_eval_hparams = False
    backcast_length_multiplier = 8
    forecast_length = 6
    hidden_layer_units = {hidden_layer_units}
    n_blocks = {n_blocks}
    n_blocks_shared = 1
    ensemble_size = 1
    zero_mean = True
    unit_variance = True

    # Training and evaluation
    eval_mode = 'validation'  # KEY: use validation set for HP tuning
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

    model = LitNBEATSS(
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

    from src.data.M3 import load_data
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

NHITS_MAIN_TEMPLATE = '''# N-HiTS-S Hyperparameter Tuning Run (auto-generated)
import os
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
import torch

from src.methods.NHITSS import LitNHITSS
from src.utils.callbacks import LoadModelWarning, WriteForecastsToCSV, PlotTestPredictions

def main():
    wandb.login()
    project_name = "NBEATSS_thesis"

    # Dataset
    dataset = "M3"
    subset = "Monthly"
    dataset_id = "M3M"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    # Model architecture
    load_model = False
    update_loaded_model_specific_training_and_eval_hparams = False
    backcast_length_multiplier = 8
    forecast_length = 6
    hidden_layer_units = {hidden_layer_units}
    n_blocks = {n_blocks}
    n_blocks_shared = 1
    ensemble_size = 1
    zero_mean = True
    unit_variance = True

    # Training and evaluation
    eval_mode = 'validation'  # KEY: use validation set for HP tuning
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

    from src.data.M3 import load_data
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


def count_parameters(arch, hidden_units, n_blocks):
    """Calculate parameter count without instantiating the model."""
    import torch
    if arch == "NBEATS":
        from src.methods.NBEATSS import LitNBEATSS, num_parameters
        model = LitNBEATSS(
            backcast_length_multiplier=8, forecast_length=6,
            hidden_layer_units=hidden_units, n_blocks=n_blocks,
            n_blocks_shared=1, ensemble_size=1)
        n_params = num_parameters(model.model[0])
    else:
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


def run_single_config(arch, hidden_units, n_blocks, seed):
    """Run a single configuration and return metrics."""
    
    # Write temporary runner script
    if arch == "NBEATS":
        template = NBEATS_MAIN_TEMPLATE
    else:
        template = NHITS_MAIN_TEMPLATE
    
    script_content = template.format(
        hidden_layer_units=hidden_units,
        n_blocks=n_blocks,
        seed=seed,
    )
    
    temp_script = BASE_DIR / f"_hp_tune_temp_{arch.lower()}.py"
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
            timeout=7200,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 2 hours")
        return None, None
    
    elapsed = time.time() - start_time
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        # Print last 20 lines of stderr for debugging
        err_lines = stderr.strip().split('\n')
        for line in err_lines[-20:]:
            print(f"    {line}")
        return None, elapsed
    
    # Parse metrics from output
    combined_output = stdout + "\n" + stderr
    metrics = parse_metrics_from_output(combined_output)
    
    # Find wandb run
    run_dir = find_new_run(existing_runs)
    if run_dir:
        run_id = re.search(r'-([a-z0-9]+)$', run_dir)
        if run_id:
            metrics['wandb_run_id'] = run_id.group(1)
    
    # Clean up temp script
    try:
        temp_script.unlink()
    except:
        pass
    
    return metrics, elapsed


def main():
    print("=" * 70)
    print("  HYPERPARAMETER TUNING (Validation-Based)")
    print("=" * 70)
    print(f"\n  Grid: hidden_units = {HIDDEN_UNITS_GRID}")
    print(f"         n_blocks    = {N_BLOCKS_GRID}")
    print(f"  Architectures:      {ARCHITECTURES}")
    print(f"  Seed:               {SEED}")
    
    total_configs = len(HIDDEN_UNITS_GRID) * len(N_BLOCKS_GRID) * len(ARCHITECTURES)
    print(f"  Total runs:         {total_configs}")
    
    # Print parameter counts for all configs
    print(f"\n  Parameter counts:")
    print(f"  {'Arch':<8} {'Hidden':<8} {'Blocks':<8} {'Params':>12}")
    print(f"  {'-'*40}")
    for arch in ARCHITECTURES:
        for h, b in product(HIDDEN_UNITS_GRID, N_BLOCKS_GRID):
            try:
                n_params = count_parameters(arch, h, b)
                print(f"  {arch:<8} {h:<8} {b:<8} {n_params:>12,}")
            except Exception as e:
                print(f"  {arch:<8} {h:<8} {b:<8} {'ERROR':>12}")
    
    # Prepare CSV
    fieldnames = [
        'architecture', 'hidden_layer_units', 'n_blocks', 'n_parameters',
        'seed', 'val_sMAPE', 'val_RMSSE', 'val_RMSSC', 'val_sMAPC',
        'time_seconds', 'wandb_run_id', 'timestamp'
    ]
    
    write_header = not RESULTS_FILE.exists()
    
    completed = 0
    start_total = time.time()
    
    for arch in ARCHITECTURES:
        for hidden_units, n_blocks_val in product(HIDDEN_UNITS_GRID, N_BLOCKS_GRID):
            completed += 1
            config_name = f"{arch} h={hidden_units} b={n_blocks_val}"
            
            print(f"\n{'='*70}")
            print(f"  [{completed}/{total_configs}] {config_name} (seed={SEED})")
            print(f"{'='*70}")
            
            # Count parameters
            try:
                n_params = count_parameters(arch, hidden_units, n_blocks_val)
            except:
                n_params = -1
            
            # Run experiment
            metrics, elapsed = run_single_config(arch, hidden_units, n_blocks_val, SEED)
            
            if metrics is None:
                print(f"  SKIPPED (failed)")
                row = {
                    'architecture': arch,
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
                    'architecture': arch,
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
    
    # Print summary table
    print_summary()


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
    print(f"  HYPERPARAMETER TUNING RESULTS SUMMARY")
    print(f"{'='*85}")
    
    for arch in ARCHITECTURES:
        arch_rows = [r for r in rows if r['architecture'] == arch]
        if not arch_rows:
            continue
        
        print(f"\n  {arch}:")
        print(f"  {'Hidden':<8} {'Blocks':<8} {'Params':>10} {'val_sMAPE':>12} {'val_RMSSC':>12} {'Time':>10}")
        print(f"  {'-'*62}")
        
        best_smape = float('inf')
        best_config = None
        
        for r in arch_rows:
            try:
                smape = float(r['val_sMAPE'])
            except (ValueError, TypeError):
                smape = float('inf')
            
            try:
                rmssc = float(r['val_RMSSC'])
            except (ValueError, TypeError):
                rmssc = '?'
            
            try:
                t = float(r['time_seconds'])
                time_str = format_time(t)
            except:
                time_str = '?'
            
            is_best = ""
            if smape < best_smape:
                best_smape = smape
                best_config = r
            
            print(f"  {r['hidden_layer_units']:<8} {r['n_blocks']:<8} "
                  f"{r['n_parameters']:>10} {smape:>12.4f} "
                  f"{rmssc if isinstance(rmssc, str) else f'{rmssc:>12.4f}':>12} {time_str:>10}")
        
        if best_config:
            print(f"\n  >>> BEST {arch}: hidden={best_config['hidden_layer_units']}, "
                  f"blocks={best_config['n_blocks']} "
                  f"(val_sMAPE={best_config['val_sMAPE']}, "
                  f"val_RMSSC={best_config['val_RMSSC']})")
    
    print(f"\n{'='*85}")
    print("  Next step: Re-run the best configuration(s) with 3 seeds on the TEST set.")
    print(f"{'='*85}")


if __name__ == '__main__':
    main()
