"""
N-BEATS-S Lambda Re-check Sweep
=================================
Runs N-BEATS-S on M3 Scratch with converged settings (bs=512, batches=93,
max_epochs=155, fp32) across lambda values {0.02, 0.05, 0.1, 0.15}.

Purpose: original lambda=0.02 was tuned on under-trained models (30 epochs,
bs=32). Re-validating on converged training budget to confirm or shift the
optimal stability weight.

Decision rule: pick lambda with best accuracy/stability trade-off —
lowest sMAPE among configurations where RMSSC <= 110% of Standard model RMSSC.

Results saved to lambda_recheck_results.csv.
"""

import subprocess
import os
import sys
import re
import csv
import time
import shutil
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
PYTHON_EXE = sys.executable
MAIN_PY = BASE_DIR / "main.py"
MAIN_PY_BACKUP = BASE_DIR / "main_backup_lambda_recheck.py"
RESULTS_FILE = BASE_DIR / "lambda_recheck_results.csv"
WANDB_DIR = BASE_DIR / "wandb"

LAMBDA_VALUES = [0.0, 0.02, 0.05, 0.1, 0.15]
SEED = 1

CONFIG_TEMPLATE = '''    ##########################
    # EXPERIMENT CONFIGURATION
    ##########################

    # Dataset
    dataset = "M3"
    subset = "Monthly"
    dataset_id = "M3M"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    # Model
    load_model = False
    update_loaded_model_specific_training_and_eval_hparams = False

    # Model architecture - load existing model or specify model hyperparameters
    if load_model:
        model_id = ""
        checkpoint = "last"
    else:
        backcast_length_multiplier = 6
        forecast_length = 6
        hidden_layer_units = 256
        n_blocks = 20
        n_blocks_shared = 1
        ensemble_size = 1
        zero_mean = True
        unit_variance = True

    # Model training and evaluation
    eval_mode = 'validation'
    random_seed = {seed}
    ## Data hparams
    forecasting_origin_range_multiplier = 1e6
    batch_size = 512
    num_workers = 0
    ## Model-specific training and evaluation hparams
    if not load_model or update_loaded_model_specific_training_and_eval_hparams:
    # model-specific training and evaluation hparams below are ignored if (load_model == True) AND (update_loaded_model_specific_training_and_eval_hparams == False)
        lambda_stability = {lambda_stability}
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = 1e-3
        explr_gamma = 1.0
        ema_decay = 0.0
    ## Trainer hparams
    max_norm = 1.0
    batches_per_epoch = 93
    patience = 1e6
    max_epochs = 155

    # Other
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    save_forecasts = False
    plot_forecasts = False
'''


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


def extract_run_id(run_dir_name):
    match = re.search(r'-([a-z0-9]+)$', run_dir_name)
    return match.group(1) if match else None


def parse_metrics_from_output(text):
    metrics = {}
    for metric in ["RMSSE", "sMAPE", "RMSSC", "sMAPC"]:
        match = re.search(rf'{metric}\s+([0-9.]+)', text)
        if not match:
            match = re.search(rf'test_{metric}[\'"]?\s*[:\|]\s*([0-9.]+)', text)
        if match:
            metrics[metric] = float(match.group(1))
    return metrics


def modify_main_py(config_values):
    content = MAIN_PY.read_text(encoding='utf-8')
    start_marker = "    ##########################\n    # EXPERIMENT CONFIGURATION\n    ##########################"
    end_marker = "\n    ###################################################################################################"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find config section boundaries in main.py")
    new_config = CONFIG_TEMPLATE.format(**config_values)
    new_content = content[:start_idx] + new_config + content[end_idx:]
    MAIN_PY.write_text(new_content, encoding='utf-8')


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_experiment(lambda_val, seed):
    name = f"Lambda_{lambda_val}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  RUNNING: {name}  (lambda_stability={lambda_val})")
    print(f"{'='*60}")

    existing_runs = get_existing_wandb_runs()
    modify_main_py({"lambda_stability": lambda_val, "seed": seed})

    start_time = time.time()
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [PYTHON_EXE, str(MAIN_PY)],
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=86400,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {name}")
        return None, None

    elapsed = time.time() - start_time
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        print(f"  ERROR (exit code {result.returncode}):")
        print(f"  STDERR: {stderr[-1000:]}")
        return None, None

    new_run = find_new_run(existing_runs)
    run_id = extract_run_id(new_run) if new_run else None

    metrics = parse_metrics_from_output(stdout)
    if not metrics and new_run:
        output_log = WANDB_DIR / new_run / "files" / "output.log"
        if output_log.exists():
            metrics = parse_metrics_from_output(output_log.read_text(encoding='utf-8'))

    print(f"  Completed in {format_time(elapsed)} | Run ID: {run_id}")
    if metrics:
        print(f"  sMAPE={metrics.get('sMAPE', 'N/A')}, RMSSE={metrics.get('RMSSE', 'N/A')}, "
              f"RMSSC={metrics.get('RMSSC', 'N/A')}, sMAPC={metrics.get('sMAPC', 'N/A')}")

    return run_id, metrics


def save_results(results):
    if not results:
        return
    fieldnames = ["lambda_stability", "seed", "run_id", "sMAPE", "RMSSE", "RMSSC", "sMAPC"]
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    print(f"\n  Results saved to {RESULTS_FILE}")


def pick_winner(results):
    standard = next((r for r in results if r['lambda_stability'] == 0.0), None)
    if not standard or 'RMSSC' not in standard:
        print("  WARNING: No Standard (lambda=0) baseline for comparison.")
        return None
    threshold = standard['RMSSC'] * 1.10
    candidates = [r for r in results if r.get('lambda_stability', 0) > 0
                  and 'sMAPE' in r and 'RMSSC' in r and r['RMSSC'] <= threshold]
    if not candidates:
        print("  No lambda value improves stability within 110% RMSSC threshold.")
        print("  Defaulting to lambda=0.02.")
        return 0.02
    winner = min(candidates, key=lambda r: r['sMAPE'])
    return winner['lambda_stability']


def main():
    overall_start = time.time()
    print(f"\n{'#'*60}")
    print(f"  N-BEATS-S LAMBDA RE-CHECK SWEEP")
    print(f"  Lambdas: {LAMBDA_VALUES}")
    print(f"  Settings: bs=512, batches=93, max_epochs=155, fp32, seed={SEED}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    shutil.copy2(MAIN_PY, MAIN_PY_BACKUP)

    results = []

    try:
        for lam in LAMBDA_VALUES:
            run_id, metrics = run_experiment(lam, SEED)
            row = {"lambda_stability": lam, "seed": SEED, "run_id": run_id or ""}
            if metrics:
                row.update(metrics)
            results.append(row)
            save_results(results)
    finally:
        if MAIN_PY_BACKUP.exists():
            shutil.copy2(MAIN_PY_BACKUP, MAIN_PY)
            print(f"\n  Restored main.py from backup")

    print(f"\n  Total runtime: {format_time(time.time() - overall_start)}")
    print(f"\n  LAMBDA RE-CHECK SUMMARY:")
    print(f"  {'Lambda':<10} {'sMAPE':<12} {'RMSSE':<12} {'RMSSC':<12} {'sMAPC':<12}")
    print(f"  {'-'*55}")
    for r in results:
        smape = f"{r['sMAPE']:.4f}" if 'sMAPE' in r else 'N/A'
        rmsse = f"{r['RMSSE']:.4f}" if 'RMSSE' in r else 'N/A'
        rmssc = f"{r['RMSSC']:.4f}" if 'RMSSC' in r else 'N/A'
        smapc = f"{r['sMAPC']:.4f}" if 'sMAPC' in r else 'N/A'
        print(f"  {r['lambda_stability']:<10} {smape:<12} {rmsse:<12} {rmssc:<12} {smapc:<12}")

    winner = pick_winner(results)
    if winner is not None:
        print(f"\n  WINNER: lambda_stability = {winner}")
        print(f"  Update run_experiments_tuned.py and run_nhits_experiments_tuned.py")
        print(f"  with lambda_stability={winner} in Stabilized phase configs.")

    save_results(results)


if __name__ == '__main__':
    main()
