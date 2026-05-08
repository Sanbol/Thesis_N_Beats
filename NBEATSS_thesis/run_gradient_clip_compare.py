"""
Gradient Clipping Sensitivity Comparison
==========================================
Per Jente's request: test whether gradient_clip_val=1.0 severely impacts
learning curves vs effectively-no-clipping (clip=10.0).

Two N-BEATS-S runs on M3 Scratch, otherwise identical (Scratch Stabilized
config, lambda=0.02, bs_mult=6, max_epochs=155, batches_per_epoch=93,
bs=512, fp32, eval_mode='validation').

Decision rule (from Jente):
- Small difference (< 5% in final val_loss): clipping is benign. KEEP 1.0.
- Large difference (>= 5%): clipping deviates from no-clipping reference.
  Consider raising to 10.0 for closer alignment. Document in methodology.

Results saved to gradient_clip_compare.csv.
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
MAIN_PY_BACKUP = BASE_DIR / "main_backup_clip_compare.py"
RESULTS_FILE = BASE_DIR / "gradient_clip_compare.csv"
WANDB_DIR = BASE_DIR / "wandb"

CLIP_VALUES = [1.0, 10.0]  # 10.0 = effectively no clipping for this model
SEED = 1
LAMBDA = 0.02  # update this to the winner from run_lambda_recheck.py if it changed

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
    max_norm = {max_norm}
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
    for metric in ["RMSSE", "sMAPE", "RMSSC", "sMAPC", "vloss"]:
        match = re.search(rf'{metric}\s+([0-9.]+)', text)
        if not match:
            match = re.search(rf'[\'\"]{metric}[\'\"]?\s*[:\|]\s*([0-9.]+)', text)
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


def run_experiment(clip_val, lambda_val, seed):
    label = f"clip_{clip_val}_lambda_{lambda_val}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")

    existing_runs = get_existing_wandb_runs()
    modify_main_py({"max_norm": clip_val, "lambda_stability": lambda_val, "seed": seed})

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
        print(f"  TIMEOUT for {label}")
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
              f"vloss={metrics.get('vloss', 'N/A')}")

    return run_id, metrics


def save_results(results):
    if not results:
        return
    fieldnames = ["clip_val", "lambda_stability", "seed", "run_id",
                  "sMAPE", "RMSSE", "RMSSC", "sMAPC", "vloss"]
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    print(f"\n  Results saved to {RESULTS_FILE}")


def main():
    overall_start = time.time()
    print(f"\n{'#'*60}")
    print(f"  GRADIENT CLIPPING SENSITIVITY COMPARISON")
    print(f"  Clip values: {CLIP_VALUES}  (10.0 = effectively no clipping)")
    print(f"  Config: M3 Scratch Stabilized, lambda={LAMBDA}, bs_mult=6")
    print(f"  Budget: bs=512, batches=93, max_epochs=155, fp32, val mode")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    shutil.copy2(MAIN_PY, MAIN_PY_BACKUP)

    results = []

    try:
        for clip in CLIP_VALUES:
            run_id, metrics = run_experiment(clip, LAMBDA, SEED)
            row = {"clip_val": clip, "lambda_stability": LAMBDA,
                   "seed": SEED, "run_id": run_id or ""}
            if metrics:
                row.update(metrics)
            results.append(row)
            save_results(results)
    finally:
        if MAIN_PY_BACKUP.exists():
            shutil.copy2(MAIN_PY_BACKUP, MAIN_PY)
            print(f"\n  Restored main.py from backup")

    print(f"\n  Total runtime: {format_time(time.time() - overall_start)}")

    print(f"\n  GRADIENT CLIPPING SUMMARY:")
    print(f"  {'Clip':<10} {'sMAPE':<12} {'RMSSE':<12} {'vloss':<12}")
    print(f"  {'-'*45}")
    for r in results:
        smape = f"{r['sMAPE']:.4f}" if 'sMAPE' in r else 'N/A'
        rmsse = f"{r['RMSSE']:.4f}" if 'RMSSE' in r else 'N/A'
        vloss = f"{r['vloss']:.6f}" if 'vloss' in r else 'N/A'
        print(f"  {r['clip_val']:<10} {smape:<12} {rmsse:<12} {vloss:<12}")

    if len(results) == 2 and 'sMAPE' in results[0] and 'sMAPE' in results[1]:
        diff_pct = abs(results[0]['sMAPE'] - results[1]['sMAPE']) / results[0]['sMAPE'] * 100
        print(f"\n  sMAPE difference: {diff_pct:.1f}%")
        if diff_pct < 5:
            print(f"  VERDICT: < 5% difference — clipping is benign. KEEP gradient_clip_val=1.0.")
        else:
            print(f"  VERDICT: >= 5% difference — clipping substantially affects training.")
            print(f"  Consider raising gradient_clip_val to 10.0 for closer alignment with prof.")
            print(f"  Update max_norm in main.py and main_nhits.py if you decide to change.")

    save_results(results)


if __name__ == '__main__':
    main()
