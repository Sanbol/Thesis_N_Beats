"""
Tuned N-BEATS-S Experiment Runner
==================================
Runs all experiments with architecture from Van Belle et al. (2023):
  hidden_layer_units = 256, n_blocks = 20  (from Van Belle et al., 2023, Table 3)

Conditions:
  A. Scratch Standard   (M3 only, lambda=0.0, ema=0.0)
  B. Scratch Stabilized (M3 only, lambda=0.02, ema=0.99)
  C. TL Standard        (M4->M3, lambda=0.0, ema=0.0)
  D. TL Stabilized      (M4->M3, lambda=0.02, ema=0.99)
  E. ZeroShot Standard  (M4->test M3, lambda=0.0, ema=0.0)
  F. ZeroShot Stabilized(M4->test M3, lambda=0.02, ema=0.99)

3 seeds each = 24 total runs (6 scratch + 6 pretrain + 6 finetune + 6 zeroshot).
Results saved to experiment_results_tuned.csv.
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

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
PYTHON_EXE = sys.executable  # works on Windows, Linux, and macOS
MAIN_PY = BASE_DIR / "main.py"
MAIN_PY_BACKUP = BASE_DIR / "main_backup_tuned.py"
WANDB_DIR = BASE_DIR / "wandb"
RESULTS_FILE = BASE_DIR / "experiment_results_tuned.csv"

# ============================================================
# EXPERIMENTAL DESIGN
# ============================================================
SEEDS = [1, 2, 3]

# ============================================================
# CONFIG TEMPLATE — Van Belle et al. (2023): hidden=256, n_blocks=20
# ============================================================
CONFIG_TEMPLATE = '''    ##########################
    # EXPERIMENT CONFIGURATION
    ##########################

    # Dataset
    dataset = "{dataset}"
    subset = "Monthly"
    dataset_id = "{dataset_id}"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    # Model
    load_model = {load_model}
    update_loaded_model_specific_training_and_eval_hparams = {update_hparams}

    # Model architecture - load existing model or specify model hyperparameters
    if load_model:
        model_id = "{model_id}"
        checkpoint = "last"
    else:
        backcast_length_multiplier = 8
        forecast_length = 6
        hidden_layer_units = 256
        n_blocks = 20
        n_blocks_shared = 1
        ensemble_size = 1
        zero_mean = True
        unit_variance = True

    # Model training and evaluation
    eval_mode = 'test'
    random_seed = {seed}
    ## Data hparams
    forecasting_origin_range_multiplier = 1e6
    batch_size = 32
    num_workers = 0
    ## Model-specific training and evaluation hparams
    if not load_model or update_loaded_model_specific_training_and_eval_hparams:
    # model-specific training and evaluation hparams below are ignored if (load_model == True) AND (update_loaded_model_specific_training_and_eval_hparams == False)
        lambda_stability = {lambda_stability}
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = {learning_rate}
        explr_gamma = {explr_gamma}
        ema_decay = {ema_decay}
    ## Trainer hparams
    max_norm = 1.0
    batches_per_epoch = 250
    patience = 1e6
    max_epochs = {max_epochs}

    # Other
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    save_forecasts = False
    plot_forecasts = False
'''


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def run_experiment(name, config_values):
    print(f"\n{'='*70}")
    print(f"  RUNNING: {name}")
    print(f"  Config: dataset={config_values['dataset']}, "
          f"lambda={config_values['lambda_stability']}, "
          f"ema={config_values['ema_decay']}, "
          f"lr={config_values['learning_rate']}, "
          f"epochs={config_values['max_epochs']}, "
          f"seed={config_values['seed']}, "
          f"load_model={config_values['load_model']}")
    print(f"{'='*70}")

    existing_runs = get_existing_wandb_runs()
    modify_main_py(config_values)

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
            timeout=172800  # 48 hour timeout (M4 eval is very slow)
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 48 hours for {name}")
        return None, None

    elapsed = time.time() - start_time

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        print(f"  ERROR in {name} (exit code {result.returncode}):")
        print(f"  STDOUT: {stdout[-1000:]}")
        print(f"  STDERR: {stderr[-1000:]}")
        return None, None

    new_run = find_new_run(existing_runs)
    run_id = extract_run_id(new_run) if new_run else None

    metrics = parse_metrics_from_output(stdout)
    if not metrics and new_run:
        output_log = WANDB_DIR / new_run / "files" / "output.log"
        if output_log.exists():
            metrics = parse_metrics_from_output(output_log.read_text(encoding='utf-8'))

    print(f"  Completed in {format_time(elapsed)}")
    print(f"  Run ID: {run_id}")
    if metrics:
        print(f"  Metrics: sMAPE={metrics.get('sMAPE', 'N/A')}, "
              f"RMSSE={metrics.get('RMSSE', 'N/A')}, "
              f"RMSSC={metrics.get('RMSSC', 'N/A')}, "
              f"sMAPC={metrics.get('sMAPC', 'N/A')}")

    return run_id, metrics


def save_results(results):
    if not results:
        return
    fieldnames = ["experiment", "condition", "seed", "run_id",
                  "sMAPE", "RMSSE", "RMSSC", "sMAPC"]
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    print(f"\n  Results saved to {RESULTS_FILE}")


def print_summary(results):
    core_results = [r for r in results if 'pretrain' not in r.get('condition', '')]

    print(f"\n{'='*90}")
    print(f"  N-BEATS-S TUNED RESULTS SUMMARY (hidden=256, blocks=20)")
    print(f"{'='*90}")
    print(f"  {'Condition':<30} {'Seed':<6} {'sMAPE':<12} {'RMSSE':<12} {'RMSSC':<12} {'sMAPC':<12}")
    print(f"  {'-'*85}")

    for r in core_results:
        smape = f"{r['sMAPE']:.4f}" if 'sMAPE' in r else 'N/A'
        rmsse = f"{r['RMSSE']:.4f}" if 'RMSSE' in r else 'N/A'
        rmssc = f"{r['RMSSC']:.4f}" if 'RMSSC' in r else 'N/A'
        smapc = f"{r['sMAPC']:.4f}" if 'sMAPC' in r else 'N/A'
        print(f"  {r['condition']:<30} {r['seed']:<6} {smape:<12} {rmsse:<12} {rmssc:<12} {smapc:<12}")

    conditions = ["Scratch_Standard", "Scratch_Stabilized",
                   "TL_Standard", "TL_Stabilized",
                   "ZeroShot_Standard", "ZeroShot_Stabilized"]
    print(f"\n  {'='*90}")
    print(f"  MEAN +/- STD (across seeds)")
    print(f"  {'='*90}")
    print(f"  {'Condition':<30} {'sMAPE':<16} {'RMSSE':<16} {'RMSSC':<16} {'sMAPC':<16}")
    print(f"  {'-'*85}")

    import statistics
    for cond in conditions:
        cond_results = [r for r in core_results if r.get('condition') == cond]
        if not cond_results:
            continue
        for metric in ["sMAPE", "RMSSE", "RMSSC", "sMAPC"]:
            values = [r[metric] for r in cond_results if metric in r]
            if values:
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                if metric == "sMAPE":
                    print(f"  {cond:<30} ", end="")
                print(f"{mean:.4f}+/-{std:.4f} ", end="")
        print()

    print(f"\n  KEY COMPARISON (thesis answer):")
    tl_stab = [r for r in core_results if r.get('condition') == 'TL_Stabilized']
    tl_std = [r for r in core_results if r.get('condition') == 'TL_Standard']
    if tl_stab and tl_std:
        stab_smape = statistics.mean([r['sMAPE'] for r in tl_stab if 'sMAPE' in r])
        std_smape = statistics.mean([r['sMAPE'] for r in tl_std if 'sMAPE' in r])
        diff = std_smape - stab_smape
        print(f"  TL Stabilized sMAPE: {stab_smape:.4f}")
        print(f"  TL Standard sMAPE:   {std_smape:.4f}")
        print(f"  Difference:          {diff:+.4f} ({'Stabilized wins' if diff > 0 else 'Standard wins'})")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    overall_start = time.time()
    print(f"\n{'#'*70}")
    print(f"  N-BEATS-S TUNED EXPERIMENT RUNNER")
    print(f"  Architecture: hidden=256, n_blocks=20 (Van Belle et al., 2023)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total experiments: 24 (6 scratch + 6 pretrain + 6 finetune + 6 zeroshot)")
    print(f"  NOTE: Van Belle architecture (256/20) — M4 runs can take many hours")
    print(f"{'#'*70}")

    shutil.copy2(MAIN_PY, MAIN_PY_BACKUP)
    print(f"\n  Backed up main.py -> main_backup_tuned.py")

    results = []
    pretrain_model_ids = {}

    try:
        # ===========================================================
        # PHASE A: SCRATCH STANDARD (3 seeds)
        # ===========================================================
        print(f"\n{'#'*70}")
        print(f"  PHASE A: Scratch Standard (N-BEATS on M3, hidden=256, blocks=20)")
        print(f"{'#'*70}")

        for seed in SEEDS:
            name = f"A_Scratch_Standard_seed{seed}"
            config = {
                "dataset": "M3", "dataset_id": "M3M",
                "load_model": "False", "update_hparams": "False",
                "model_id": "",
                "seed": seed,
                "lambda_stability": 0.0, "ema_decay": 0.0,
                "learning_rate": "1e-3", "explr_gamma": 1.0,
                "max_epochs": 30,
            }
            run_id, metrics = run_experiment(name, config)
            if metrics:
                results.append({
                    "experiment": name, "condition": "Scratch_Standard",
                    "seed": seed, "run_id": run_id, **metrics
                })
            save_results(results)

        # ===========================================================
        # PHASE A: SCRATCH STABILIZED (3 seeds)
        # ===========================================================
        print(f"\n{'#'*70}")
        print(f"  PHASE A: Scratch Stabilized (N-BEATS-S on M3, hidden=256, blocks=20)")
        print(f"{'#'*70}")

        for seed in SEEDS:
            name = f"B_Scratch_Stabilized_seed{seed}"
            config = {
                "dataset": "M3", "dataset_id": "M3M",
                "load_model": "False", "update_hparams": "False",
                "model_id": "",
                "seed": seed,
                "lambda_stability": 0.02, "ema_decay": 0.99,
                "learning_rate": "1e-3", "explr_gamma": 1.0,
                "max_epochs": 30,
            }
            run_id, metrics = run_experiment(name, config)
            if metrics:
                results.append({
                    "experiment": name, "condition": "Scratch_Stabilized",
                    "seed": seed, "run_id": run_id, **metrics
                })
            save_results(results)

        # ===========================================================
        # PHASE B: TL STANDARD PRE-TRAIN ON M4 (3 seeds)
        # ===========================================================
        print(f"\n{'#'*70}")
        print(f"  PHASE B: TL Standard Pre-train (N-BEATS on M4, hidden=256, blocks=20)")
        print(f"  NOTE: M4 runs can take many hours due to large eval set")
        print(f"{'#'*70}")

        for seed in SEEDS:
            name = f"C_TL_Standard_pretrain_seed{seed}"
            config = {
                "dataset": "M4", "dataset_id": "M4M",
                "load_model": "False", "update_hparams": "False",
                "model_id": "",
                "seed": seed,
                "lambda_stability": 0.0, "ema_decay": 0.0,
                "learning_rate": "1e-3", "explr_gamma": 1.0,
                "max_epochs": 30,
            }
            run_id, metrics = run_experiment(name, config)
            pretrain_model_ids[f"Standard_seed{seed}"] = run_id
            if metrics:
                results.append({
                    "experiment": name, "condition": "TL_Standard_pretrain",
                    "seed": seed, "run_id": run_id, **metrics
                })
            save_results(results)
            print(f"  >> Stored pretrain model_id for Standard seed{seed}: {run_id}")

        # ===========================================================
        # PHASE B: TL STABILIZED PRE-TRAIN ON M4 (3 seeds)
        # ===========================================================
        print(f"\n{'#'*70}")
        print(f"  PHASE B: TL Stabilized Pre-train (N-BEATS-S on M4, hidden=256, blocks=20)")
        print(f"{'#'*70}")

        for seed in SEEDS:
            name = f"D_TL_Stabilized_pretrain_seed{seed}"
            config = {
                "dataset": "M4", "dataset_id": "M4M",
                "load_model": "False", "update_hparams": "False",
                "model_id": "",
                "seed": seed,
                "lambda_stability": 0.02, "ema_decay": 0.99,
                "learning_rate": "1e-3", "explr_gamma": 1.0,
                "max_epochs": 30,
            }
            run_id, metrics = run_experiment(name, config)
            pretrain_model_ids[f"Stabilized_seed{seed}"] = run_id
            if metrics:
                results.append({
                    "experiment": name, "condition": "TL_Stabilized_pretrain",
                    "seed": seed, "run_id": run_id, **metrics
                })
            save_results(results)
            print(f"  >> Stored pretrain model_id for Stabilized seed{seed}: {run_id}")

        # ===========================================================
        # PHASE C: TL STANDARD FINE-TUNE ON M3 (3 seeds)
        # ===========================================================
        print(f"\n{'#'*70}")
        print(f"  PHASE C: TL Standard Fine-tune (M4->M3, hidden=256, blocks=20)")
        print(f"{'#'*70}")

        for seed in SEEDS:
            model_id = pretrain_model_ids.get(f"Standard_seed{seed}")
            if not model_id:
                print(f"  SKIPPING: No pretrain model_id for Standard seed{seed}")
                continue
            name = f"C_TL_Standard_finetune_seed{seed}"
            config = {
                "dataset": "M3", "dataset_id": "M3M",
                "load_model": "True", "update_hparams": "True",
                "model_id": model_id,
                "seed": seed,
                "lambda_stability": 0.0, "ema_decay": 0.0,
                "learning_rate": "1e-5", "explr_gamma": 0.97,
                "max_epochs": 20,
            }
            run_id, metrics = run_experiment(name, config)
            if metrics:
                results.append({
                    "experiment": name, "condition": "TL_Standard",
                    "seed": seed, "run_id": run_id, **metrics
                })
            save_results(results)

        # ===========================================================
        # PHASE C: TL STABILIZED FINE-TUNE ON M3 (3 seeds)
        # ===========================================================
        print(f"\n{'#'*70}")
        print(f"  PHASE C: TL Stabilized Fine-tune (M4->M3, hidden=256, blocks=20)")
        print(f"{'#'*70}")

        for seed in SEEDS:
            model_id = pretrain_model_ids.get(f"Stabilized_seed{seed}")
            if not model_id:
                print(f"  SKIPPING: No pretrain model_id for Stabilized seed{seed}")
                continue
            name = f"D_TL_Stabilized_finetune_seed{seed}"
            config = {
                "dataset": "M3", "dataset_id": "M3M",
                "load_model": "True", "update_hparams": "True",
                "model_id": model_id,
                "seed": seed,
                "lambda_stability": 0.02, "ema_decay": 0.99,
                "learning_rate": "1e-5", "explr_gamma": 0.97,
                "max_epochs": 20,
            }
            run_id, metrics = run_experiment(name, config)
            if metrics:
                results.append({
                    "experiment": name, "condition": "TL_Stabilized",
                    "seed": seed, "run_id": run_id, **metrics
                })
            save_results(results)

        # ===========================================================
        # PHASE D: ZERO-SHOT (load M4 pretrained, test on M3, no training)
        # ===========================================================
        print(f"\n{'#'*70}")
        print(f"  PHASE D: Zero-Shot (M4->test M3, no fine-tuning)")
        print(f"{'#'*70}")

        for condition_name, lambda_val, ema_val in [("Standard", 0.0, 0.0), ("Stabilized", 0.02, 0.99)]:
            for seed in SEEDS:
                model_id = pretrain_model_ids.get(f"{condition_name}_seed{seed}")
                if not model_id:
                    print(f"  SKIPPING: No pretrain model_id for {condition_name} seed{seed}")
                    continue
                name = f"E_ZeroShot_{condition_name}_seed{seed}"
                config = {
                    "dataset": "M3", "dataset_id": "M3M",
                    "load_model": "True", "update_hparams": "True",
                    "model_id": model_id,
                    "seed": seed,
                    "lambda_stability": lambda_val, "ema_decay": ema_val,
                    "learning_rate": "1e-5", "explr_gamma": 1.0,
                    "max_epochs": 0,  # ZERO SHOT: no training
                }
                run_id, metrics = run_experiment(name, config)
                if metrics:
                    results.append({
                        "experiment": name, "condition": f"ZeroShot_{condition_name}",
                        "seed": seed, "run_id": run_id, **metrics
                    })
                save_results(results)

    finally:
        if MAIN_PY_BACKUP.exists():
            shutil.copy2(MAIN_PY_BACKUP, MAIN_PY)
            print(f"\n  Restored main.py from backup")

    total_time = time.time() - overall_start
    print(f"\n  Total runtime: {format_time(total_time)}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    save_results(results)
    print_summary(results)


if __name__ == '__main__':
    main()
