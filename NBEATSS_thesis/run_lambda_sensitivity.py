"""
Lambda Sensitivity Analysis for N-BEATS-S
==========================================
Tests different λ values to map the accuracy-stability Pareto frontier.

We already have results for:
  λ = 0.00 (Scratch Standard, β=0.0) — from experiment_results.csv
  λ = 0.02 (Scratch Stabilized, β=0.99) — from experiment_results.csv

This script runs NEW experiments for:
  λ ∈ {0.005, 0.01, 0.05, 0.10, 0.20} with β=0.99 (EMA on)

All runs: N-BEATS-S, Scratch on M3, 10 epochs, 3 seeds each.
Results saved to lambda_sensitivity_results.csv.
After all runs, generates Pareto frontier plot.
"""

import subprocess
import os
import re
import csv
import time
import shutil
import statistics
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
PYTHON_EXE = str(BASE_DIR.parent / ".venv" / "Scripts" / "python.exe")
MAIN_PY = BASE_DIR / "main.py"
MAIN_PY_BACKUP = BASE_DIR / "main_backup_lambda.py"
WANDB_DIR = BASE_DIR / "wandb"
RESULTS_FILE = BASE_DIR / "lambda_sensitivity_results.csv"
EXISTING_RESULTS = BASE_DIR / "experiment_results.csv"

SEEDS = [1, 2, 3]

# Lambda values to test (NEW runs only - we already have 0.0 and 0.02)
NEW_LAMBDAS = [0.005, 0.01, 0.05, 0.10, 0.20]

# All runs use EMA β=0.99, scratch M3 training, 10 epochs
EMA_DECAY = 0.99

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

    # Model architecture
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

    # Model training and evaluation
    eval_mode = 'test'
    random_seed = {seed}
    ## Data hparams
    forecasting_origin_range_multiplier = 1e6
    batch_size = 32
    num_workers = 0
    ## Model-specific training and evaluation hparams
    if not load_model or update_loaded_model_specific_training_and_eval_hparams:
        lambda_stability = {lambda_stability}
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = 1e-3
        explr_gamma = 1.0
        ema_decay = {ema_decay}
    ## Trainer hparams
    max_norm = 1.0
    batches_per_epoch = 50
    patience = 1e6
    max_epochs = 10

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
    if not text:
        return {}
    metrics = {}
    for metric in ["RMSSE", "sMAPE", "RMSSC", "sMAPC"]:
        match = re.search(rf'{metric}\s+([0-9.]+)', text)
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
    print(f"  λ={config_values['lambda_stability']}, β={config_values['ema_decay']}, seed={config_values['seed']}")
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
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=7200
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {name}")
        return None, None

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"  ERROR in {name} (exit code {result.returncode}):")
        print(f"  {result.stderr[-1000:]}")
        return None, None

    new_run = find_new_run(existing_runs)
    run_id = extract_run_id(new_run) if new_run else None

    metrics = parse_metrics_from_output(result.stdout)
    if not metrics and new_run:
        output_log = WANDB_DIR / new_run / "files" / "output.log"
        if output_log.exists():
            metrics = parse_metrics_from_output(output_log.read_text(encoding='utf-8'))

    print(f"  Completed in {format_time(elapsed)}")
    if metrics:
        print(f"  sMAPE={metrics.get('sMAPE', 'N/A')}, "
              f"RMSSE={metrics.get('RMSSE', 'N/A')}, "
              f"RMSSC={metrics.get('RMSSC', 'N/A')}, "
              f"sMAPC={metrics.get('sMAPC', 'N/A')}")
    return run_id, metrics


def load_existing_results():
    rows = []
    if not EXISTING_RESULTS.exists():
        print("  WARNING: experiment_results.csv not found, skipping existing results")
        return rows

    with open(EXISTING_RESULTS, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row.get('condition', '')
            if cond == 'Scratch_Standard':
                rows.append({
                    'lambda': 0.0,
                    'ema_decay': 0.0,
                    'seed': int(row['seed']),
                    'run_id': row.get('run_id', ''),
                    'sMAPE': float(row['sMAPE']),
                    'RMSSE': float(row['RMSSE']),
                    'RMSSC': float(row['RMSSC']),
                    'sMAPC': float(row['sMAPC']),
                })
            elif cond == 'Scratch_Stabilized':
                rows.append({
                    'lambda': 0.02,
                    'ema_decay': 0.99,
                    'seed': int(row['seed']),
                    'run_id': row.get('run_id', ''),
                    'sMAPE': float(row['sMAPE']),
                    'RMSSE': float(row['RMSSE']),
                    'RMSSC': float(row['RMSSC']),
                    'sMAPC': float(row['sMAPC']),
                })
    print(f"  Loaded {len(rows)} existing results (λ=0.0 and λ=0.02)")
    return rows


def save_results(results):
    fieldnames = ["lambda", "ema_decay", "seed", "run_id",
                  "sMAPE", "RMSSE", "RMSSC", "sMAPC"]
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    print(f"  Results saved to {RESULTS_FILE}")


def generate_pareto_plot(results):
    """Generate Pareto frontier plot: sMAPE vs RMSSC for each λ."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available, skipping plot generation")
        return

    # Group by lambda, compute mean ± std
    lambda_groups = {}
    for r in results:
        lam = r['lambda']
        if lam not in lambda_groups:
            lambda_groups[lam] = {'sMAPE': [], 'RMSSC': [], 'sMAPC': [], 'RMSSE': []}
        for m in ['sMAPE', 'RMSSC', 'sMAPC', 'RMSSE']:
            lambda_groups[lam][m].append(r[m])

    lambdas_sorted = sorted(lambda_groups.keys())
    smape_means = [statistics.mean(lambda_groups[l]['sMAPE']) for l in lambdas_sorted]
    smape_stds = [statistics.stdev(lambda_groups[l]['sMAPE']) if len(lambda_groups[l]['sMAPE']) > 1 else 0 for l in lambdas_sorted]
    rmssc_means = [statistics.mean(lambda_groups[l]['RMSSC']) for l in lambdas_sorted]
    rmssc_stds = [statistics.stdev(lambda_groups[l]['RMSSC']) if len(lambda_groups[l]['RMSSC']) > 1 else 0 for l in lambdas_sorted]
    smapc_means = [statistics.mean(lambda_groups[l]['sMAPC']) for l in lambdas_sorted]
    smapc_stds = [statistics.stdev(lambda_groups[l]['sMAPC']) if len(lambda_groups[l]['sMAPC']) > 1 else 0 for l in lambdas_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Plot 1: sMAPE vs λ ---
    ax1 = axes[0]
    ax1.errorbar(lambdas_sorted, smape_means, yerr=smape_stds,
                 fmt='o-', color='#2196F3', capsize=5, linewidth=2, markersize=8)
    for i, lam in enumerate(lambdas_sorted):
        ax1.annotate(f'λ={lam}', (lam, smape_means[i]),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=8, color='gray')
    ax1.set_xlabel('λ (stability weight)', fontsize=12)
    ax1.set_ylabel('sMAPE (accuracy, lower=better)', fontsize=12)
    ax1.set_title('Accuracy vs λ', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: RMSSC vs λ ---
    ax2 = axes[1]
    ax2.errorbar(lambdas_sorted, rmssc_means, yerr=rmssc_stds,
                 fmt='s-', color='#FF5722', capsize=5, linewidth=2, markersize=8)
    for i, lam in enumerate(lambdas_sorted):
        ax2.annotate(f'λ={lam}', (lam, rmssc_means[i]),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=8, color='gray')
    ax2.set_xlabel('λ (stability weight)', fontsize=12)
    ax2.set_ylabel('RMSSC (stability, lower=better)', fontsize=12)
    ax2.set_title('Stability vs λ', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Pareto frontier (sMAPE vs RMSSC) ---
    ax3 = axes[2]
    ax3.errorbar(rmssc_means, smape_means,
                 xerr=rmssc_stds, yerr=smape_stds,
                 fmt='D-', color='#4CAF50', capsize=5, linewidth=2, markersize=8)
    for i, lam in enumerate(lambdas_sorted):
        ax3.annotate(f'λ={lam}', (rmssc_means[i], smape_means[i]),
                     textcoords="offset points", xytext=(10, 5),
                     ha='left', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax3.set_xlabel('RMSSC (stability, lower=better)', fontsize=12)
    ax3.set_ylabel('sMAPE (accuracy, lower=better)', fontsize=12)
    ax3.set_title('Pareto Frontier: Accuracy vs Stability', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Highlight the "ideal" corner (bottom-left)
    ax3.annotate('← ideal', xy=(min(rmssc_means) - 0.01, min(smape_means) - 0.1),
                 fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    plot_path = BASE_DIR / "lambda_pareto_frontier.png"
    plt.savefig(str(plot_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Pareto frontier plot saved: {plot_path}")

    # --- Also make a summary table plot ---
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    table_data = [['λ', 'β (EMA)', 'sMAPE ↓', 'RMSSE ↓', 'RMSSC ↓', 'sMAPC ↓']]
    for lam in lambdas_sorted:
        g = lambda_groups[lam]
        ema = 0.0 if lam == 0.0 else 0.99
        table_data.append([
            f'{lam}',
            f'{ema}',
            f'{statistics.mean(g["sMAPE"]):.3f} ± {statistics.stdev(g["sMAPE"]) if len(g["sMAPE"])>1 else 0:.3f}',
            f'{statistics.mean(g["RMSSE"]):.3f} ± {statistics.stdev(g["RMSSE"]) if len(g["RMSSE"])>1 else 0:.3f}',
            f'{statistics.mean(g["RMSSC"]):.3f} ± {statistics.stdev(g["RMSSC"]) if len(g["RMSSC"])>1 else 0:.3f}',
            f'{statistics.mean(g["sMAPC"]):.3f} ± {statistics.stdev(g["sMAPC"]) if len(g["sMAPC"])>1 else 0:.3f}',
        ])

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight best RMSSC row
    best_rmssc_idx = rmssc_means.index(min(rmssc_means))
    for j in range(len(table_data[0])):
        table[best_rmssc_idx + 1, j].set_facecolor('#E8F5E9')

    ax.set_title('Lambda Sensitivity Analysis — N-BEATS-S Scratch on M3',
                 fontsize=14, fontweight='bold', pad=20)
    
    table_path = BASE_DIR / "lambda_sensitivity_table.png"
    plt.savefig(str(table_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Summary table saved: {table_path}")


def print_summary(results):
    lambda_groups = {}
    for r in results:
        lam = r['lambda']
        if lam not in lambda_groups:
            lambda_groups[lam] = {'sMAPE': [], 'RMSSC': [], 'sMAPC': [], 'RMSSE': []}
        for m in ['sMAPE', 'RMSSC', 'sMAPC', 'RMSSE']:
            lambda_groups[lam][m].append(r[m])

    print(f"\n{'='*90}")
    print(f"  LAMBDA SENSITIVITY ANALYSIS — RESULTS")
    print(f"{'='*90}")
    print(f"  {'λ':<8} {'β':<6} {'sMAPE':<16} {'RMSSE':<16} {'RMSSC':<16} {'sMAPC':<16}")
    print(f"  {'-'*85}")

    for lam in sorted(lambda_groups.keys()):
        g = lambda_groups[lam]
        ema = '0.0' if lam == 0.0 else '0.99'
        parts = []
        for m in ['sMAPE', 'RMSSE', 'RMSSC', 'sMAPC']:
            mean = statistics.mean(g[m])
            std = statistics.stdev(g[m]) if len(g[m]) > 1 else 0
            parts.append(f"{mean:.3f}±{std:.3f}")
        print(f"  {lam:<8} {ema:<6} {'  '.join(parts)}")

    print(f"\n  Total runs: {len(results)} ({len(results)//3} conditions × 3 seeds)")


def main():
    overall_start = time.time()
    n_new = len(NEW_LAMBDAS) * len(SEEDS)

    print(f"\n{'#'*70}")
    print(f"  LAMBDA SENSITIVITY ANALYSIS — N-BEATS-S")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  New λ values: {NEW_LAMBDAS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  New experiments: {n_new}")
    print(f"  Existing results: λ=0.0 and λ=0.02 (6 runs)")
    print(f"  Total after merge: {n_new + 6} runs, {len(NEW_LAMBDAS) + 2} λ values")
    print(f"  Estimated time: ~{n_new * 30 // 60} minutes")
    print(f"{'#'*70}")

    shutil.copy2(MAIN_PY, MAIN_PY_BACKUP)
    print(f"\n  Backed up main.py → {MAIN_PY_BACKUP.name}")

    all_results = load_existing_results()

    try:
        for lam in NEW_LAMBDAS:
            print(f"\n{'#'*70}")
            print(f"  λ = {lam}  (β = {EMA_DECAY})")
            print(f"{'#'*70}")

            for seed in SEEDS:
                name = f"Lambda_{lam}_seed{seed}"
                config = {
                    "seed": seed,
                    "lambda_stability": lam,
                    "ema_decay": EMA_DECAY,
                }
                run_id, metrics = run_experiment(name, config)
                if metrics:
                    all_results.append({
                        'lambda': lam,
                        'ema_decay': EMA_DECAY,
                        'seed': seed,
                        'run_id': run_id or '',
                        **metrics
                    })
                save_results(all_results)

    except KeyboardInterrupt:
        print(f"\n  Interrupted! Saving partial results...")
    finally:
        # Restore main.py
        if MAIN_PY_BACKUP.exists():
            shutil.copy2(MAIN_PY_BACKUP, MAIN_PY)
            print(f"  Restored main.py from backup")

    elapsed = time.time() - overall_start
    print(f"\n  Total time: {format_time(elapsed)}")

    print_summary(all_results)
    save_results(all_results)
    print(f"\n  Generating Pareto frontier plots...")
    generate_pareto_plot(all_results)

    print(f"\n{'#'*70}")
    print(f"  DONE — Lambda Sensitivity Analysis Complete")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
