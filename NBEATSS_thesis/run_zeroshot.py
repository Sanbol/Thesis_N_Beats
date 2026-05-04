"""
Zero-Shot Experiment Runner — N-BEATS-S Thesis
Loads M4 pre-trained checkpoints and directly tests on M3 (no fine-tuning).
This is "zero-shot transfer": the model never sees M3 during training.

Zero-shot adds a 5th condition to the thesis:
  A. Scratch Standard        - train on M3 from scratch, no stability
  B. Scratch Stabilized      - train on M3 from scratch, with stability
  C. TL Standard             - M4 pretrain -> M3 fine-tune, no stability
  D. TL Stabilized           - M4 pretrain -> M3 fine-tune, with stability
  E. ZeroShot Standard  (NEW)- M4 pretrain -> TEST on M3 directly (no fine-tune)
  F. ZeroShot Stabilized(NEW)- M4 pretrain -> TEST on M3 directly (no fine-tune)
"""

import subprocess
import sys
import os
import re
import csv
import shutil
import time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MAIN_PY      = os.path.join(SCRIPT_DIR, "main.py")
BACKUP_PY    = os.path.join(SCRIPT_DIR, "main_backup_zeroshot.py")
RESULTS_CSV  = os.path.join(SCRIPT_DIR, "experiment_results.csv")
ZEROSHOT_CSV = os.path.join(SCRIPT_DIR, "zeroshot_results.csv")

M4_PRETRAINED = {
    "Standard": [
        {"seed": 1, "model_id": "9gsv17gu"},
        {"seed": 2, "model_id": "9350xmaa"},
        {"seed": 3, "model_id": "4k98rjvf"},
    ],
    "Stabilized": [
        {"seed": 1, "model_id": "4wqfk2vw"},
        {"seed": 2, "model_id": "95i77vw3"},
        {"seed": 3, "model_id": "10gvfak4"},
    ],
}

SEEDS = [1, 2, 3]


def make_config(condition: str, seed: int, model_id: str) -> str:
    is_stabilized = condition == "Stabilized"
    lambda_stability = 0.02 if is_stabilized else 0.0
    ema_decay        = 0.99 if is_stabilized else 0.0

    return f"""
    # Dataset
    dataset = "M3"
    subset = "Monthly"
    dataset_id = "M3M"
    validation_periods = 18
    test_periods = 18
    test_mode_nrows = None

    # Model
    load_model = True
    update_loaded_model_specific_training_and_eval_hparams = True

    # Model architecture
    if load_model:
        model_id = "{model_id}"
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
    forecasting_origin_range_multiplier = 1e6
    batch_size = 32
    num_workers = 0
    if not load_model or update_loaded_model_specific_training_and_eval_hparams:
        lambda_stability = {lambda_stability}
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = 1e-5
        explr_gamma = 1.0
        ema_decay = {ema_decay}
    max_norm = 1.0
    batches_per_epoch = 50
    patience = 1e6
    max_epochs = 0  # ZERO SHOT: no training, just test

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    save_forecasts = False
    plot_forecasts = False
"""


def patch_main_py(config_str: str):
    with open(MAIN_PY, "r", encoding="utf-8") as f:
        content = f.read()

    start_marker = "##########################\n    # EXPERIMENT CONFIGURATION"
    end_marker   = "    ###################################################################################################"

    start_idx = content.find(start_marker)
    end_idx   = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find config markers in main.py")

    new_content = (
        content[:start_idx]
        + "##########################\n    # EXPERIMENT CONFIGURATION\n    ##########################"
        + config_str
        + "\n    "
        + content[end_idx:]
    )

    with open(MAIN_PY, "w", encoding="utf-8") as f:
        f.write(new_content)


def get_latest_wandb_run():
    wandb_dir = os.path.join(SCRIPT_DIR, "wandb")
    if not os.path.exists(wandb_dir):
        return None
    runs = [d for d in os.listdir(wandb_dir) if d.startswith("offline-run-")]
    if not runs:
        return None
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(wandb_dir, d)), reverse=True)
    return os.path.join(wandb_dir, runs[0])


def parse_metrics(run_dir: str) -> dict:
    metrics = {}
    log_path = os.path.join(run_dir, "logs", "debug.log")
    if not os.path.exists(log_path):
        log_path = os.path.join(run_dir, "logs", "debug-internal.log")
    if not os.path.exists(log_path):
        return metrics

    with open(log_path, "r", errors="ignore") as f:
        content = f.read()

    for key, pattern in [
        ("sMAPE",  r"test_sMAPE['\"]?\s*[:\|]\s*([\d.]+)"),
        ("RMSSE",  r"test_RMSSE['\"]?\s*[:\|]\s*([\d.]+)"),
        ("RMSSC",  r"test_RMSSC['\"]?\s*[:\|]\s*([\d.]+)"),
        ("sMAPC",  r"test_sMAPC['\"]?\s*[:\|]\s*([\d.]+)"),
    ]:
        m = re.search(pattern, content)
        if m:
            metrics[key] = float(m.group(1))

    return metrics


def run_experiment(exp_name: str, condition: str, seed: int, model_id: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running: {exp_name}  (seed={seed}, model_id={model_id})")
    print(f"{'='*60}")

    before_run = time.time()
    patch_main_py(make_config(condition, seed, model_id))

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, MAIN_PY],
        capture_output=True, cwd=SCRIPT_DIR,
        env=env, encoding="utf-8", errors="replace"
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    print(stdout[-3000:] if len(stdout) > 3000 else stdout)
    if result.returncode != 0:
        print("STDERR:", stderr[-2000:])

    wandb_dir = os.path.join(SCRIPT_DIR, "wandb")
    if os.path.exists(wandb_dir):
        new_runs = [
            d for d in os.listdir(wandb_dir)
            if d.startswith("offline-run-")
            and os.path.getmtime(os.path.join(wandb_dir, d)) >= before_run
        ]
        new_runs.sort(key=lambda d: os.path.getmtime(os.path.join(wandb_dir, d)), reverse=True)
        run_dir = os.path.join(wandb_dir, new_runs[0]) if new_runs else None
    else:
        run_dir = None

    run_id = ""
    if run_dir:
        match = re.search(r"offline-run-\d+_\d+-(\w+)$", os.path.basename(run_dir))
        if match:
            run_id = match.group(1)

    # Try to parse metrics from stdout directly first (faster)
    metrics = {}
    for key, pattern in [
        ("sMAPE", r"test_sMAPE['\"]?\s*[:\|]\s*([\d.]+)"),
        ("RMSSE", r"test_RMSSE['\"]?\s*[:\|]\s*([\d.]+)"),
        ("RMSSC", r"test_RMSSC['\"]?\s*[:\|]\s*([\d.]+)"),
        ("sMAPC", r"test_sMAPC['\"]?\s*[:\|]\s*([\d.]+)"),
    ]:
        m = re.search(pattern, stdout + stderr)
        if m:
            metrics[key] = float(m.group(1))

    if not metrics and run_dir:
        metrics = parse_metrics(run_dir)

    return {
        "experiment": exp_name,
        "condition":  f"ZeroShot_{condition}",
        "seed":       seed,
        "source_model_id": model_id,
        "run_id":     run_id,
        "sMAPE":      metrics.get("sMAPE", ""),
        "RMSSE":      metrics.get("RMSSE", ""),
        "RMSSC":      metrics.get("RMSSC", ""),
        "sMAPC":      metrics.get("sMAPC", ""),
    }


def main():
    print("N-BEATS-S Zero-Shot Experiments")
    print("Loads M4 pre-trained checkpoints -> tests on M3 with NO fine-tuning\n")

    shutil.copy(MAIN_PY, BACKUP_PY)
    print(f"Backed up main.py -> {BACKUP_PY}")

    all_results = []

    fieldnames = ["experiment", "condition", "seed", "source_model_id",
                  "run_id", "sMAPE", "RMSSE", "RMSSC", "sMAPC"]
    with open(ZEROSHOT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    try:
        for condition, entries in M4_PRETRAINED.items():
            for entry in entries:
                seed     = entry["seed"]
                model_id = entry["model_id"]
                exp_name = f"E_ZeroShot_{condition}_seed{seed}"

                row = run_experiment(exp_name, condition, seed, model_id)
                all_results.append(row)

                # Save after each run (crash-safe)
                with open(ZEROSHOT_CSV, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row)

                print(f"\n  >> sMAPE: {row.get('sMAPE', 'N/A')}  RMSSC: {row.get('RMSSC', 'N/A')}")

    finally:
        shutil.copy(BACKUP_PY, MAIN_PY)
        print(f"\nRestored main.py from backup.")

    print("\n" + "="*65)
    print("ZERO-SHOT RESULTS SUMMARY")
    print("="*65)
    print(f"{'Condition':<30} {'sMAPE':>8} {'RMSSE':>8} {'RMSSC':>8}")
    print("-"*65)

    import statistics
    for condition in ["Standard", "Stabilized"]:
        cond_rows = [r for r in all_results if r["condition"] == f"ZeroShot_{condition}"]
        smapes = [float(r["sMAPE"]) for r in cond_rows if r["sMAPE"] != ""]
        rmsses = [float(r["RMSSE"]) for r in cond_rows if r["RMSSE"] != ""]
        rmsscs = [float(r["RMSSC"]) for r in cond_rows if r["RMSSC"] != ""]
        if smapes:
            print(f"  ZeroShot {condition:<20} "
                  f"{statistics.mean(smapes):>7.3f}±{statistics.stdev(smapes):.3f}  "
                  f"{statistics.mean(rmsses):>7.3f}±{statistics.stdev(rmsses):.3f}  "
                  f"{statistics.mean(rmsscs):>7.3f}±{statistics.stdev(rmsscs):.3f}")

    print(f"\nResults saved to: {ZEROSHOT_CSV}")
    print("\nContext: Fine-tuned results for comparison:")
    print("  TL Standard:   sMAPE 13.02% ± 0.06,  RMSSC 0.326 ± 0.015")
    print("  TL Stabilized: sMAPE 13.01% ± 0.01,  RMSSC 0.310 ± 0.011")
    print(
        "\nIf zero-shot sMAPE >> 13%, fine-tuning is essential.\n"
        "If zero-shot sMAPE ~= 13%, the model transfers well without any adaptation."
    )


if __name__ == "__main__":
    main()
