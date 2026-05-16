"""
One-off diagnostic run: validation loss curves for N-BEATS and N-HiTS
Scratch Stabilized. All params match run_*_experiments_tuned.py
except eval_mode=validation and max_epochs (configurable).
"""
import subprocess, os, sys, re, time, shutil
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
PYTHON_EXE = sys.executable

RUN_NBEATS = True
RUN_NHITS = True
MAX_EPOCHS = 155

TEMPLATE_NBEATS = '''    ##########################
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
    random_seed = 1
    ## Data hparams
    forecasting_origin_range_multiplier = 1e6
    batch_size = 512
    num_workers = 0
    ## Model-specific training and evaluation hparams
    if not load_model or update_loaded_model_specific_training_and_eval_hparams:
        lambda_stability = 0.15
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = 1e-3
        explr_gamma = 1.0
        ema_decay = 0.0
    ## Trainer hparams
    max_norm = 1.0
    batches_per_epoch = 93
    patience = 1000000
    max_epochs = {max_epochs}

    # Other
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    save_forecasts = False
    plot_forecasts = False
'''

TEMPLATE_NHITS = '''    ##########################
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
        backcast_length_multiplier = 6
        forecast_length = 6
        hidden_layer_units = 512
        n_blocks = 10
        n_blocks_shared = 1
        ensemble_size = 1
        zero_mean = True
        unit_variance = True

    # Model training and evaluation
    eval_mode = 'validation'
    random_seed = 1
    ## Data hparams
    forecasting_origin_range_multiplier = 1e6
    batch_size = 512
    num_workers = 0
    ## Model-specific training and evaluation hparams
    if not load_model or update_loaded_model_specific_training_and_eval_hparams:
        lambda_stability = 0.15
        enforce_nonnegative_forecast_metric_calculation = True
        learning_rate = 1e-3
        explr_gamma = 1.0
        ema_decay = 0.0
    ## Trainer hparams
    max_norm = 1.0
    batches_per_epoch = 93
    patience = 1000000
    max_epochs = {max_epochs}

    # Other
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    save_forecasts = False
    plot_forecasts = False
'''

def patch_and_run(main_file, backup_file, template, max_epochs, label):
    print(f"\n{'='*70}\n  RUNNING: {label}  (eval_mode='validation', epochs={max_epochs})\n{'='*70}", flush=True)
    shutil.copy2(main_file, backup_file)
    print(f"  Backed up {main_file.name} -> {backup_file.name}", flush=True)
    content = main_file.read_text(encoding='utf-8')
    start_marker = "    ##########################\n    # EXPERIMENT CONFIGURATION\n    ##########################"
    end_marker = "\n    ###################################################################################################"
    si = content.find(start_marker); ei = content.find(end_marker)
    if si == -1 or ei == -1:
        print(f"  ERROR: could not find config markers in {main_file.name}", flush=True)
        return
    new_config = template.format(max_epochs=max_epochs)
    main_file.write_text(content[:si] + new_config + content[ei:], encoding='utf-8')
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    try:
        r = subprocess.run([PYTHON_EXE, str(main_file)], cwd=str(BASE_DIR),
                           env=env, capture_output=True, encoding="utf-8",
                           errors="replace", timeout=14400)
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"  ERROR exit {r.returncode}\n  STDOUT tail: {r.stdout[-2000:]}\n  STDERR tail: {r.stderr[-2000:]}", flush=True)
        else:
            print(f"  Completed in {int(dt//60)}m {int(dt%60)}s", flush=True)
            wd = BASE_DIR / "wandb"
            if wd.exists():
                latest = sorted([d for d in wd.iterdir() if d.name.startswith("offline-run-")],
                                key=lambda p: p.stat().st_mtime)[-1:]
                if latest:
                    rid = re.search(r'-([a-z0-9]+)$', latest[0].name)
                    print(f"  Run ID: {rid.group(1) if rid else '?'}", flush=True)
    finally:
        shutil.copy2(backup_file, main_file)
        print(f"  Restored {main_file.name} from backup", flush=True)

def main():
    t_total = time.time()
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    if RUN_NBEATS:
        patch_and_run(BASE_DIR / "main.py",
                      BASE_DIR / "main_validation_backup.py",
                      TEMPLATE_NBEATS, MAX_EPOCHS, "N-BEATS Scratch Stabilized (val mode)")
    if RUN_NHITS:
        patch_and_run(BASE_DIR / "main_nhits.py",
                      BASE_DIR / "main_nhits_validation_backup.py",
                      TEMPLATE_NHITS, MAX_EPOCHS, "N-HiTS Scratch Stabilized (val mode)")
    print(f"\nALL DONE in {int((time.time()-t_total)//60)}m at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

if __name__ == "__main__":
    main()