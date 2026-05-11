#!/usr/bin/env python3
"""
Recover experiment results from wandb offline binary files.
Extracts test metrics (sMAPE, RMSSE, RMSSC, sMAPC) for each run.
"""
import os, json, csv
from pathlib import Path

WANDB_DIR = Path("~/Thesis_N_Beats/NBEATSS_thesis/wandb").expanduser()

# Map run IDs to experiment metadata (from pipeline.log)
# N-BEATS runs (Stage 5)
NBEATS_RUNS = {
    "zraud12i": ("A_Scratch_Standard_seed1",    "Scratch_Standard",    1),
    "fffhryej": ("B_Scratch_Stabilized_seed1",  "Scratch_Stabilized",  1),
    "4lf1k5v9": ("C_TL_Standard_pretrain_seed1","TL_Standard_pretrain",1),
    "5e64f54n": ("D_TL_Stabilized_pretrain_seed1","TL_Stabilized_pretrain",1),
    "5emfx9rz": ("C_TL_Standard_finetune_seed1","TL_Standard_finetune",1),
    "311ywt1t": ("D_TL_Stabilized_finetune_seed1","TL_Stabilized_finetune",1),
    "d2kjmcuk": ("E_ZeroShot_Standard_seed1",   "ZeroShot_Standard",   1),
    "38hr9l3q": ("E_ZeroShot_Stabilized_seed1", "ZeroShot_Stabilized", 1),
}

# N-HiTS runs (Stage 6) - fill in run IDs from wandb listing
NHITS_RUNS = {
    "2u03kymc": ("NHITS_A_Scratch_Standard_seed1",    "Scratch_Standard",    1),
    "v88b9jqe": ("NHITS_B_Scratch_Stabilized_seed1",  "Scratch_Stabilized",  1),
    "v0hrzes0": ("NHITS_C_TL_Standard_pretrain_seed1","TL_Standard_pretrain",1),
    "5gl0atq7": ("NHITS_D_TL_Stabilized_pretrain_seed1","TL_Stabilized_pretrain",1),
    # Last 4 N-HiTS runs - IDs unknown, will auto-detect below
}

def find_run_dir(run_id):
    matches = list(WANDB_DIR.glob(f"offline-run-*-{run_id}"))
    return matches[0] if matches else None

def extract_metrics(run_id):
    """Try multiple sources to get test metrics for a run."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        return None

    # Source 1: wandb-summary.json (created if wandb.finish() was called)
    summary = run_dir / "files" / "wandb-summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        smape = data.get("sMAPE") or data.get("test/sMAPE")
        rmsse = data.get("RMSSE") or data.get("test/RMSSE")
        rmssc = data.get("RMSSC") or data.get("test/RMSSC")
        smapc = data.get("sMAPC") or data.get("test/sMAPC")
        if smape:
            return {"sMAPE": smape, "RMSSE": rmsse, "RMSSC": rmssc, "sMAPC": smapc,
                    "source": "summary.json"}

    # Source 2: output.log (captured stdout from training)
    output = run_dir / "files" / "output.log"
    if output.exists():
        text = output.read_text()
        metrics = {}
        for line in text.splitlines():
            for metric in ["sMAPE", "RMSSE", "RMSSC", "sMAPC"]:
                if metric in line and "│" in line:
                    parts = [p.strip() for p in line.split("│") if p.strip()]
                    if len(parts) >= 2:
                        try:
                            metrics[metric] = float(parts[-1])
                        except:
                            pass
        if metrics:
            return {**metrics, "source": "output.log"}

    return {"source": "NOT_FOUND", "sMAPE": None, "RMSSE": None,
            "RMSSC": None, "sMAPC": None}

def write_csv(runs_dict, output_file):
    fieldnames = ["experiment", "condition", "seed", "run_id",
                  "sMAPE", "RMSSE", "RMSSC", "sMAPC", "source"]
    rows = []
    for run_id, (experiment, condition, seed) in runs_dict.items():
        metrics = extract_metrics(run_id) or {}
        rows.append({
            "experiment": experiment,
            "condition": condition,
            "seed": seed,
            "run_id": run_id,
            "sMAPE": metrics.get("sMAPE", ""),
            "RMSSE": metrics.get("RMSSE", ""),
            "RMSSC": metrics.get("RMSSC", ""),
            "sMAPC": metrics.get("sMAPC", ""),
            "source": metrics.get("source", ""),
        })
        status = "✓" if metrics.get("sMAPE") else "✗ MISSING"
        print(f"  {status} {experiment}: sMAPE={metrics.get('sMAPE', 'N/A')}")

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Written to {output_file}")

# Auto-detect remaining N-HiTS run IDs from wandb dir
print("Auto-detecting N-HiTS Stage 6 run IDs...")
all_runs = sorted(WANDB_DIR.glob("offline-run-*"), key=os.path.getmtime)
known_ids = set(list(NBEATS_RUNS.keys()) + list(NHITS_RUNS.keys()))
# Pre-flight IDs to exclude
preflight_ids = {"fa4mdcj9","bbqh1eq7","5qbin6lz","cd7s4pvf","sq766yqq",
                 "htrrkfns","plycsuz0","tzkqoww0","b0w3ajqf","byafbced"}
nhits_phase_names = [
    ("NHITS_C_TL_Standard_finetune_seed1",   "TL_Standard_finetune",   1),
    ("NHITS_D_TL_Stabilized_finetune_seed1", "TL_Stabilized_finetune", 1),
    ("NHITS_E_ZeroShot_Standard_seed1",      "ZeroShot_Standard",      1),
    ("NHITS_E_ZeroShot_Stabilized_seed1",    "ZeroShot_Stabilized",    1),
]
unknown_runs = [r for r in all_runs
                if r.name.split("-")[-1] not in known_ids
                and r.name.split("-")[-1] not in preflight_ids]
for i, (run_dir, meta) in enumerate(zip(unknown_runs, nhits_phase_names)):
    run_id = run_dir.name.split("-")[-1]
    NHITS_RUNS[run_id] = meta
    print(f"  Auto-detected: {run_id} → {meta[0]}")

print("\n=== Recovering N-BEATS-S results ===")
write_csv(NBEATS_RUNS, "experiment_results_tuned_RECOVERED.csv")

print("\n=== Recovering N-HiTS-S results ===")
write_csv(NHITS_RUNS, "nhits_experiment_results_tuned_RECOVERED.csv")

print("\nDone. Check the RECOVERED csv files.")
