#!/usr/bin/env python3
"""
Recover test metrics from pipeline.log.
Lightning prints test metric tables to stdout, captured by tee.
"""
import re, csv, os
from pathlib import Path

LOG = Path("~/Thesis_N_Beats/NBEATSS_thesis/pipeline.log").expanduser()
text = LOG.read_text()

# Find all RUNNING blocks with their test metrics
pattern = r'RUNNING:\s+(\S+).*?Run ID:\s+(\S+)'
runs = list(re.finditer(pattern, text, re.DOTALL))

# Find all test metric tables (Lightning format)
metric_pattern = r'│\s+(sMAPE|RMSSE|RMSSC|sMAPC)\s+│\s+([0-9.]+)\s+│'
tables = list(re.finditer(
    r'(┏.*?Test metric.*?┗.*?┛)', text, re.DOTALL
))

print(f"Found {len(runs)} RUNNING entries")
print(f"Found {len(tables)} test metric tables")
print()

# Associate each test table with the nearest preceding RUNNING entry
results = []
for table in tables:
    table_pos = table.start()
    # Find the closest RUNNING entry before this table
    closest_run = None
    for run in runs:
        if run.start() < table_pos:
            closest_run = run
    if not closest_run:
        continue

    experiment = closest_run.group(1)
    run_id = closest_run.group(2)

    # Extract metrics from the table
    metrics = {}
    for m in re.finditer(metric_pattern, table.group(1)):
        metrics[m.group(1)] = float(m.group(2))

    if metrics:
        results.append({
            "experiment": experiment,
            "run_id": run_id,
            "sMAPE": metrics.get("sMAPE", ""),
            "RMSSE": metrics.get("RMSSE", ""),
            "RMSSC": metrics.get("RMSSC", ""),
            "sMAPC": metrics.get("sMAPC", ""),
        })
        print(f"  ✓ {experiment}: sMAPE={metrics.get('sMAPE','?'):.4f}")

# Split into N-BEATS and N-HiTS
nbeats = [r for r in results if not r["experiment"].startswith("NHITS")]
nhits = [r for r in results if r["experiment"].startswith("NHITS")]

# Also grab pre-flight runs (lambda, LR, clip)
preflight = [r for r in results if r["experiment"].startswith(("Lambda","NHITS_LR","clip"))]

# Create output directory
out_dir = Path("~/Thesis_N_Beats/NBEATSS_thesis/results_final").expanduser()
out_dir.mkdir(exist_ok=True)

fieldnames = ["experiment", "run_id", "sMAPE", "RMSSE", "RMSSC", "sMAPC"]

def write(data, filename):
    path = out_dir / filename
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(data)
    print(f"\n  → {path} ({len(data)} rows)")

if nbeats:
    write(nbeats, "nbeats_results.csv")
if nhits:
    write(nhits, "nhits_results.csv")
if preflight:
    write(preflight, "preflight_results.csv")

# Copy pipeline.log and other useful files to results_final
import shutil
for f in ["pipeline.log", "lambda_sensitivity_results.csv",
          "lambda_pareto_frontier.png"]:
    src = Path("~/Thesis_N_Beats/NBEATSS_thesis").expanduser() / f
    if src.exists():
        shutil.copy(src, out_dir / f)
        print(f"  Copied: {f}")

print(f"\n{'='*60}")
print(f"All results in: {out_dir}")
print(f"{'='*60}")
