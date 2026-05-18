#!/usr/bin/env python3
"""Auto-pick winners from pre-flight CSVs and update runner configs.
Halts pipeline (exit 1) if no valid winner can be picked from CSVs."""
import csv, re, sys
from pathlib import Path

BASE = Path(__file__).parent

def read_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def safe_float(s, default=float('inf')):
    try: return float(s)
    except: return default

print("=" * 60)
print("Reading pre-flight results...")
print("=" * 60)

# Pick lambda winner (lowest val_sMAPE among stabilized configs)
lambda_rows = read_csv(BASE / "lambda_recheck_results.csv")
if not lambda_rows:
    print("\n  ERROR: lambda_recheck_results.csv is missing or empty.")
    print("  Halting pipeline. Investigate run_lambda_recheck.py output.")
    sys.exit(1)

candidates = [(safe_float(r.get('lambda_stability', 0)),
               safe_float(r.get('sMAPE'))) for r in lambda_rows]
stabilized = [(l, s) for l, s in candidates if l > 0 and s != float('inf')]

if not stabilized:
    print("\n  ERROR: No valid stabilized lambda candidates in CSV.")
    print(f"  Raw rows: {lambda_rows}")
    print("  Halting pipeline.")
    sys.exit(1)

lambda_winner, best_smape = min(stabilized, key=lambda x: x[1])
print(f"\n  Lambda winner: {lambda_winner} (val_sMAPE={best_smape:.4f})")

# Show the full ranking for transparency
print("  Full lambda ranking by val_sMAPE:")
for l, s in sorted(candidates, key=lambda x: x[1]):
    marker = " <-- WINNER" if l == lambda_winner else ""
    standard = " (Standard baseline)" if l == 0 else ""
    print(f"    lambda={l}: val_sMAPE={s:.4f}{marker}{standard}")

# Pick N-HiTS LR winner (lowest val_sMAPE)
lr_rows = read_csv(BASE / "nhits_lr_tuning_results.csv")
if not lr_rows:
    print("\n  ERROR: nhits_lr_tuning_results.csv is missing or empty.")
    sys.exit(1)

cands = [(r.get('learning_rate', '1e-3'),
          safe_float(r.get('sMAPE'))) for r in lr_rows]
valid = [(l, s) for l, s in cands if s != float('inf')]
if not valid:
    print("\n  ERROR: No valid N-HiTS LR candidates.")
    sys.exit(1)

lr_winner, best_lr_smape = min(valid, key=lambda x: x[1])
print(f"\n  N-HiTS LR winner: {lr_winner} (val_sMAPE={best_lr_smape:.4f})")
print("  Full LR ranking by val_sMAPE:")
for l, s in sorted(valid, key=lambda x: x[1]):
    marker = " <-- WINNER" if l == lr_winner else ""
    print(f"    lr={l}: val_sMAPE={s:.4f}{marker}")

# Update N-BEATS runner: lambda only
nbeats = BASE / "run_experiments_tuned.py"
text = nbeats.read_text()
text = re.sub(r'"lambda_stability":\s*0\.02',
              f'"lambda_stability": {lambda_winner}', text)
nbeats.write_text(text)
print(f"\n  Updated run_experiments_tuned.py: lambda={lambda_winner}")

# Update N-HiTS runner: lambda AND scratch/pretrain LR
nhits = BASE / "run_nhits_experiments_tuned.py"
text = nhits.read_text()
text = re.sub(r'"lambda_stability":\s*0\.02',
              f'"lambda_stability": {lambda_winner}', text)
text = re.sub(r'"learning_rate":\s*"1e-3"',
              f'"learning_rate": "{lr_winner}"', text)
nhits.write_text(text)
print(f"  Updated run_nhits_experiments_tuned.py: lambda={lambda_winner}, lr={lr_winner}")

print("\n  Done. Main runs will use these values.")
print("=" * 60)
