"""
EXHAUSTIVE VERIFICATION OF BOTH TABLE IMAGES
=============================================
Computes every single number to 10+ decimal places, then rounds to table precision.
Checks: means, stds, deltas, percentages.
"""
import csv
import numpy as np

np.set_printoptions(precision=10)

# ============================================================
# N-BEATS-S
# ============================================================
print("=" * 80)
print("N-BEATS-S EXHAUSTIVE VERIFICATION")
print("=" * 80)

with open('experiment_results_tuned.csv') as f:
    nbeats = list(csv.DictReader(f))

nbeats_conditions = {
    'Scratch_Standard': [r for r in nbeats if r['condition'] == 'Scratch_Standard'],
    'Scratch_Stabilized': [r for r in nbeats if r['condition'] == 'Scratch_Stabilized'],
    'TL_Standard': [r for r in nbeats if r['condition'] == 'TL_Standard'],
    'TL_Stabilized': [r for r in nbeats if r['condition'] == 'TL_Stabilized'],
    'ZeroShot_Standard': [r for r in nbeats if r['condition'] == 'ZeroShot_Standard'],
    'ZeroShot_Stabilized': [r for r in nbeats if r['condition'] == 'ZeroShot_Stabilized'],
}

metrics = ['sMAPE', 'RMSSE', 'RMSSC', 'sMAPC']

print("\nRAW VALUES (all seeds):")
for cond, rows in nbeats_conditions.items():
    print(f"\n  {cond} ({len(rows)} seeds):")
    for r in rows:
        print(f"    seed={r['seed']}  sMAPE={float(r['sMAPE']):.10f}  RMSSE={float(r['RMSSE']):.10f}  RMSSC={float(r['RMSSC']):.10f}  sMAPC={float(r['sMAPC']):.10f}")

print("\n\nCOMPUTED MEANS AND STDS (population std, ddof=0):")
nbeats_means = {}
nbeats_stds = {}
for cond, rows in nbeats_conditions.items():
    nbeats_means[cond] = {}
    nbeats_stds[cond] = {}
    for m in metrics:
        vals = [float(r[m]) for r in rows]
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        nbeats_means[cond][m] = mean
        nbeats_stds[cond][m] = std
        rounded_mean = round(mean, 2)
        rounded_std = round(std, 2)
        print(f"  {cond:25s} {m}: mean={mean:.10f} -> {rounded_mean:.2f}  std={std:.10f} -> {rounded_std:.2f}")

# Expected table values for N-BEATS-S
nbeats_expected = {
    'Scratch_Standard':    {'sMAPE': '12.80 ± 0.49', 'RMSSE': '1.38 ± 0.11', 'RMSSC': '0.51 ± 0.04', 'sMAPC': '5.59 ± 0.32'},
    'Scratch_Stabilized':  {'sMAPE': '12.49 ± 0.16', 'RMSSE': '1.24 ± 0.01', 'RMSSC': '0.30 ± 0.01', 'sMAPC': '3.80 ± 0.04'},
    'TL_Standard':         {'sMAPE': '12.35 ± 0.08', 'RMSSE': '1.17 ± 0.01', 'RMSSC': '0.40 ± 0.01', 'sMAPC': '5.55 ± 0.13'},
    'TL_Stabilized':       {'sMAPE': '12.30 ± 0.04', 'RMSSE': '1.17 ± 0.01', 'RMSSC': '0.38 ± 0.01', 'sMAPC': '5.30 ± 0.16'},
    'ZeroShot_Standard':   {'sMAPE': '12.77 ± 0.24', 'RMSSE': '1.25 ± 0.02', 'RMSSC': '0.47 ± 0.04', 'sMAPC': '6.02 ± 0.39'},
    'ZeroShot_Stabilized': {'sMAPE': '12.44 ± 0.04', 'RMSSE': '1.27 ± 0.01', 'RMSSC': '0.33 ± 0.01', 'sMAPC': '4.60 ± 0.10'},
}

print("\n\nVERIFICATION vs TABLE VALUES (N-BEATS-S):")
errors_nbeats = []
for cond in nbeats_expected:
    for m in metrics:
        computed = f"{nbeats_means[cond][m]:.2f} ± {nbeats_stds[cond][m]:.2f}"
        expected = nbeats_expected[cond][m]
        match = "OK" if computed == expected else "MISMATCH"
        if match != "OK":
            errors_nbeats.append(f"  {cond} {m}: computed={computed} vs table={expected}")
        print(f"  {cond:25s} {m}: computed={computed}  table={expected}  -> {match}")

# Verify deltas
print("\n\nDELTA VERIFICATION (N-BEATS-S):")
nbeats_delta_expected = {
    'Scratch':  {'sMAPE': '−0.30 (−2.4%)', 'RMSSE': '−0.14 (−9.9%)', 'RMSSC': '−0.21 (−40.7%)', 'sMAPC': '−1.79 (−32.0%)'},
    'TL':       {'sMAPE': '−0.05 (−0.4%)', 'RMSSE': '+0.00 (+0.1%)', 'RMSSC': '−0.02 (−4.7%)', 'sMAPC': '−0.25 (−4.6%)'},
    'ZeroShot': {'sMAPE': '−0.33 (−2.6%)', 'RMSSE': '+0.01 (+1.2%)', 'RMSSC': '−0.13 (−28.5%)', 'sMAPC': '−1.43 (−23.7%)'},
}

pairs = [('Scratch', 'Scratch_Standard', 'Scratch_Stabilized'),
         ('TL', 'TL_Standard', 'TL_Stabilized'),
         ('ZeroShot', 'ZeroShot_Standard', 'ZeroShot_Stabilized')]

for label, std_c, stab_c in pairs:
    for m in metrics:
        delta = nbeats_means[stab_c][m] - nbeats_means[std_c][m]
        pct = (delta / nbeats_means[std_c][m]) * 100

        # Format like the table
        sign_d = '+' if delta >= 0 else '−'
        sign_p = '+' if pct >= 0 else '−'
        abs_d = abs(delta)
        abs_p = abs(pct)
        computed = f"{sign_d}{abs_d:.2f} ({sign_p}{abs_p:.1f}%)"
        expected = nbeats_delta_expected[label][m]

        match = "OK" if computed == expected else "MISMATCH"
        if match != "OK":
            errors_nbeats.append(f"  DELTA {label} {m}: computed={computed} vs table={expected}")
        print(f"  {label:10s} {m}: delta={delta:+.10f} pct={pct:+.10f}%  -> computed={computed}  table={expected}  -> {match}")

print(f"\n  N-BEATS-S ERRORS: {len(errors_nbeats)}")
for e in errors_nbeats:
    print(e)

# ============================================================
# N-HiTS-S
# ============================================================
print("\n\n" + "=" * 80)
print("N-HiTS-S EXHAUSTIVE VERIFICATION")
print("=" * 80)

with open('nhits_experiment_results_tuned.csv') as f:
    nhits = list(csv.DictReader(f))

nhits_conditions = {
    'Scratch_Standard': [r for r in nhits if r['condition'] == 'NHITS_Scratch_Standard'],
    'Scratch_Stabilized': [r for r in nhits if r['condition'] == 'NHITS_Scratch_Stabilized'],
    'TL_Standard': [r for r in nhits if r['condition'] == 'NHITS_TL_Standard'],
    'TL_Stabilized': [r for r in nhits if r['condition'] == 'NHITS_TL_Stabilized'],
    'ZeroShot_Standard': [r for r in nhits if r['condition'] == 'NHITS_ZeroShot_Standard'],
    'ZeroShot_Stabilized': [r for r in nhits if r['condition'] == 'NHITS_ZeroShot_Stabilized'],
}

print("\nRAW VALUES (all seeds):")
for cond, rows in nhits_conditions.items():
    print(f"\n  {cond} ({len(rows)} seeds):")
    for r in rows:
        print(f"    seed={r['seed']}  sMAPE={float(r['sMAPE']):.10f}  RMSSE={float(r['RMSSE']):.10f}  RMSSC={float(r['RMSSC']):.10f}  sMAPC={float(r['sMAPC']):.10f}")

print("\n\nCOMPUTED MEANS AND STDS (population std, ddof=0):")
nhits_means = {}
nhits_stds = {}
for cond, rows in nhits_conditions.items():
    nhits_means[cond] = {}
    nhits_stds[cond] = {}
    for m in metrics:
        vals = [float(r[m]) for r in rows]
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        nhits_means[cond][m] = mean
        nhits_stds[cond][m] = std
        rounded_mean = round(mean, 2)
        rounded_std = round(std, 2)
        print(f"  {cond:25s} {m}: mean={mean:.10f} -> {rounded_mean:.2f}  std={std:.10f} -> {rounded_std:.2f}")

nhits_expected = {
    'Scratch_Standard':    {'sMAPE': '13.05 ± 0.08', 'RMSSE': '1.33 ± 0.04', 'RMSSC': '0.46 ± 0.04', 'sMAPC': '5.74 ± 0.62'},
    'Scratch_Stabilized':  {'sMAPE': '13.26 ± 0.32', 'RMSSE': '1.33 ± 0.06', 'RMSSC': '0.34 ± 0.03', 'sMAPC': '4.48 ± 0.51'},
    'TL_Standard':         {'sMAPE': '13.30 ± 0.19', 'RMSSE': '1.21 ± 0.01', 'RMSSC': '0.52 ± 0.02', 'sMAPC': '7.78 ± 0.51'},
    'TL_Stabilized':       {'sMAPE': '13.12 ± 0.17', 'RMSSE': '1.22 ± 0.01', 'RMSSC': '0.48 ± 0.02', 'sMAPC': '6.89 ± 0.36'},
    'ZeroShot_Standard':   {'sMAPE': '13.99 ± 0.12', 'RMSSE': '1.33 ± 0.06', 'RMSSC': '0.54 ± 0.02', 'sMAPC': '7.52 ± 0.47'},
    'ZeroShot_Stabilized': {'sMAPE': '13.41 ± 0.09', 'RMSSE': '1.32 ± 0.02', 'RMSSC': '0.42 ± 0.03', 'sMAPC': '5.84 ± 0.64'},
}

print("\n\nVERIFICATION vs TABLE VALUES (N-HiTS-S):")
errors_nhits = []
for cond in nhits_expected:
    for m in metrics:
        computed = f"{nhits_means[cond][m]:.2f} ± {nhits_stds[cond][m]:.2f}"
        expected = nhits_expected[cond][m]
        match = "OK" if computed == expected else "MISMATCH"
        if match != "OK":
            errors_nhits.append(f"  {cond} {m}: computed={computed} vs table={expected}")
        print(f"  {cond:25s} {m}: computed={computed}  table={expected}  -> {match}")

# Verify deltas
print("\n\nDELTA VERIFICATION (N-HiTS-S):")
nhits_delta_expected = {
    'Scratch':  {'sMAPE': '+0.21 (+1.6%)', 'RMSSE': '−0.00 (−0.2%)', 'RMSSC': '−0.12 (−25.9%)', 'sMAPC': '−1.25 (−21.8%)'},
    'TL':       {'sMAPE': '−0.18 (−1.3%)', 'RMSSE': '+0.01 (+0.8%)', 'RMSSC': '−0.05 (−8.7%)', 'sMAPC': '−0.89 (−11.4%)'},
    'ZeroShot': {'sMAPE': '−0.58 (−4.1%)', 'RMSSE': '−0.01 (−0.4%)', 'RMSSC': '−0.12 (−22.2%)', 'sMAPC': '−1.68 (−22.4%)'},
}

pairs_nhits = [('Scratch', 'Scratch_Standard', 'Scratch_Stabilized'),
               ('TL', 'TL_Standard', 'TL_Stabilized'),
               ('ZeroShot', 'ZeroShot_Standard', 'ZeroShot_Stabilized')]

for label, std_c, stab_c in pairs_nhits:
    for m in metrics:
        delta = nhits_means[stab_c][m] - nhits_means[std_c][m]
        pct = (delta / nhits_means[std_c][m]) * 100

        sign_d = '+' if delta >= 0 else '−'
        sign_p = '+' if pct >= 0 else '−'
        abs_d = abs(delta)
        abs_p = abs(pct)
        computed = f"{sign_d}{abs_d:.2f} ({sign_p}{abs_p:.1f}%)"
        expected = nhits_delta_expected[label][m]

        match = "OK" if computed == expected else "MISMATCH"
        if match != "OK":
            errors_nhits.append(f"  DELTA {label} {m}: computed={computed} vs table={expected}")
        print(f"  {label:10s} {m}: delta={delta:+.10f} pct={pct:+.10f}%  -> computed={computed}  table={expected}  -> {match}")

print(f"\n  N-HiTS-S ERRORS: {len(errors_nhits)}")
for e in errors_nhits:
    print(e)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
total_errors = len(errors_nbeats) + len(errors_nhits)
if total_errors == 0:
    print("ALL DATA POINTS VERIFIED — ZERO ERRORS")
else:
    print(f"TOTAL ERRORS FOUND: {total_errors}")
    print("\nN-BEATS-S errors:")
    for e in errors_nbeats:
        print(e)
    print("\nN-HiTS-S errors:")
    for e in errors_nhits:
        print(e)
