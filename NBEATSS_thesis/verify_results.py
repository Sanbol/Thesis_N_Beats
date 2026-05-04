import csv, statistics

# === N-BEATS ===
with open('experiment_results_tuned.csv') as f:
    rows = list(csv.DictReader(f))

def mean3(rows, cond, col):
    vals = [float(r[col]) for r in rows if r['condition'] == cond]
    return statistics.mean(vals)

def std3(rows, cond, col):
    vals = [float(r[col]) for r in rows if r['condition'] == cond]
    return statistics.stdev(vals)

print("=== TABLE 5 DELTAS (N-BEATS) ===")
for std_c, stab_c, label in [('Scratch_Standard','Scratch_Stabilized','Scratch'),
                               ('TL_Standard','TL_Stabilized','FT-TL'),
                               ('ZeroShot_Standard','ZeroShot_Stabilized','ZeroShot')]:
    ds = mean3(rows, stab_c, 'sMAPE') - mean3(rows, std_c, 'sMAPE')
    dr = mean3(rows, stab_c, 'RMSSC') - mean3(rows, std_c, 'RMSSC')
    print(f"  {label:10s}  dsMAPE={ds:+.2f} pp  dRMSSC={dr:+.3f}")

print()
print("=== STD DEV (N-BEATS sMAPE) ===")
for cond, label in [('Scratch_Standard','Scratch Std'), ('Scratch_Stabilized','Scratch Stab'),
                     ('ZeroShot_Standard','ZS Std'), ('ZeroShot_Stabilized','ZS Stab')]:
    sd = std3(rows, cond, 'sMAPE')
    print(f"  {label:15s}  std={sd:.2f}")

print()
print("=== PERCENTAGE REDUCTIONS (N-BEATS) ===")
for std_c, stab_c, label in [('Scratch_Standard','Scratch_Stabilized','Scratch'),
                               ('TL_Standard','TL_Stabilized','FT-TL'),
                               ('ZeroShot_Standard','ZeroShot_Stabilized','ZeroShot')]:
    rmssc_std = mean3(rows, std_c, 'RMSSC')
    rmssc_stab = mean3(rows, stab_c, 'RMSSC')
    smapc_std = mean3(rows, std_c, 'sMAPC')
    smapc_stab = mean3(rows, stab_c, 'sMAPC')
    pct_rmssc = (rmssc_stab - rmssc_std) / rmssc_std * 100
    pct_smapc = (smapc_stab - smapc_std) / smapc_std * 100
    print(f"  {label:10s}  RMSSC: {rmssc_std:.3f}->{rmssc_stab:.3f}  {pct_rmssc:+.1f}%   sMAPC: {smapc_std:.3f}->{smapc_stab:.3f}  {pct_smapc:+.1f}%")

# === N-HiTS ===
with open('nhits_experiment_results_tuned.csv') as f:
    nrows = list(csv.DictReader(f))

def nmean3(cond, col):
    vals = [float(r[col]) for r in nrows if r['condition'] == 'NHITS_'+cond]
    return statistics.mean(vals)

print()
print("=== TABLE 6: N-HiTS ===")
for cond, label in [('Scratch_Standard','Scratch N-HiTS'), ('Scratch_Stabilized','Scratch N-HiTS-S'),
                     ('TL_Standard','FT-TL N-HiTS'), ('TL_Stabilized','FT-TL N-HiTS-S'),
                     ('ZeroShot_Standard','ZS N-HiTS'), ('ZeroShot_Stabilized','ZS N-HiTS-S')]:
    s = nmean3(cond, 'sMAPE')
    r = nmean3(cond, 'RMSSE')
    rc = nmean3(cond, 'RMSSC')
    sp = nmean3(cond, 'sMAPC')
    print(f"  {label:25s}  sMAPE={s:.2f}  RMSSE={r:.3f}  RMSSC={rc:.3f}  sMAPC={sp:.3f}")

print()
print("=== TABLE 6 DELTAS (N-HiTS) ===")
for std_c, stab_c, label in [('Scratch_Standard','Scratch_Stabilized','Scratch'),
                               ('TL_Standard','TL_Stabilized','FT-TL'),
                               ('ZeroShot_Standard','ZeroShot_Stabilized','ZeroShot')]:
    ds = nmean3(stab_c, 'sMAPE') - nmean3(std_c, 'sMAPE')
    dr = nmean3(stab_c, 'RMSSC') - nmean3(std_c, 'RMSSC')
    print(f"  {label:10s}  dsMAPE={ds:+.2f} pp  dRMSSC={dr:+.3f}")

print()
print("=== PERCENTAGE REDUCTIONS (N-HiTS) ===")
for std_c, stab_c, label in [('Scratch_Standard','Scratch_Stabilized','Scratch'),
                               ('TL_Standard','TL_Stabilized','FT-TL'),
                               ('ZeroShot_Standard','ZeroShot_Stabilized','ZeroShot')]:
    rmssc_std = nmean3(std_c, 'RMSSC')
    rmssc_stab = nmean3(stab_c, 'RMSSC')
    smapc_std = nmean3(std_c, 'sMAPC')
    smapc_stab = nmean3(stab_c, 'sMAPC')
    pct_rmssc = (rmssc_stab - rmssc_std) / rmssc_std * 100
    pct_smapc = (smapc_stab - smapc_std) / smapc_std * 100
    print(f"  {label:10s}  RMSSC: {rmssc_std:.3f}->{rmssc_stab:.3f}  {pct_rmssc:+.1f}%   sMAPC: {smapc_std:.3f}->{smapc_stab:.3f}  {pct_smapc:+.1f}%")

print()
zs_std = nmean3('ZeroShot_Standard', 'sMAPE')
zs_stab = nmean3('ZeroShot_Stabilized', 'sMAPE')
print(f"N-HiTS ZS relative improvement: {(zs_std - zs_stab)/zs_std*100:.1f}%")

# === LAMBDA ===
with open('lambda_sensitivity_results.csv') as f:
    lrows = list(csv.DictReader(f))

print()
print("=== TABLE 8: LAMBDA ===")
for lam in ['0.0', '0.005', '0.01', '0.02', '0.05', '0.1', '0.2']:
    vals_s = [float(r['sMAPE']) for r in lrows if r['lambda'] == lam]
    vals_r = [float(r['RMSSE']) for r in lrows if r['lambda'] == lam]
    vals_rc = [float(r['RMSSC']) for r in lrows if r['lambda'] == lam]
    vals_sp = [float(r['sMAPC']) for r in lrows if r['lambda'] == lam]
    print(f"  L={lam:5s}  sMAPE={statistics.mean(vals_s):.2f}  RMSSE={statistics.mean(vals_r):.3f}  RMSSC={statistics.mean(vals_rc):.3f}  sMAPC={statistics.mean(vals_sp):.3f}")

baseline_rmssc = statistics.mean([float(r['RMSSC']) for r in lrows if r['lambda'] == '0.0'])
baseline_smapc = statistics.mean([float(r['sMAPC']) for r in lrows if r['lambda'] == '0.0'])
lam02_rmssc = statistics.mean([float(r['RMSSC']) for r in lrows if r['lambda'] == '0.02'])
lam02_smapc = statistics.mean([float(r['sMAPC']) for r in lrows if r['lambda'] == '0.02'])
lam02_smape = statistics.mean([float(r['sMAPE']) for r in lrows if r['lambda'] == '0.02'])
base_smape = statistics.mean([float(r['sMAPE']) for r in lrows if r['lambda'] == '0.0'])
print()
print(f"RMSSC reduction at 0.02: {(lam02_rmssc - baseline_rmssc)/baseline_rmssc*100:+.1f}%")
print(f"sMAPC reduction at 0.02: {(lam02_smapc - baseline_smapc)/baseline_smapc*100:+.1f}%")
print(f"sMAPE increase at 0.02: {lam02_smape - base_smape:+.2f} pp")

# Cross-check: N-BEATS-S ZS RMSSC vs N-BEATS FT RMSSC
nbeats_zs_stab_rmssc = mean3(rows, 'ZeroShot_Stabilized', 'RMSSC')
nbeats_ft_std_rmssc = mean3(rows, 'TL_Standard', 'RMSSC')
print()
print(f"N-BEATS-S ZS RMSSC={nbeats_zs_stab_rmssc:.3f} vs N-BEATS FT RMSSC={nbeats_ft_std_rmssc:.3f}")
print(f"Claim: ZS stab < FT std => {nbeats_zs_stab_rmssc < nbeats_ft_std_rmssc}")
