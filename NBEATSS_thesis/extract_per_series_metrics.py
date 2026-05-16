"""
Extract per-series sMAPE, RMSSE, sMAPC, RMSSC for all conditions.

Loads each trained checkpoint, runs it over the M3 test set, and records
metrics at the individual series level.  Results are saved to:
  per_series_results_raw.csv   - one row per (series, condition, seed, origin)
  per_series_results.csv       - one row per (series, condition) after averaging
                                 over origins AND seeds  (input to stat tests)
"""

import sys, os, glob, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np

from src.data.M3 import load_data
from src.methods.NBEATSS import LitNBEATSS
from src.methods.NHITSS import LitNHITSS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


RUNS = {
    ("Scratch",   "NBEATS", "Standard"):    [(1,"zraud12i","NBEATSS_thesis")],
    ("Scratch",   "NBEATS", "Stabilized"):  [(1,"rj0o61ve","NBEATSS_thesis")],
    ("TL",        "NBEATS", "Standard"):    [(1,"5emfx9rz","NBEATSS_thesis")],
    ("TL",        "NBEATS", "Stabilized"):  [(1,"kzac1sps","NBEATSS_thesis")],
    ("ZeroShot",  "NBEATS", "Standard"):    [(1,"4lf1k5v9","NBEATSS_thesis")],
    ("ZeroShot",  "NBEATS", "Stabilized"):  [(1,"j99n8rrl","NBEATSS_thesis")],
    ("Scratch",   "NHITS",  "Standard"):    [(1,"2u03kymc","NBEATSS_thesis")],
    ("Scratch",   "NHITS",  "Stabilized"):  [(1,"outh05z5","NBEATSS_thesis")],
    ("TL",        "NHITS",  "Standard"):    [(1,"6rvcpjjp","NBEATSS_thesis")],
    ("TL",        "NHITS",  "Stabilized"):  [(1,"oodho4ts","NBEATSS_thesis")],
    ("ZeroShot",  "NHITS",  "Standard"):    [(1,"v0hrzes0","NBEATSS_thesis")],
    ("ZeroShot",  "NHITS",  "Stabilized"):  [(1,"yyvya5ql","NBEATSS_thesis")],
}


def find_checkpoint(subdir, run_id):
    base = f"{subdir}/{run_id}/checkpoints"
    for name in ["last.ckpt", "best.ckpt"]:
        p = f"{base}/{name}"
        if os.path.exists(p):
            return p

    candidates = sorted(glob.glob(f"{base}/epoch=*.ckpt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint found in {base}")


def load_model(arch, ckpt_path):
    cls = LitNHITSS if arch == "NHITS" else LitNBEATSS
    model = cls.load_from_checkpoint(ckpt_path, map_location=DEVICE)
    model.eval()
    model.to(DEVICE)
    return model


@torch.no_grad()
def eval_model(model, test_dl):
    """
    Run model over test dataloader and return a list of dicts, one per sample:
      group_id, sMAPE, RMSSE, sMAPC, RMSSC
    """
    rows = []
    for x, _ in test_dl:
        enc = x["encoder_cont"].to(DEVICE)
        dec = x["decoder_cont"].to(DEVICE)
        groups = x["groups"].to(DEVICE)

        bs = enc.shape[0]


        lookback        = enc[:, :, 4]
        lookback_lagged = enc[:, :, 5]
        mean_val        = dec[:, :, 0]
        std_val         = dec[:, :, 1]
        scaling_sq      = enc[:, -1, 3]
        forecast_period = dec[:, :, 4]


        actual = forecast_period * std_val + mean_val


        fc_norm = model.ema_model[0](lookback)
        fl_norm = model.ema_model[0](lookback_lagged)

        fc = fc_norm * std_val + mean_val
        fl = fl_norm * std_val + mean_val


        smape = 200 * torch.mean(
            torch.abs(actual - fc) / (torch.abs(actual) + torch.abs(fc) + 1e-3),
            dim=-1)


        mse   = torch.mean((actual - fc)**2, dim=-1)
        rmsse = torch.sqrt(mse / (scaling_sq + 1e-3))
        rmsse = torch.clamp(rmsse, 0.0, 5.0)


        fc_ov = fc[:, :-1]
        fl_ov = fl[:, 1:]

        smapc = 200 * torch.mean(
            torch.abs(fc_ov - fl_ov) / (torch.abs(fc_ov) + torch.abs(fl_ov) + 1e-3),
            dim=-1)

        mse_s  = torch.mean((fc_ov - fl_ov)**2, dim=-1)
        rmssc  = torch.sqrt(mse_s / (scaling_sq + 1e-3))
        rmssc  = torch.clamp(rmssc, 0.0, 5.0)


        gids   = groups.squeeze(-1).cpu().numpy()
        smape_np  = smape.cpu().numpy()
        rmsse_np  = rmsse.cpu().numpy()
        smapc_np  = smapc.cpu().numpy()
        rmssc_np  = rmssc.cpu().numpy()

        for i in range(bs):
            rows.append({
                "series_id": int(gids[i]),
                "sMAPE":  float(smape_np[i]),
                "RMSSE":  float(rmsse_np[i]),
                "sMAPC":  float(smapc_np[i]),
                "RMSSC":  float(rmssc_np[i]),
            })

    return rows


print("Loading M3 test dataloaders (two variants)...")

def make_dl(bs_mult):
    _, _, _, _, dl = load_data(
        subset="Monthly",
        backcast_length_multiplier=bs_mult,
        forecast_length=6,
        validation_periods=18,
        test_periods=18,
        zero_mean=True,
        unit_variance=True,
        forecasting_origin_range_multiplier=1_000_000,
        batch_size=32,
        num_workers=0,
    )
    return dl

print("  Loading bs_mult=6 (Scratch models, backcast_length=36)...")
test_dl_6 = make_dl(6)
print("  Loading bs_mult=4 (TL/ZeroShot models, backcast_length=24)...")
test_dl_4 = make_dl(4)
print(f"  bs_mult=6: {sum(len(b[0]['groups']) for b in test_dl_6):,} windows")
print(f"  bs_mult=4: {sum(len(b[0]['groups']) for b in test_dl_4):,} windows\n")


all_records = []
total = sum(len(v) for v in RUNS.values())
done  = 0

for (scenario, arch, variant), seed_list in RUNS.items():
    for seed, run_id, subdir in seed_list:
        done += 1
        label = f"{scenario}_{arch}_{variant} seed={seed} ({run_id})"
        try:
            ckpt = find_checkpoint(subdir, run_id)
        except FileNotFoundError as e:
            print(f"  [{done}/{total}] SKIP {label} - {e}")
            continue

        print(f"  [{done}/{total}] {label}  ckpt={os.path.basename(ckpt)}")
        model = load_model(arch, ckpt)

        test_dl = test_dl_6 if scenario == "Scratch" else test_dl_4
        test_dl = test_dl_6 if scenario == "Scratch" else test_dl_4
        rows = eval_model(model, test_dl)
        for r in rows:
            r["scenario"]  = scenario
            r["arch"]      = arch
            r["variant"]   = variant
            r["seed"]      = seed
        all_records.extend(rows)


        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\nCollected {len(all_records):,} sample-level records.")

raw_df = pd.DataFrame(all_records)
raw_df.to_csv("per_series_results_raw.csv", index=False)
print("Saved per_series_results_raw.csv")


agg1 = (raw_df
        .groupby(["series_id", "scenario", "arch", "variant", "seed"])[["sMAPE","RMSSE","sMAPC","RMSSC"]]
        .mean()
        .reset_index())


agg2 = (agg1
        .groupby(["series_id", "scenario", "arch", "variant"])[["sMAPE","RMSSE","sMAPC","RMSSC"]]
        .mean()
        .reset_index())


agg2["model"] = agg2["arch"] + "_" + agg2["variant"]
agg2.to_csv("per_series_results.csv", index=False)
print(f"Saved per_series_results.csv  ({len(agg2):,} rows)")

print("\n=== Sanity check vs experiment_results_tuned.csv ===")
known_nb = pd.read_csv("experiment_results_tuned.csv")
known_nh = pd.read_csv("nhits_experiment_results_tuned.csv")

condition_map = {
    ("Scratch","NBEATS","Standard"):   "Scratch_Standard",
    ("Scratch","NBEATS","Stabilized"): "Scratch_Stabilized",
    ("TL","NBEATS","Standard"):        "TL_Standard",
    ("TL","NBEATS","Stabilized"):      "TL_Stabilized",
    ("ZeroShot","NBEATS","Standard"):  "ZeroShot_Standard",
    ("ZeroShot","NBEATS","Stabilized"):"ZeroShot_Stabilized",
    ("Scratch","NHITS","Standard"):    "NHITS_Scratch_Standard",
    ("Scratch","NHITS","Stabilized"):  "NHITS_Scratch_Stabilized",
    ("TL","NHITS","Standard"):         "NHITS_TL_Standard",
    ("TL","NHITS","Stabilized"):       "NHITS_TL_Stabilized",
    ("ZeroShot","NHITS","Standard"):   "NHITS_ZeroShot_Standard",
    ("ZeroShot","NHITS","Stabilized"): "NHITS_ZeroShot_Stabilized",
}
known_all = pd.concat([known_nb, known_nh])

print(f"{'Condition':<35} {'Metric':<8} {'Known':>8} {'Extracted':>10} {'Diff':>8}")
print("-"*75)
for (s,a,v), cond_name in condition_map.items():
    subset = agg2[(agg2.scenario==s)&(agg2.arch==a)&(agg2.variant==v)]
    known_sub = known_all[known_all.condition==cond_name]
    if subset.empty or known_sub.empty:
        continue
    for metric in ["sMAPE","RMSSE","sMAPC","RMSSC"]:
        if metric not in known_sub.columns:
            continue
        ext_val  = subset[metric].mean()
        know_val = known_sub[metric].mean()
        diff = ext_val - know_val
        flag = " OK" if abs(diff) < 0.1 else " <- CHECK"
        print(f"{cond_name:<35} {metric:<8} {know_val:>8.4f} {ext_val:>10.4f} {diff:>+8.4f}{flag}")

print("\nDone.")
