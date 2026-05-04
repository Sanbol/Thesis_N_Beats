# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis investigating **forecast stability** in deep learning time series models. The core question: can spectral regularization (λ_stability) and EMA smoothing improve forecast consistency when the input window shifts by one step, without sacrificing accuracy?

Two model families are studied:
- **N-BEATS-S** (`src/methods/NBEATSS.py`): doubly-residual blocks with generic/trend/seasonality stacks
- **N-HiTS-S** (`src/methods/NHITSS.py`): hierarchical interpolation variant with multi-rate pooling

**Stability extensions** (both models):
- `lambda_stability` (0–1): weight on the stability term in the composite loss
- `ema_decay` (0–1): exponential moving average of model weights used at evaluation time

## Running Experiments

**Single run (edit config at top of `main.py` lines 53–131, then):**
```bash
python main.py
```

**Full automated experiment suite (18 runs: 6 conditions × 3 seeds for N-BEATS-S):**
```bash
python run_experiments.py
```

**N-HiTS-S equivalent:**
```bash
python main_nhits.py
```

**Hyperparameter tuning:**
```bash
python hp_tune.py                     # N-BEATS-S
python _hp_tune_temp_nhits_m4.py      # N-HiTS-S
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Configuration

All experiment parameters are set in the config block at the top of `main.py` (or `main_nhits.py`). Key parameters:

| Parameter | Default | Notes |
|---|---|---|
| `DATASET` | `'M3'` | `'M3'` or `'M4'` |
| `lambda_stability` | `0.02` | 0.0 = no stability loss |
| `ema_decay` | `0.99` | 0.0 = no EMA |
| `learning_rate` | `1e-3` | Use `1e-5` for fine-tuning |
| `max_epochs` | `10` | 15 for transfer learning |
| `PRETRAIN` | `False` | Set True + point `PRETRAIN_PATH` for transfer learning |
| `ZERO_SHOT` | `False` | Evaluate M4-trained model directly on M3 |

## Architecture

### Loss Function
```
loss = (1 − λ) · RMSSE(ŷ, y) + λ · RMSSE(ŷ[:-1], ŷ_shifted[1:])
```
The second term penalises forecast change when the input window shifts by one time step.

### Data Pipeline
1. Raw CSVs in `data/raw/` → `src/data/M3.py` / `src/data/M4.py`
2. `src/data/utils/_utils.py`: train/val/test splits, min-max scaling per series, conversion to PyTorch Forecasting `TimeSeriesDataSet`
3. Each sample has 6 channels: `[value, log_value, month_sin, month_cos, trend, series_id_embed]`
4. Processed tensors cached in `data/processed/` (git-ignored)

### Evaluation Metrics
- **Accuracy**: sMAPE (lower=better, 0–200), RMSSE (lower=better, 0–5)
- **Stability** (novel): RMSSC — root mean squared scaled change between adjacent forecasts; sMAPC — percentage equivalent

Metrics implemented in `src/utils/metrics.py`.

### Experiment Tracking
Runs are logged to **Weights & Biases** (wandb). Set `wandb` project/entity in the config block. Forecasts optionally saved to CSV via the `WriteForecastsToCSV` callback in `src/utils/callbacks.py`.

## Key Results (thesis)
- Transfer Learning + Stabilization is optimal: sMAPE=13.01, RMSSC=0.310
- Stabilization reduces RMSSC by 15–20% with negligible sMAPE degradation
- Zero-shot transfer is competitive (only ~0.1 sMAPE worse than fine-tuned)
- Stability extensions are model-agnostic (confirmed on both N-BEATS-S and N-HiTS-S)
