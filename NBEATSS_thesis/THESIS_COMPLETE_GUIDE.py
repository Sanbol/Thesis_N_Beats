# ============================================================================
#  COMPLETE THESIS GUIDE — N-BEATS-S: STABILITY IN TRANSFER LEARNING
#  For the team: Everything explained from absolute zero
# ============================================================================

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================
# PART 1: THE BIG PICTURE — What are we doing and why?
# PART 2: THE DATA — What goes in?
# PART 3: THE MODEL — How N-BEATS works (block by block)
# PART 4: THE "S" — What makes N-BEATS-S different from N-BEATS
# PART 5: TRAINING — How the model learns
# PART 6: TRANSFER LEARNING — The main thesis idea
# PART 7: THE EXPERIMENTS — What we ran and why
# PART 8: THE RESULTS — What we found
# PART 9: CODE WALKTHROUGH — Every file, line by line
# PART 10: GLOSSARY — Every term explained
# ============================================================================


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 1: THE BIG PICTURE                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# THESIS QUESTION (one sentence):
#   "Does adding a stability term to N-BEATS improve transfer learning
#    from a large dataset (M4) to a small dataset (M3), without hurting 
#    accuracy?"
#
# WHY THIS MATTERS (real world):
#   Imagine you're a company that needs to forecast sales for 100 products.
#   You only have 2 years of monthly data per product (= small dataset).
#   But there exists a public dataset with 100,000 other monthly time series.
#   
#   Question: Can you use those 100,000 series to help your 100-product forecast?
#   Answer: YES — this is called TRANSFER LEARNING.
#   
#   But there's a catch: when you update your forecast each month with new data,
#   the predictions might "jump around" wildly. A forecast of 500 units suddenly
#   becomes 800, then drops to 300. This is UNSTABLE.
#   
#   Our thesis asks: Can we make transfer learning work WHILE keeping
#   forecasts STABLE (not jumping around)?
#
# THE MODEL WE USE:
#   N-BEATS = Neural Basis Expansion Analysis for Time Series
#   N-BEATS-S = N-BEATS with a Stability penalty (the "S")
#   
#   This was developed by Professor Jente Van Belle.
#   We are testing whether the stability component helps during transfer learning.


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 2: THE DATA                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# We use two well-known forecasting competition datasets:
#
# ┌─────────┬────────────┬──────────────┬─────────────────────────────────┐
# │ Dataset │ # Series   │ Frequency    │ Role in our thesis              │
# ├─────────┼────────────┼──────────────┼─────────────────────────────────┤
# │ M4      │ 100,000+   │ Monthly      │ SOURCE (big) — for pre-training │
# │ M3      │ 861        │ Monthly      │ TARGET (small) — what we care   │
# │         │            │              │ about forecasting well          │
# └─────────┴────────────┴──────────────┴─────────────────────────────────┘
#
# Each "series" is just a sequence of numbers over time. For example:
#   Series 1: [100, 105, 98, 110, 115, 108, 120, ...]  (monthly values)
#   Series 2: [50, 52, 48, 55, 53, 51, 56, ...]
#   ...
#   Series 861: [200, 210, 195, 220, ...]
#
# HOW WE SPLIT THE DATA (for M3):
# ┌──────────────────────────────────────────────────────────────────────┐
# │  Each time series (e.g., 126 months long):                         │
# │                                                                     │
# │  [████████████████████████████████████|██████████████|██████████████] │
# │  ←————— Training data ——————————————→←— Validation —→←——— Test ———→ │
# │         (first ~90 months)            (18 months)     (18 months)   │
# │                                                                     │
# │  Training: model learns from this                                   │
# │  Validation: used to tune hyperparameters (not in our test runs)   │
# │  Test: NEVER seen during training — used to measure final accuracy │
# └──────────────────────────────────────────────────────────────────────┘
#
# WHAT THE MODEL SEES (input → output):
#   Input (lookback window):  48 past values  (backcast_length = 8 × 6 = 48)
#   Output (forecast):         6 future values (forecast_length = 6)
#
#   Example:
#   Input:  [100, 105, 98, 110, ..., 120]  ← 48 months of history
#   Output: [125, 130, 128, 135, 140, 142] ← predict next 6 months
#
# WHY 48 AND 6?
#   backcast_length_multiplier = 8
#   forecast_length = 6
#   backcast_length = 8 × 6 = 48
#   
#   The model looks back 48 months (4 years) to predict 6 months ahead.
#   This is a common choice for monthly data — 4 years captures seasonality.
#
# NORMALIZATION (zero_mean=True, unit_variance=True):
#   Raw data might be: Series A has values around 10,000 and Series B around 5.
#   The model would struggle with such different scales.
#   
#   Solution: For each series, subtract the mean and divide by std deviation.
#   Now all series are centered around 0 with similar spread.
#   
#   Before: [10000, 10500, 9800, 11000]
#   After:  [-0.89, 0.15, -1.61, 1.63]   ← roughly mean=0, std=1
#   
#   After forecasting, we "undo" this: multiply back by std, add mean.
#   This happens at line 310 of NBEATSS.py: 
#       rescaled_final_forecast = final_forecast * std + mean
#
# LAGGED VALUES (the key to stability):
#   For every time series window, the data also includes a "lagged" version.
#   
#   Normal window:  uses data up to month T      → forecasts months T+1 to T+6
#   Lagged window:  uses data up to month T-1    → forecasts months T to T+5
#   
#   Why? To measure STABILITY: if we compare the forecast made at time T
#   with the forecast made at time T-1 for overlapping months, they should
#   be similar. If they're wildly different, the model is UNSTABLE.
#
#   Visual example:
#   At time T-1, model predicts:  [100, 105, 110, 115, 120, 125]
#                                   ↑    ↑    ↑    ↑    ↑
#   At time T,   model predicts:       [106, 112, 114, 119, 122, 128]
#                                   
#   Stability = how different are the overlapping predictions?
#   [105 vs 106], [110 vs 112], [115 vs 114], [120 vs 119], [125 vs 122]
#   Small differences = STABLE (good!)
#   Large differences = UNSTABLE (bad!)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 3: THE MODEL — How N-BEATS Works                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# N-BEATS is built from BLOCKS stacked together.
# Think of it like an assembly line where each worker (block) refines the 
# product a little more.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │                    N-BEATS ARCHITECTURE (Our Config)                    │
# │                                                                         │
# │    Input: 48 past values                                                │
# │      │                                                                  │
# │      ▼                                                                  │
# │  ┌─────────┐                                                            │
# │  │ Block 1 │ → partial forecast₁ + residual₁                           │
# │  └─────────┘                                                            │
# │      │ (residual = input minus what Block 1 "explained")               │
# │      ▼                                                                  │
# │  ┌─────────┐                                                            │
# │  │ Block 2 │ → partial forecast₂ + residual₂                           │
# │  └─────────┘                                                            │
# │      │                                                                  │
# │      ▼                                                                  │
# │  ┌─────────┐                                                            │
# │  │ Block 3 │ → partial forecast₃                                       │
# │  └─────────┘                                                            │
# │                                                                         │
# │  Final forecast = forecast₁ + forecast₂ + forecast₃                   │
# │                                                                         │
# │  Think of it as:                                                        │
# │  Block 1 captures the main trend                                        │
# │  Block 2 captures seasonality the trend missed                          │
# │  Block 3 captures remaining patterns                                    │
# └──────────────────────────────────────────────────────────────────────────┘
#
# === INSIDE ONE BLOCK ===
#
# Each block is a small neural network with these layers:
#
#   Input (48 values)
#     │
#     ▼
#   fc1: Linear(48 → 32) + LeakyReLU    ← "compress" 48 values into 32 numbers
#     │
#     ▼
#   fc2: Linear(32 → 32) + LeakyReLU    ← process further
#     │
#     ▼
#   fc3: Linear(32 → 32) + LeakyReLU    ← process further
#     │
#     ▼
#   fc4: Linear(32 → 32) + LeakyReLU    ← process further
#     │
#     ├──────────────────────┐
#     ▼                      ▼
#   fc_backcast(32→32)    fc_forecast(32→32)   ← split into two tasks
#     │                      │
#     ▼                      ▼
#   fc_backcast_out(32→48) fc_forecast_out(32→6)
#     │                      │
#     ▼                      ▼
#   BACKCAST (48 values)   FORECAST (6 values)
#
# What is a "Linear" layer?
#   It's just: output = input × weights + bias
#   Where weights and bias are the learnable numbers.
#   Linear(48 → 32) means: take 48 numbers in, multiply by a 48×32 matrix, 
#   add 32 bias values, get 32 numbers out.
#
# What is "LeakyReLU"?
#   A simple function applied after each linear layer:
#   If x > 0: output = x        (keep positive values as-is)
#   If x < 0: output = 0.01 * x (shrink negative values to almost zero)
#   
#   This adds "non-linearity" — without it, stacking linear layers would
#   just be one big linear layer (useless for complex patterns).
#
# What is a BACKCAST?
#   The block's "reconstruction" of the input it received.
#   If Block 1 receives [100, 105, 98, ...] and outputs backcast [99, 103, 97, ...],
#   then the RESIDUAL passed to Block 2 is:
#   [100-99, 105-103, 98-97, ...] = [1, 2, 1, ...]
#   
#   This is the part that Block 1 DIDN'T explain. Block 2 tries to model this.
#
# What is a FORECAST?
#   The block's prediction of the 6 future values.
#   Each block contributes a partial forecast.
#   The final forecast is the SUM of all blocks' forecasts.
#
# === CODE REFERENCE (NBEATSS.py lines 22-64) ===
#
# class NBEATS_block(nn.Module):
#     def __init__(self, backcast_length, forecast_length, hidden_layer_units):
#         # fc1-fc4: four shared layers (48→32→32→32→32)
#         # fc_backcast + fc_forecast: two task-specific layers (32→32 each)
#         # fc_backcast_output: produces backcast (32→48)
#         # fc_forecast_output: produces forecast (32→6)
#
#     def forward(self, x):
#         # h1 = LeakyReLU(fc1(x))      ← feed input through layer 1
#         # h2 = LeakyReLU(fc2(h1))     ← feed h1 through layer 2
#         # h3 = LeakyReLU(fc3(h2))     ← feed h2 through layer 3
#         # h4 = LeakyReLU(fc4(h3))     ← feed h3 through layer 4
#         # h_backcast = LeakyReLU(fc_backcast(h4))   ← backcast branch
#         # h_forecast = LeakyReLU(fc_forecast(h4))   ← forecast branch
#         # backcast = fc_backcast_output(h_backcast)  ← final backcast
#         # forecast = fc_forecast_output(h_forecast)  ← final forecast
#         # return backcast, forecast

# === PARAMETER COUNT ===
#
# Each block has:
#   fc1:     48×32 + 32 = 1,568 parameters
#   fc2:     32×32 + 32 = 1,056
#   fc3:     32×32 + 32 = 1,056
#   fc4:     32×32 + 32 = 1,056
#   fc_bc:   32×32 + 32 = 1,056
#   fc_fc:   32×32 + 32 = 1,056
#   fc_bc_o: 32×48 + 48 = 1,584
#   fc_fc_o: 32×6  + 6  = 198
#   ─────────────────────────────
#   Total per block:       8,630
#
# With 3 blocks: 3 × 8,630 = 25,890 parameters
# With EMA copy: 25,890 × 2 = 51,780 parameters (but EMA is not trained separately)
#
# This is a TINY model. GPT-4 has 1.7 TRILLION parameters.
# Our model has 51,800. That's 33 million times smaller.


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 4: THE "S" — What makes N-BEATS-S Special                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# N-BEATS-S adds TWO things to standard N-BEATS:
#
# ═══ ADDITION 1: STABILITY PENALTY (lambda_stability) ═══
#
# Standard N-BEATS loss function:
#   loss = accuracy_loss
#   "Just make the forecast as accurate as possible"
#
# N-BEATS-S loss function:
#   loss = (1 - λ) × accuracy_loss + λ × stability_loss
#   "Make the forecast accurate AND stable"
#
# In our experiments:
#   Standard:   λ = 0.00 → loss = 1.0 × accuracy + 0.0 × stability
#   Stabilized: λ = 0.02 → loss = 0.98 × accuracy + 0.02 × stability
#
# What is stability_loss exactly?
#   Remember the lagged forecasts from Part 2?
#   
#   forecast at time T:    [f₁, f₂, f₃, f₄, f₅, f₆]
#   forecast at time T-1:  [g₁, g₂, g₃, g₄, g₅, g₆]
#   
#   The overlapping predictions are:
#   f₁ vs g₂, f₂ vs g₃, f₃ vs g₄, f₄ vs g₅, f₅ vs g₆
#   (f₁ predicts the same month as g₂, etc.)
#   
#   stability_loss = how different are these?
#   Computed as RMSSE between forecast[:,:-1] and forecast_lagged[:,1:]
#   
#   Code (NBEATSS.py line 340):
#   loss_stability = RMSSE_calculation(
#       final_forecast[:,:-1],          ← [f₁, f₂, f₃, f₄, f₅]
#       final_forecast_lagged[:,1:],    ← [g₂, g₃, g₄, g₅, g₆]
#       scaling_constant)
#
# WHY λ = 0.02 and not higher?
#   If λ = 1.0, the model would ONLY care about stability and ignore accuracy.
#   The forecast would be perfectly stable but completely wrong.
#   λ = 0.02 is a gentle nudge: "try to be stable, but accuracy is still king."
#
# ═══ ADDITION 2: EMA MODEL (ema_decay) ═══
#
# EMA = Exponential Moving Average
#
# The idea: During training, model weights bounce around a lot.
#   Step 1: weights = [0.5, -0.3, 0.8, ...]
#   Step 2: weights = [0.7, -0.1, 0.6, ...]  ← jumped!
#   Step 3: weights = [0.4, -0.4, 0.9, ...]  ← jumped again!
#
# EMA keeps a "smoothed" copy of the weights:
#   ema_weights = 0.99 × ema_weights_old + 0.01 × current_weights
#
# This means:
#   - The EMA model changes very slowly (only 1% of each update)
#   - It's like a "rolling average" of all the models during training
#   - It tends to be more stable and generalizes better
#
# CRITICAL: During training, we use self.model (the regular model).
#           During evaluation/testing, we use self.ema_model (the smooth one).
#
# Code (NBEATSS.py line 375-378):
#   def on_train_batch_end(self, ...):
#       for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
#           ema_param.data = 0.99 * ema_param.data + 0.01 * model_param.data
#
# Code (NBEATSS.py lines 297-303):
#   if not evaluate:  # training
#       model_forecast = self.model[...](lookback_window)      ← regular model
#   else:             # evaluation/testing
#       model_forecast = self.ema_model[...](lookback_window)  ← smoothed model
#
# In our experiments:
#   Standard:   ema_decay = 0.0  → ema = 0.0 × ema_old + 1.0 × current
#                                  = just copies current weights (EMA disabled)
#   Stabilized: ema_decay = 0.99 → ema = 0.99 × ema_old + 0.01 × current
#                                  = slow-moving average (EMA enabled)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 5: TRAINING — How the Model Learns                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Training is an iterative process. Here's exactly what happens:
#
# STEP 1: INITIALIZE
#   - Create model with 51,800 random numbers (weights)
#   - These weights are meaningless at this point
#
# STEP 2: TRAINING LOOP (10 epochs × 50 batches = 500 iterations)
#   For each epoch (1 to 10):
#     For each batch (1 to 50):
#       a) Take 32 random time series windows from the dataset
#       b) Feed the 48-value lookback windows into the model
#       c) Model outputs 6-value forecasts
#       d) Compare forecasts to actual values → compute loss
#       e) Compute gradients (which direction to adjust each weight)
#       f) Update weights: weight = weight - learning_rate × gradient
#       g) Update EMA model: ema = 0.99 × ema + 0.01 × weight
#
# STEP 3: TESTING
#   - Use all 861 test windows (the last 18 months of each series)
#   - Feed through EMA model → get forecasts
#   - Compare to actual values → compute metrics (sMAPE, RMSSE, RMSSC)
#
# === KEY CONCEPTS ===
#
# BATCH:
#   We don't feed all 861 series at once (would use too much memory).
#   We feed 32 at a time (batch_size = 32).
#   50 batches per epoch means 50 × 32 = 1,600 windows per epoch.
#
# EPOCH:
#   One complete pass through the training data.
#   After each epoch, the learning rate decays: lr = lr × 0.97
#
# LEARNING RATE:
#   How big each weight adjustment is.
#   
#   High LR (1e-3 = 0.001):
#     weight = 0.5 → gradient says "go up" → weight = 0.5 + 0.001 × gradient
#     Big steps → learns fast but might overshoot
#   
#   Low LR (1e-5 = 0.00001):
#     weight = 0.5 → gradient says "go up" → weight = 0.5 + 0.00001 × gradient
#     Tiny steps → learns slow but precise
#   
#   We use 1e-3 for training from scratch (need to learn a lot)
#   We use 1e-5 for fine-tuning (just making small adjustments)
#
# GRADIENT:
#   A number that tells you "which direction to move this weight to reduce loss."
#   Computed automatically by PyTorch using calculus (backpropagation).
#   You don't need to understand the math — just know that:
#   positive gradient → decrease the weight
#   negative gradient → increase the weight
#
# LOSS FUNCTION:
#   The "score" that tells the model how bad its predictions are.
#   Lower = better.
#   
#   Our loss (for stabilized version):
#   loss = 0.98 × RMSSE(forecast, actual) + 0.02 × RMSSE(forecast, forecast_lagged)
#   
#   The optimizer tries to minimize this number by adjusting weights.
#
# DATA AUGMENTATION (NBEATSS.py lines 268-276):
#   During training only, the data is slightly modified:
#   - Random shift: add a small random number to all values
#   - Random scale: multiply all values by a random factor (0.5 to 1.5)
#   
#   This is like creating "fake" new training data.
#   It helps the model generalize (not memorize specific patterns).
#   
#   Example:
#   Original: [100, 105, 98, 110]
#   Shifted:  [103, 108, 101, 113]  (added 3)
#   Scaled:   [103, 108, 101, 113] × 1.2 = [123.6, 129.6, 121.2, 135.6]
#
# GRADIENT CLIPPING (max_norm = 1.0):
#   Sometimes gradients get very large ("exploding gradients").
#   This would cause weights to change too drastically in one step.
#   Gradient clipping caps the gradient magnitude at 1.0.
#   Like putting a speed limit on weight updates.
#
# OPTIMIZER (AdamW):
#   The algorithm that decides HOW to update weights.
#   AdamW is smart — it keeps track of:
#   - Average gradient direction (momentum)
#   - Average gradient magnitude (adaptive learning rate)
#   - Weight decay (slightly shrinks weights each step to prevent overfitting)
#   
#   It's the most popular optimizer in deep learning.
#   Code (NBEATSS.py line 356):
#   optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 5.5: THE ENSEMBLE (How Multiple Copies Work Together)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# The code supports running MULTIPLE copies of N-BEATS simultaneously.
# This is controlled by: ensemble_size (in our experiments = 1, so OFF)
#
# HOW IT WORKS (if ensemble_size = 3):
#
# ┌────────────────────────────────────────────────────────────────────┐
# │  Same input (48 values) goes to ALL 3 copies:                    │
# │                                                                    │
# │  Copy 1 (random init A) → forecast: [100, 105, 110, 115, 120, 125]│
# │  Copy 2 (random init B) → forecast: [98, 107, 112, 113, 118, 127] │
# │  Copy 3 (random init C) → forecast: [102, 104, 108, 117, 122, 123]│
# │                                                                    │
# │  Final = average across copies:                                    │
# │  Final forecast: [100, 105.3, 110, 115, 120, 125]                │
# │                                                                    │
# │  Each copy starts with DIFFERENT random weights,                  │
# │  so they learn slightly different things.                          │
# │  Averaging cancels out individual errors.                          │
# └────────────────────────────────────────────────────────────────────┘
#
# CLEVER TRAINING TRICK (optimizer_step, NBEATSS.py lines 367-372):
#   With ensemble_size > 1, NOT all copies are updated every batch.
#   On batch 0: only Copy 1's gradients are active
#   On batch 1: only Copy 2's gradients are active
#   On batch 2: only Copy 3's gradients are active
#   On batch 3: only Copy 1's gradients are active
#   ...
#   
#   This means each copy sees different batches → more diversity.
#   Code:
#     ensemble_id_update = batch_idx % ensemble_size  ← which copy to update
#     for idx, model in enumerate(self.model):
#         if idx != ensemble_id_update:
#             for param in model.parameters():
#                 param.grad = None  ← zero out gradients for other copies
#
# IN OUR EXPERIMENTS:
#   ensemble_size = 1 → only 1 copy → no averaging → ensemble is OFF
#   We could set it to 3 to enable it (just a config change above line 105)
#   It would triple the model size (25.9K × 3 = 77.7K params) and slow training.


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 6: TRANSFER LEARNING                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# THE PROBLEM:
#   M3 has only 861 time series. That's not much to learn general patterns from.
#   The model might memorize M3 instead of learning real forecasting skills.
#
# THE IDEA:
#   M4 has 100,000+ time series. If we train on M4 first, the model learns
#   GENERAL patterns of monthly time series (trends, seasonality, noise).
#   Then we can take those learned weights and refine them on M3.
#
# THE ANALOGY:
#   Imagine learning to cook:
#   
#   SCRATCH (no transfer):
#     You've never cooked before → someone gives you 10 recipes → 
#     you learn OK but make mistakes because 10 recipes isn't enough
#   
#   TRANSFER LEARNING:
#     You first train at a restaurant with 10,000 recipes (M4) →
#     you learn knife skills, flavor pairing, timing, etc. →
#     then someone gives you 10 specific recipes (M3) →
#     you learn these MUCH faster because you already know the basics
#
# STEP BY STEP:
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  STEP 1: PRE-TRAIN ON M4                                       │
# │                                                                  │
# │  Random weights [0.3, -0.7, 0.1, ...]                          │
# │       │                                                          │
# │       ▼  Train for 10 epochs on 100,000 M4 series              │
# │       │  Learning rate: 1e-3 (big steps, lots to learn)        │
# │       ▼                                                          │
# │  Smart weights [0.52, -0.18, 0.73, ...]                        │
# │       │                                                          │
# │       ▼  SAVE to checkpoint file (.ckpt)                       │
# │  File: NBEATSS_thesis/<run_id>/checkpoints/last.ckpt           │
# └──────────────────────────────────────────────────────────────────┘
#         │
#         ▼
# ┌──────────────────────────────────────────────────────────────────┐
# │  STEP 2: FINE-TUNE ON M3                                       │
# │                                                                  │
# │  Load smart weights from checkpoint                             │
# │  [0.52, -0.18, 0.73, ...]                                     │
# │       │                                                          │
# │       ▼  Train for 15 epochs on 861 M3 series                  │
# │       │  Learning rate: 1e-5 (100x smaller! tiny adjustments)  │
# │       │  Key: Don't destroy what we learned from M4             │
# │       ▼                                                          │
# │  M3-optimized weights [0.54, -0.16, 0.71, ...]                │
# │       │                                                          │
# │       ▼  TEST on M3 test set → get final metrics               │
# └──────────────────────────────────────────────────────────────────┘
#
# WHY LOWER LEARNING RATE FOR FINE-TUNING?
#   Pre-training LR:  1e-3 = 0.001    (big steps — start from random, lots to learn)
#   Fine-tuning LR:   1e-5 = 0.00001  (tiny steps — already good, just refine)
#   
#   If we used 1e-3 for fine-tuning:
#   Step 1: weights = [0.52, -0.18, 0.73]  ← smart M4 knowledge
#   Step 2: weights = [1.85, 0.42, -0.31]  ← all M4 knowledge DESTROYED
#   
#   This is called "CATASTROPHIC FORGETTING" — the model forgets everything
#   it learned from M4 in the first few batches.
#   
#   With 1e-5:
#   Step 1: weights = [0.52, -0.18, 0.73]  ← smart M4 knowledge
#   Step 2: weights = [0.5201, -0.1799, 0.7299]  ← barely changed! Safe.
#
# IN THE CODE (main.py):
#   Pre-training:
#     load_model = False              ← start from random
#     dataset = "M4"                  ← train on M4
#     learning_rate = 1e-3            ← big steps
#   
#   Fine-tuning:
#     load_model = True               ← load checkpoint
#     model_id = "<run_id>"           ← which checkpoint to load
#     dataset = "M3"                  ← now train on M3
#     learning_rate = 1e-5            ← tiny steps
#     update_loaded_model_specific_training_and_eval_hparams = True  ← override LR


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 7: THE EXPERIMENTS — What We Ran                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# We ran 4 CONDITIONS, each with 3 different SEEDS = 12 core experiments.
# Plus 6 M4 pre-training runs = 18 total runs.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  THE 4 CONDITIONS (2×2 design)                                         │
# │                                                                         │
# │                    │  Standard (N-BEATS) │ Stabilized (N-BEATS-S)       │
# │                    │  λ=0.0, ema=0.0     │ λ=0.02, ema=0.99           │
# │  ──────────────────┼─────────────────────┼─────────────────────────────│
# │  Scratch           │  A. Scratch Std     │ B. Scratch Stab             │
# │  (M3 only)         │  (baseline)         │ (does stability help?)     │
# │  ──────────────────┼─────────────────────┼─────────────────────────────│
# │  Transfer Learning │  C. TL Std          │ D. TL Stab                  │
# │  (M4 → M3)         │  (does TL help?)    │ (MAIN THESIS QUESTION)     │
# └─────────────────────────────────────────────────────────────────────────┘
#
# WHY 3 SEEDS?
#   A "seed" controls all randomness: initial weights, batch order, augmentation.
#   Same seed → same results. Different seed → slightly different results.
#   
#   If we only ran 1 seed and got 12.5%, maybe we got lucky.
#   With 3 seeds (1, 2, 3), we get:
#     Seed 1: 13.00%
#     Seed 2: 13.00%
#     Seed 3: 13.02%
#   Mean = 13.01% ± 0.01% → very consistent! Not luck.
#   
#   This is why academic papers require multiple seeds.
#
# HYPERPARAMETERS KEPT IDENTICAL ACROSS ALL CONDITIONS:
#   ┌──────────────────────────────────────────────────┐
#   │ Architecture:                                     │
#   │   backcast_length_multiplier = 8                  │
#   │   forecast_length = 6                             │
#   │   hidden_layer_units = 32                         │
#   │   n_blocks = 3                                    │
#   │   n_blocks_shared = 1                             │
#   │   ensemble_size = 1                               │
#   │   zero_mean = True                                │
#   │   unit_variance = True                            │
#   │                                                    │
#   │ Training:                                          │
#   │   batch_size = 32                                  │
#   │   batches_per_epoch = 50                           │
#   │   max_norm = 1.0 (gradient clipping)              │
#   │   enforce_nonnegative_forecast = True              │
#   │                                                    │
#   │ Scratch training: LR=1e-3, gamma=1.0, epochs=10  │
#   │ Fine-tuning:      LR=1e-5, gamma=0.97, epochs=15 │
#   └──────────────────────────────────────────────────┘
#
# ONLY THESE TWO THINGS CHANGE BETWEEN STANDARD/STABILIZED:
#   ┌───────────────────────────────────────────┐
#   │ Standard:   lambda_stability = 0.0        │
#   │             ema_decay = 0.0               │
#   │                                            │
#   │ Stabilized: lambda_stability = 0.02       │
#   │             ema_decay = 0.99              │
#   └───────────────────────────────────────────┘
#   
#   This is CRITICAL for a fair comparison. If you change 10 things at once,
#   you can't tell which change caused the improvement. By changing ONLY
#   these two, we can definitively say: "the stability component caused this."


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 8: THE RESULTS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# === METRICS EXPLAINED ===
#
# sMAPE (Symmetric Mean Absolute Percentage Error):
#   Formula: 200 × mean(|actual - forecast| / (|actual| + |forecast|))
#   Range: 0% (perfect) to 200% (terrible)
#   Interpretation: "On average, how far off are we in percentage terms?"
#   Example: actual=100, forecast=110 → 200×|100-110|/(100+110) = 9.5%
#
# RMSSE (Root Mean Squared Scaled Error):
#   Formula: sqrt(mean((actual - forecast)² / scaling_constant))
#   The scaling_constant is based on the in-sample naive forecast error.
#   < 1.0 means we beat the naive forecast (predicting "tomorrow = today")
#   > 1.0 means we're worse than naive
#   Our values are around 1.2-1.3 which is normal for M3 Monthly.
#
# RMSSC (Root Mean Squared Scaled Change — STABILITY METRIC):
#   Same formula as RMSSE but applied to:
#   consecutive forecast revisions instead of forecast vs actual.
#   Lower = more stable forecasts (less jumping between updates)
#   This is THE key metric for our thesis.
#
# sMAPC (Symmetric Mean Absolute Percentage Change):
#   Like sMAPE but for stability.
#   Measures percentage change between consecutive forecast revisions.
#
# === ACTUAL RESULTS ===
#
# ┌─────────────────────────┬────────────────┬────────────────┬────────────────┐
# │ Condition               │ sMAPE (↓)      │ RMSSE (↓)      │ RMSSC (↓)      │
# │                         │ accuracy       │ accuracy       │ STABILITY     │
# ├─────────────────────────┼────────────────┼────────────────┼────────────────┤
# │ A. Scratch Standard     │ 13.17 ± 0.45   │ 1.340 ± 0.168  │ 0.379 ± 0.012  │
# │ B. Scratch Stabilized   │ 13.23 ± 0.07   │ 1.335 ± 0.015  │ 0.305 ± 0.002  │
# │ C. TL Standard          │ 13.02 ± 0.06   │ 1.265 ± 0.022  │ 0.326 ± 0.015  │
# │ D. TL Stabilized        │ 13.01 ± 0.01   │ 1.261 ± 0.014  │ 0.310 ± 0.011  │
# └─────────────────────────┴────────────────┴────────────────┴────────────────┘
#
# === HOW TO READ THIS TABLE ===
#
# "13.17 ± 0.45" means:
#   Average across 3 seeds = 13.17%
#   Standard deviation = 0.45%
#   So the true value is somewhere between ~12.7% and ~13.6%
#   Large ± = inconsistent results. Small ± = consistent results.
#
# === KEY FINDINGS ===
#
# FINDING 1: Stability helps stability (obvious but important)
#   B vs A (scratch): RMSSC 0.305 vs 0.379 → 19.5% improvement!
#   Stabilized forecasts are much smoother.
#   Cost: slight sMAPE increase (13.23 vs 13.17) but within noise.
#
# FINDING 2: Transfer learning improves accuracy
#   C vs A (standard): sMAPE 13.02 vs 13.17 → 1.1% improvement
#   D vs B (stabilized): sMAPE 13.01 vs 13.23 → 1.7% improvement
#   TL helps MORE for stabilized models! (1.7% vs 1.1%)
#
# FINDING 3: Stability + TL = best combination (THESIS ANSWER)
#   D vs C: sMAPE 13.01 vs 13.02 → same accuracy (stability doesn't hurt!)
#   D vs C: RMSSC 0.310 vs 0.326 → 4.9% better stability
#   → YES, stability improves TL without hurting accuracy ✓
#
# FINDING 4: Stabilized models are more CONSISTENT
#   Scratch Standard: sMAPE ± 0.45 (jumps across seeds)
#   Scratch Stabilized: sMAPE ± 0.07 (very consistent)
#   TL Stabilized: sMAPE ± 0.01 (!!) (almost identical across seeds)
#
# FINDING 5: Best model overall = D (TL Stabilized)
#   Lowest sMAPE (13.01%)
#   Lowest RMSSE (1.261)
#   Near-lowest RMSSC (0.310, only B is lower at 0.305)
#   Most consistent (±0.01 sMAPE)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 9: CODE WALKTHROUGH — Every File Explained                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# === FILE STRUCTURE ===
# NBEATSS_thesis/
# ├── main.py                    ← ENTRY POINT: configure and run experiments
# ├── run_experiments.py         ← AUTOMATION: runs all 18 experiments
# ├── experiment_results.csv     ← RESULTS: all metrics saved here
# ├── data/
# │   ├── raw/
# │   │   └── Monthly_clean.csv  ← M3 raw data (861 series)
# │   └── processed/
# │       └── *.pt files         ← Preprocessed PyTorch datasets (cached)
# ├── src/
# │   ├── data/
# │   │   ├── M3.py              ← M3 data loading pipeline
# │   │   ├── M4.py              ← M4 data loading pipeline
# │   │   └── utils/
# │   │       └── _utils.py      ← Data preprocessing utilities
# │   ├── methods/
# │   │   └── NBEATSS.py         ← THE MODEL: N-BEATS-S implementation
# │   └── utils/
# │       ├── metrics.py          ← RMSSE and sMAPE calculations
# │       ├── callbacks.py        ← WandB logging, plotting, warnings
# │       └── plotting.py         ← Forecast visualization
# └── wandb/
#     └── offline-run-*/          ← One folder per experiment run
#         ├── files/output.log    ← Training output and final metrics
#         └── checkpoints/last.ckpt ← Saved model weights
#
#
# === main.py LINE BY LINE ===
#
# Lines 1-15: COMMENTS
#   Explains three design decisions:
#   1. Evaluation uses only forecasts (no data leakage)
#   2. Processed datasets are cached to save time
#   3. Architecture follows original N-BEATS paper
#
# Lines 18-28: IMPORTS (what we install/load)
#   - os: file system operations
#   - LitNBEATSS: our model class
#   - callbacks: helper functions for logging
#   - lightning: PyTorch Lightning (simplifies training code)
#   - wandb: Weights & Biases (experiment tracking tool)
#   - torch: PyTorch (the deep learning framework)
#
# Lines 30-33: wandb.login() and project name
#   Connects to Weights & Biases for experiment tracking.
#   project_name = "NBEATSS_thesis" (folder name for checkpoints)
#
# Lines 36-48: DATASET CONFIG
#   dataset = "M3" or "M4"     ← which dataset to use
#   subset = "Monthly"          ← we only use monthly frequency
#   validation_periods = 18     ← last 18 months reserved for validation
#   test_periods = 18           ← last 18 months reserved for testing
#
# Lines 50-52: MODEL LOADING CONFIG
#   load_model = True/False     ← start from checkpoint or random?
#   update_loaded_model_specific_training_and_eval_hparams = True/False
#     ← when loading a model, do we override its learning rate etc.?
#
# Lines 54-65: ARCHITECTURE CONFIG
#   If load_model=True: just specify which checkpoint to load (model_id)
#   If load_model=False: define the architecture from scratch
#     backcast_length_multiplier = 8  ← lookback = 8 × 6 = 48 months
#     forecast_length = 6             ← predict 6 months ahead
#     hidden_layer_units = 32         ← width of hidden layers
#     n_blocks = 3                    ← number of N-BEATS blocks
#     ensemble_size = 1               ← number of model copies (1 = off)
#
# Lines 67-80: TRAINING CONFIG
#   eval_mode = 'test'          ← evaluate on test set (not validation)
#   random_seed = 2956          ← controls all randomness
#   batch_size = 32             ← process 32 series at a time
#   lambda_stability = 0.02    ← stability penalty weight
#   learning_rate = 1e-5       ← step size for weight updates
#   explr_gamma = 0.97         ← LR decay per epoch
#   ema_decay = 0.99           ← EMA smoothing strength
#
# Lines 82-86: TRAINER CONFIG
#   max_norm = 1.0             ← gradient clipping threshold
#   batches_per_epoch = 50     ← number of batches per epoch
#   max_epochs = 15            ← total training rounds
#
# Lines 88-92: OTHER CONFIG
#   GPU precision setting, forecast saving, plotting (all off for speed)
#
# === BELOW LINE 105 (DO NOT MODIFY) ===
#
# Lines 107-108: SEED EVERYTHING
#   L.seed_everything(random_seed, workers=True)
#   ← locks ALL randomness: torch, numpy, python random, dataloader workers
#
# Lines 110-130: MODEL INITIALIZATION
#   If load_model == True:
#     Load weights from .ckpt file
#     Extract architecture params (forecast_length, etc.)
#     Optionally override training params (LR, lambda, etc.)
#   Else:
#     Create new model with random weights
#
# Lines 132-155: DATA LOADING
#   Import load_data from M3.py or M4.py depending on dataset
#   Call load_data() → returns 5 dataloaders
#   (train, validation, validation_target, trainandvalidation, test_target)
#
# Lines 170-175: WANDB LOGGER + CALLBACKS
#   wandb_logger = WandbLogger(...) ← tracks all metrics
#   ModelSummary ← prints model architecture
#   ModelCheckpoint ← saves weights after training
#   LoadModelWarning ← prints which model was loaded
#
# Lines 180-220: TRAINING + EVALUATION
#   If eval_mode == 'test':
#     trainer.fit(model, trainandvalidation_dataloader)  ← TRAIN
#     trainer.test(model, test_dataloader, "last")       ← TEST
#   If eval_mode == 'validation':
#     trainer.fit(model, train_dataloader, val_dataloader)
#     trainer.test(model, val_dataloader_target, "best")
#
# Lines 222-250: LOG HYPERPARAMETERS
#   Save all config to wandb for record-keeping
#
#
# === NBEATSS.py LINE BY LINE ===
#
# Lines 1-17: IMPORTS
#   torch, nn (neural network layers), F (functions), LightningModule
#
# Lines 22-64: NBEATS_block CLASS
#   One building block. 4 shared layers + 2 task layers + 2 output layers.
#   Input: 48 values → Output: backcast (48) + forecast (6)
#
# Lines 78-99: NBEATS_module CLASS
#   Stacks multiple blocks together. Implements residual connections.
#   Each block subtracts its backcast from the input (residual learning).
#   All block forecasts are summed.
#
# Lines 103-145: LitNBEATSS.__init__
#   Creates the model and its EMA copy.
#   self.model = the actual model (used during training)
#   self.ema_model = smoothed copy (used during evaluation)
#
# Lines 147-175: training_step
#   Called once per batch during training.
#   Computes loss, logs metrics, returns loss for backpropagation.
#
# Lines 177-207: validation_step
#   Called during validation. Similar to training but no augmentation,
#   uses EMA model, doesn't backpropagate.
#
# Lines 209-237: test_step
#   Called during testing. Computes all 4 metrics (RMSSE, sMAPE, RMSSC, sMAPC).
#   Returns forecasts for potential saving/plotting.
#
# Lines 239-354: _get_losses (THE CORE METHOD)
#   This is where everything happens:
#   1. Extract lookback windows and forecast periods from batch
#   2. Apply data augmentation (training only)
#   3. Run through model (or EMA model) for each ensemble member
#   4. Average ensemble forecasts
#   5. Rescale predictions back to original scale
#   6. Compute accuracy loss (RMSSE)
#   7. Compute stability loss (RMSSE between consecutive forecasts)
#   8. Combine: loss = (1-λ) × accuracy + λ × stability
#
# Lines 356-365: configure_optimizers
#   Sets up AdamW optimizer and ExponentialLR scheduler.
#   LR decays each epoch: lr = lr × gamma
#
# Lines 367-372: optimizer_step
#   Ensemble-specific: only updates one ensemble member per batch.
#   With ensemble_size=1, this does nothing special.
#
# Lines 374-378: on_train_batch_end
#   Updates EMA model after every training batch:
#   ema = 0.99 × ema + 0.01 × model
#
#
# === metrics.py ===
#
# RMSSE_calculation(forecast, actual, scaling_constant):
#   1. Compute squared errors: (actual - forecast)²
#   2. Scale by naive forecast error: / scaling_constant
#   3. Average across forecast horizon: mean(...)
#   4. Take square root: sqrt(...)
#   5. Clamp to [0, 5] (prevent extreme outliers)
#   6. Average across batch: mean(...)
#
# sMAPE_calculation(forecast, actual):
#   1. Absolute error: |actual - forecast|
#   2. Denominator: |actual| + |forecast| + 0.001 (avoid ÷0)
#   3. Ratio: error / denominator
#   4. Average and multiply by 200 (makes it percentage, 0-200%)
#
#
# === M3.py ===
#
# load_data() function:
#   1. Check if processed data exists (*.pt files)
#   2. If yes: load from cache (fast!)
#   3. If no:
#      a. Read CSV file (Monthly_clean.csv)
#      b. Convert wide format to long format (one row per observation)
#      c. Add lagged values (shifted by 1 period)
#      d. Pad short series with zeros
#      e. Split into train/validation/test
#      f. Normalize (zero-mean, unit-variance)
#      g. Create PyTorch TimeSeriesDataSets
#      h. Save processed data to cache
#   4. Convert TimeSeriesDataSets to DataLoaders
#   5. Return 5 DataLoaders
#
#
# === run_experiments.py ===
#
# Our automation script that runs all 18 experiments:
#   1. Backs up main.py → main_backup.py
#   2. For each experiment:
#      a. Generates the correct config (dataset, lambda, seed, etc.)
#      b. Overwrites the config section of main.py
#      c. Runs: python main.py
#      d. Finds the new wandb run directory
#      e. Parses metrics from output
#      f. Saves to experiment_results.csv
#   3. For transfer learning: passes pre-trained model_id to fine-tune runs
#   4. Restores main.py from backup when done
#   5. Prints final summary table with mean ± std


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 10: GLOSSARY                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Backcast: A block's reconstruction of its input. Used to compute residuals.
#
# Backpropagation: Algorithm that computes gradients (how to adjust weights).
#
# Batch: A group of training samples processed together (32 in our case).
#
# Batch size: How many time series windows we process at once.
#
# Catastrophic forgetting: When fine-tuning destroys knowledge from pre-training.
#   Prevented by using a small learning rate.
#
# Checkpoint (.ckpt): A file containing all model weights. Can be loaded later.
#
# Data augmentation: Randomly modifying training data to improve generalization.
#   Our code adds random shifts and scales during training.
#
# EMA (Exponential Moving Average): A smoothed copy of model weights.
#   Updated slowly: ema = 0.99 × ema + 0.01 × model.
#   Used during evaluation for more stable predictions.
#
# Ensemble: Multiple model copies averaged together for better predictions.
#   Our config uses ensemble_size=1 (disabled).
#
# Epoch: One complete pass through the training data.
#
# Fine-tuning: Continue training a pre-trained model on a new dataset
#   with a smaller learning rate.
#
# Forecast: The model's prediction of future values.
#
# Forecast horizon: How far ahead we predict (6 months in our case).
#
# Gradient: Direction and magnitude for adjusting each weight to reduce loss.
#
# Gradient clipping: Capping gradient magnitude to prevent training instability.
#
# Hyperparameters: Settings you choose before training (LR, batch size, etc.).
#   NOT learned by the model — set by the researcher.
#
# Lambda stability (λ): Weight of stability loss in the total loss function.
#   0.0 = ignore stability, 0.02 = gentle stability penalty.
#
# Learning rate: Step size for weight updates. Larger = faster but rougher.
#
# LeakyReLU: Activation function. Keeps positive values, shrinks negative ones.
#
# Linear layer: output = input × weights + bias. The basic building block.
#
# Lookback window: The past data fed into the model (48 months).
#
# Loss function: A number measuring how bad predictions are. Training minimizes this.
#
# M3/M4: Standard forecasting competition datasets used for benchmarking.
#
# N-BEATS: Neural Basis Expansion Analysis for Time Series.
#   A neural network designed specifically for time series forecasting.
#
# N-BEATS-S: N-BEATS with Stability penalty and EMA. The "S" stands for Stability.
#
# Normalization: Transforming data to have mean=0 and std=1.
#
# Optimizer (AdamW): Algorithm that updates weights using gradients.
#
# Parameters (weights): The learnable numbers in a neural network.
#   Our model has 51,800 of them.
#
# Pre-training: Training a model on a large dataset before fine-tuning.
#
# Residual connection: Subtracting what one block learned from the input
#   to the next block. Lets each block focus on what previous blocks missed.
#
# RMSSC: Root Mean Squared Scaled Change. Stability metric.
#   Measures how much forecasts change between consecutive updates.
#
# RMSSE: Root Mean Squared Scaled Error. Accuracy metric.
#   Measures forecast error relative to naive forecast error.
#
# Seed: A number that controls randomness. Same seed = same results.
#
# sMAPE: Symmetric Mean Absolute Percentage Error. Accuracy metric.
#   Percentage error between forecast and actual.
#
# sMAPC: Symmetric Mean Absolute Percentage Change. Stability metric.
#
# Transfer learning: Using knowledge from one task to improve another.
#   We pre-train on M4 (big) and fine-tune on M3 (small).
#
# WandB (Weights & Biases): Tool for tracking experiments, logging metrics,
#   saving checkpoints. We use it in offline mode.
#
# Weight decay: Slightly shrinking weights each step to prevent overfitting.
#   Built into AdamW optimizer.


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BONUS: WHAT TO TELL YOUR TEAMMATES (THE 2-MINUTE VERSION)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# "We have a neural network called N-BEATS that predicts monthly time series.
#  Our professor added a 'stability term' that penalizes jumpy forecasts.
#  We call this version N-BEATS-S (the S = Stability).
#
#  Our thesis asks: does this stability term still help when we use 
#  transfer learning? Transfer learning = train on a big dataset first (M4, 
#  100K series), then fine-tune on a small dataset (M3, 861 series).
#
#  We ran 4 experiments, each 3 times with different random seeds:
#    A. Train from scratch, no stability        → sMAPE: 13.17% ± 0.45%
#    B. Train from scratch, with stability      → sMAPE: 13.23% ± 0.07%
#    C. Transfer learning, no stability         → sMAPE: 13.02% ± 0.06%
#    D. Transfer learning, with stability       → sMAPE: 13.01% ± 0.01%
#
#  Results: D (TL + Stability) is the best overall:
#    - Most accurate (13.01%)
#    - Most stable forecasts (RMSSC = 0.310 vs 0.326 without stability)
#    - Most consistent across seeds (±0.01%)
#
#  Answer to thesis question: YES, stability helps transfer learning.
#  It makes forecasts smoother without sacrificing accuracy."
