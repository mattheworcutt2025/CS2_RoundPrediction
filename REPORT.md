# CS2 Round Prediction — Full Technical Report

**Course:** MGTA 611 — Course Project
**Team:** Ratul Sarker, Matthew Orcutt
**Date:** March 21, 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data Pipeline](#2-data-pipeline)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Progression](#4-model-progression)
5. [Results Comparison](#5-results-comparison)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Web Application](#7-web-application)
8. [Key Findings](#8-key-findings)
9. [Repository Structure](#9-repository-structure)
10. [Reproducing Results](#10-reproducing-results)

---

## 1. Problem Statement

**Goal:** Predict which team (Counter-Terrorists or Terrorists) will win a given round in Counter-Strike 2 using machine learning and deep learning.

**Why it matters:**
- Live betting markets and esports analytics
- Team strategy optimization during matches
- Understanding what factors drive round outcomes in professional play

**The critical insight that shaped this project:** Predicting at round start (who has more money?) is nearly impossible — economy doesn't determine gunfight outcomes. Predicting from mid-round game state (who's alive, what weapons do they have, is the bomb planted?) is highly accurate. This single realization took us from 53% to 96.71% accuracy.

---

## 2. Data Pipeline

### 2.1 Raw Data

- **Source:** Kaggle CS2 Dataset — professional match demo files parsed into structured data
- **Location:** `D:\CS2_Data\csds\2023\12\10\` (~92GB)
- **Format:** Parquet files per match, each match containing:
  - `round_end` — round outcomes (winner_team_code: 2=CT, 3=T)
  - `player_status` — health, armor, money, weapons per tick
  - `player_info` — team assignments per round
  - `player_death` — kill events with weapon, headshot flag
  - `player_hurt` — damage events
  - `bomb_state` — bomb plant/defuse events
  - `round_state` — scores
  - `header` — map name, player ranks, win counts

### 2.2 Processing Pipeline

**Step 1a — Basic Preprocessing** (`scripts/1a_preprocess.py`)
- Extracts 15 features per round at round start only
- Output: 9,746 rounds from 530 matches
- Used for the LSTM baseline

**Step 1b — Snapshot Extraction** (`scripts/1b_extract_snapshots.py`)
- Extracts game state at multiple points during each round (not just start)
- Snapshot triggers: round start, after each kill, periodic time intervals, bomb plant
- Output: 171,410 snapshots with 104 features each
- This is the dataset all successful models train on

### 2.3 Data Statistics

| Metric | Value |
|--------|-------|
| Matches processed | 530 |
| Rounds extracted | 9,746 |
| Snapshots (mid-round) | 171,410 |
| Features (v1 basic) | 15 |
| Features (v2 snapshots) | 86 |
| Features (v3 final) | 104 |
| Class balance | 49.9% CT wins / 50.1% T wins |
| Train split | 80% (129,016 snapshots) |
| Validation split | 10% (16,127 snapshots) |
| Test split | 10% (16,128 snapshots) |

---

## 3. Feature Engineering

### 3.1 Feature Categories (104 total)

We extract features across 12 categories from the raw game data:

**Player State (13 features)**
- Alive count per team (0-5)
- Health totals and averages
- Armor values, armor count, helmet count
- CT defuser count
- Man advantage (T alive minus CT alive)

**Weapons (14 features)**
- Rifles, AWPs, snipers, SMGs, shotguns, heavy weapons per team
- Players with primary weapon vs pistol-only
- Rifle advantage, AWP advantage

**Utility (12 features)**
- Flashbangs, smokes, HE grenades, molotovs, decoys per team
- Total utility count and advantage

**Economy (8 features)**
- Team money total and average
- Equipment value total and average (actual gear value, not cash)
- Equipment advantage

**Game State (8 features)**
- Bomb planted flag, bomb site (A/B)
- Time since plant, time elapsed, time remaining
- Round progress (0.0 to 1.0)

**Round Type (6 features)**
- Pistol round (round 1 or 13), second round (2 or 14)
- Round type classification: eco (<$2000 avg), force ($2000-3999), full buy ($4000+)
- Force buy indicators

**Map (8 features)**
- One-hot encoding for 8 maps: Dust2, Mirage, Inferno, Nuke, Overpass, Vertigo, Ancient, Anubis

**Player Skill (6 features)**
- Average rank per team, rank differential
- Average competitive wins per team, wins differential

**Event-Based (10 features)**
- Kills this round per team
- First blood indicators (mutually exclusive)
- Headshot kills, AWP kills per team
- Time since last kill

**Damage (4 features)**
- Damage dealt and taken per team

**Historical/Momentum (6 features)** — added in v3
- Won last round flags
- Win streak and loss streak per team

### 3.2 Feature Importance (from XGBoost)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | man_advantage | 29.8% |
| 2 | equipment_advantage | 8.4% |
| 3 | health_advantage | 4.7% |
| 4 | t_alive | 1.6% |
| 5 | kills_this_round_t | 1.5% |
| 6 | t_has_armor_count | 1.5% |
| 7 | is_second_round | 1.1% |
| 8 | ct_has_primary | 1.0% |
| 9 | kills_this_round_ct | 0.9% |
| 10 | t_helmet_count | 0.9% |

**Key insight:** `man_advantage` alone accounts for nearly 30% of the model's decision-making. A 5v3 situation is highly predictive regardless of economy.

### 3.3 Feature Evolution

| Version | Features | What Changed | Impact |
|---------|----------|-------------|--------|
| v1 (basic) | 15 | Round-start only: money, health, armor, scores | 53% (LSTM) |
| v2 (snapshots) | 86 | Mid-round state: weapons, utility, bomb, kills, damage | 95.75% (MLP) |
| v3 (final) | 104 | +17 features: momentum, force buy, AWP kills, streaks | 96.71% (MLP) |

The 17 new features in v3:
- `ct_pistol_only`, `t_pistol_only` — players without primary weapons
- `ct_equipment_avg`, `t_equipment_avg` — per-player equipment value
- `is_force_buy_ct`, `is_force_buy_t` — force buy indicators
- `wins_diff` — competitive wins differential
- `awp_kills_ct`, `awp_kills_t` — AWP-specific kills
- `time_since_last_kill` — pacing/momentum indicator
- `damage_taken_ct`, `damage_taken_t` — damage received
- `ct_won_last_round`, `t_won_last_round` — previous round outcome
- `ct_win_streak`, `t_win_streak` — consecutive wins
- `ct_loss_streak`, `t_loss_streak` — consecutive losses (affects loss bonus economy)

---

## 4. Model Progression

We built 7 models in a deliberate progression, each teaching us something:

### Step 0a: Logistic Regression (Baseline)

**Architecture:** Linear classifier with 104 input features

**Why we started here:** The simplest possible classifier. If logistic regression works, it proves the features have signal. If it doesn't, the relationships are non-linear.

**Result:** 76.08% accuracy, AUC 0.854

**What we learned:** Features have signal (76% >> 50% random), but a linear decision boundary isn't enough. The relationship between game state and outcome is non-linear — a 4v5 with an AWP and good position can beat a 5v4 with pistols.

### Step 0b: K-Nearest Neighbors (K=3)

**Architecture:** Instance-based classifier, K=3 (selected from {3, 5, 7, 9, 11, 15, 21, 31, 51})

**Why KNN:** "Find the most similar game states in history and see who won." No assumptions about data distribution. If similar game states have similar outcomes, KNN will find them.

**Result:** 94.15% accuracy, AUC 0.980

**What we learned:** Similar game states DO reliably predict outcomes. K=3 was optimal — very local patterns matter. This set the bar high and proved that the problem is solvable with these features.

**K search results:**

| K | Val Accuracy |
|---|-------------|
| 3 | 93.77% |
| 5 | 92.50% |
| 7 | 90.82% |
| 9 | 89.43% |
| 11 | 88.18% |
| 15 | 85.76% |
| 21 | 84.01% |
| 31 | 82.27% |
| 51 | 80.40% |

### Step 0c: Random Forest (500 Trees)

**Architecture:** 500 decision trees, max_features='sqrt', min_samples_split=5

**Why Random Forest:** Ensemble of weak learners, handles non-linearity, provides feature importance ranking. Good baseline for tree-based methods.

**Result:** 89.78% accuracy, AUC 0.966

**What we learned:** Random Forest's feature importance confirmed `man_advantage`, `equipment_advantage`, and `health_advantage` as the top predictors. Interestingly, RF underperformed KNN — the bagging approach smooths out the local patterns that KNN captures.

### Step 2: LSTM (Round-Start Data)

**Architecture:** Bidirectional LSTM, 2 layers, 128 hidden units, dropout 0.3
- Input: sequences of 5 rounds, 15 features per round
- FC layers: 256 -> 64 -> 32 -> 1
- 199,553 parameters

**Why LSTM:** Our first deep learning attempt. The hypothesis was that sequential patterns across rounds (momentum, economy cycles) would predict outcomes.

**Result:** 53.21% accuracy — barely better than a coin flip.

**What we learned:** This was the most important failure of the project. Round-start features (economy, health, armor before the round begins) have almost no predictive power. A team with $16,000 loses to a team with $2,000 all the time in CS2. The LSTM couldn't learn anything because the input data (pre-round economy) simply doesn't determine who wins the gunfight. This motivated the switch to mid-round snapshots.

**Overfitting observed:** Train accuracy reached 82.5% while validation peaked at 58.9% — the model memorized training patterns that didn't generalize, because there was no real signal to learn.

### Step 3: XGBoost (Gradient Boosted Trees)

**Architecture:** XGBClassifier, 300 estimators, max_depth=8, learning_rate=0.1

**Why XGBoost:** Industry-standard gradient boosting. Used to validate that the snapshot features have strong signal before investing in deep learning training.

**Result:** 89.39% accuracy, AUC 0.964, 5-fold CV 85.08% +/- 0.20%

**What we learned:** The snapshot features work. XGBoost matched Random Forest (~90%) and provided detailed feature importance. The tight CV variance (0.20%) confirmed the results are stable.

### Step 4: MLP Standard (Deep Learning)

**Architecture:** Feedforward MLP [512, 256, 128, 64], BatchNorm, ReLU, Dropout 0.3
- 219,009 parameters
- Optimizer: AdamW, lr=0.001
- Early stopping on validation accuracy

**Why MLP:** The course requires deep learning. MLP is the natural first step — it's a universal function approximator that can learn the non-linear patterns KNN found, but in a learnable, generalizable way.

**Result:** 93.86% accuracy, AUC 0.989

**What we learned:** Deep learning matches KNN and significantly beats tree-based methods on this task. The MLP learns a smooth decision boundary that generalizes better than trees. The 4% gap over XGBoost (93.9% vs 89.4%) justifies deep learning for this problem.

### Step 5: MLP Tuned (Optuna Hyperparameter Search)

**Architecture:** Found by Optuna — [896, 704, 448, 448], BatchNorm, ReLU, Dropout 0.1
- 1,248,001 parameters
- Optimizer: AdamW, lr=0.000721, weight_decay=0.000576
- Batch size: 1024
- 50 Optuna trials with Bayesian optimization

**Why tune:** The standard MLP used manually chosen hyperparameters. Optuna does intelligent search over architecture (layer count, widths), learning rate, dropout, optimizer, and batch size.

**Result:** 96.71% accuracy, AUC 0.9964

**What we learned:**
- Wider networks (896 neurons) with low dropout (0.1) outperform narrower networks with high dropout
- 4 layers is optimal — Optuna explored 2-5 layers and converged on 4
- AdamW with weight decay is consistently preferred over Adam
- Batch size 1024 enables better gradient estimates on GPU

**Tuning was GPU-accelerated:** We optimized the training loop to pre-load all data on GPU and use manual batching (no DataLoader overhead), achieving 30x speedup — 50 trials completed in ~5 minutes instead of 2+ hours.

---

## 5. Results Comparison

### 5.1 Single Train/Test Split

| Step | Model | Test Accuracy | AUC-ROC | Parameters |
|------|-------|:------------:|:-------:|:----------:|
| 0a | Logistic Regression | 76.08% | 0.854 | 104 |
| 0b | KNN (K=3) | 94.15% | 0.980 | — |
| 0c | Random Forest (500) | 89.78% | 0.966 | — |
| 2 | LSTM (round-start) | 53.21% | — | 199,553 |
| 3 | XGBoost | 89.39% | 0.964 | — |
| 4 | MLP Standard | 93.86% | 0.989 | 219,009 |
| **5** | **MLP Tuned (Optuna)** | **96.71%** | **0.9964** | **1,248,001** |

### 5.2 5-Fold Stratified Cross-Validation

| Model | CV Accuracy | Std |
|-------|:----------:|:---:|
| Logistic Regression | 76.51% | +/-0.16% |
| KNN (K=3) | 93.79% | +/-0.10% |
| Random Forest | 89.80% | +/-0.11% |
| XGBoost | 90.40% | +/-0.18% |
| **MLP Tuned** | **95.83%** | **+/-0.08%** |

The MLP Tuned has both the highest accuracy AND the tightest variance — the most reliable model.

### 5.3 Confusion Matrix (MLP Tuned, Test Set)

```
              Predicted
            T Win    CT Win
Actual T   [7744      334]
       CT  [ 197     7853]
```

- T Win precision: 97.5%, recall: 95.9%
- CT Win precision: 95.9%, recall: 97.6%
- Both classes predicted with balanced precision/recall

### 5.4 What Each Model Taught Us

| Model | Lesson |
|-------|--------|
| Logistic Regression | Features have signal, but relationships are non-linear |
| KNN | Similar game states predict outcomes — local patterns matter |
| Random Forest | man_advantage is the #1 feature by far |
| LSTM | Round-start economy is useless for prediction |
| XGBoost | Snapshot features validated, stable 85% CV accuracy |
| MLP Standard | Deep learning captures patterns trees miss (+4% over XGBoost) |
| MLP Tuned | Wider/deeper with low dropout wins, 96.71% with Optuna |

---

## 6. Hyperparameter Tuning

### 6.1 Search Space

| Parameter | Range | Best Value |
|-----------|-------|:----------:|
| Number of layers | 2 to 5 | 4 |
| Neurons per layer | 64 to 1024 (step 64) | [896, 704, 448, 448] |
| Dropout | 0.1 to 0.5 (step 0.05) | 0.1 |
| Learning rate | 1e-4 to 1e-2 (log scale) | 7.21e-4 |
| Batch size | {1024, 2048, 4096} | 1024 |
| Weight decay | 1e-6 to 1e-2 (log scale) | 5.76e-4 |
| Optimizer | {Adam, AdamW} | AdamW |

### 6.2 Search Strategy

- **Optuna** with Bayesian optimization (TPE sampler)
- **MedianPruner:** Kills bad trials early if they're below median at that epoch
  - n_startup_trials=5 (let 5 trials finish before pruning)
  - n_warmup_steps=10 (don't prune in first 10 epochs)
- **30 epochs per trial** with patience 8 (just enough to rank trials)
- **Best trial retrained** for 200 epochs with patience 25

### 6.3 GPU Optimization

The tuning loop was optimized for GPU throughput:

1. **Pre-load all data to GPU** — eliminates CPU-to-GPU transfer per batch
2. **Manual batching with torch.randperm** — no DataLoader overhead
3. **Single forward pass for validation** — process entire val set at once
4. **Result:** 5 seconds per trial (30x faster than naive implementation)

### 6.4 Tuning Progress

The top 5 trials out of 50:

| Trial | Val Accuracy | Layers | Architecture |
|:-----:|:-----------:|:------:|:------------:|
| 41 | 94.80% | 5 | [896, 832, 576, 512, 192] |
| 45 | 94.78% | 5 | [896, 896, 576, 384, 128] |
| 43 | 94.74% | 5 | [960, 896, 576, 512, 128] |
| 39 | 94.73% | 5 | [1024, 832, 640, 448, 192] |
| 34 | 94.72% | 5 | [960, 896, 576, 448, 128] |

All top models converged on wide first layers (896+) and low dropout (0.1-0.15).

---

## 7. Web Application

### 7.1 Architecture

The prediction model is deployed as a browser-based web application:

- **Framework:** Next.js 16 (React)
- **Styling:** Tailwind CSS with CS2-themed design
- **Model Runtime:** ONNX Runtime Web (runs entirely in the browser)
- **Hosting:** Vercel (static deployment, zero serverless functions)
- **URL:** https://cs2predict.vercel.app

### 7.2 Model Export

The PyTorch model was exported to ONNX format:
- **Size:** 4.8 MB (single file, weights embedded)
- **Opset version:** 18
- **Input:** 104 float32 features
- **Output:** 1 float32 probability (CT win probability)

### 7.3 Feature Vector Construction

The web app builds a 104-feature vector from user inputs (sliders for alive count, health, weapons, money, etc.). Since users don't input all 104 features directly, some are derived:

- **Equipment value** — estimated from weapon loadout (rifle ~$2900, AWP ~$4750, armor ~$1000)
- **Utility** — scaled by buy type (eco = 0, force = partial, full buy = full)
- **Kills/damage** — inferred from alive counts (5 - alive = kills by other team)
- **Round type** — classified from average team money (<$2000 eco, <$4000 force, else full)

### 7.4 Audit and Testing

An audit of the feature vector construction found **52 out of 104 features** had bugs in the initial implementation. Critical fixes:

| Bug | Severity | Fix |
|-----|----------|-----|
| CT/T kills were swapped | Critical | `ct_kills = 5 - t_alive` (not `5 - ct_alive`) |
| Round duration was 175s | Critical | Changed to 115s (CS2 competitive) |
| Pistol round detection: rounds 16/17 | Critical | Changed to 13/14 (CS2 half starts at round 13) |
| Equipment value = money * 0.6 | Critical | Real estimates from weapon loadout |
| Force-buy threshold $3500 | Medium | Changed to $4000 to match training |
| Damage per kill = 30 | High | Changed to ~100 (actual HP) |
| First blood could be both teams | High | Made mutually exclusive |

**54 unit tests** were written covering:
- Kill attribution (CT vs T)
- First blood mutual exclusivity
- Damage cross-assignment
- Time/round progress calculation
- Pistol round detection
- Round type classification
- Map one-hot encoding
- Equipment value estimation
- Utility scaling
- Score features
- Symmetry (equal teams = zero advantages)
- Edge cases (all dead, zero money, max money)

### 7.5 UI Design

- **CS2-themed:** Official CS2 key art as background, yellowish dark overlay
- **T side:** Gold/yellow accents (matching in-game)
- **CT side:** Blue accents (matching in-game)
- **Inputs:** Sliders for alive, health, armor, rifles, AWPs, money, score per team
- **Round state:** Map selector, round number, time remaining, bomb planted toggle
- **Output:** Large winner banner with team name, percentage bar, advantage description

---

## 8. Key Findings

### 8.1 The Fundamental Insight

Round-start data (economy, health before the round) predicts almost nothing (53%). Mid-round game state (who's alive right now, with what weapons) predicts almost everything (96.7%). This mirrors how CS2 actually works — a pistol round win with $800 can beat a full buy with $16,000 if the players are better positioned.

### 8.2 Feature Importance Hierarchy

1. **Player count differential** (man_advantage) — 29.8% importance. A numbers advantage is the single strongest predictor.
2. **Equipment value differential** — 8.4%. Having better gear matters.
3. **Health differential** — 4.7%. Damaged players are at a disadvantage.
4. **Everything else** — long tail of smaller effects (weapons, utility, map, streaks)

### 8.3 Deep Learning vs Traditional ML

| Method | Best Accuracy | Why |
|--------|:------------:|-----|
| Linear (Logistic Regression) | 76.1% | Can't capture non-linear weapon/position interactions |
| Tree-based (XGBoost) | 89.4% | Good at feature interactions but local splits miss smooth patterns |
| Instance-based (KNN) | 94.2% | Excellent at finding similar game states, but slow and memory-heavy |
| **Neural Network (MLP)** | **96.7%** | **Learns smooth non-linear decision boundary, generalizes best** |

### 8.4 Why MLP Beats KNN Despite Simpler Architecture

KNN at K=3 means "look at the 3 most similar game states ever seen." This works well but:
- Requires storing all 129K training points in memory
- Prediction time scales with dataset size
- Can't generalize beyond seen examples

MLP compresses the training data into 1.2M learned parameters that capture the underlying patterns, not specific examples. It generalizes to game states it has never seen.

---

## 9. Repository Structure

```
CS2_RoundPrediction/
├── README.md                          # Project overview with results table
├── REPORT.md                          # This document
├── scripts/
│   ├── 0a_logistic_regression.py      # Baseline: linear classifier
│   ├── 0b_knn.py                      # Baseline: K-nearest neighbors
│   ├── 0c_random_forest.py            # Baseline: ensemble of decision trees
│   ├── 1a_preprocess.py               # Basic feature extraction (15 features)
│   ├── 1b_extract_snapshots.py        # Mid-round snapshot extraction (104 features)
│   ├── 2_train_lstm.py                # LSTM on round-start data (failed experiment)
│   ├── 3_train_xgboost.py             # XGBoost on snapshot data
│   ├── 4_train_mlp.py                 # MLP deep learning model
│   ├── 5_tune_mlp.py                  # Optuna hyperparameter tuning (GPU)
│   └── 6_generate_results.py          # Generate all evaluation artifacts
├── config/
│   ├── weapon_codes.json              # CS2 weapon ID to name mapping
│   └── weapon_mapping.json            # Weapon category classifications
├── docs/
│   ├── PLAN.md                        # Original execution plan
│   ├── RESEARCH.md                    # Literature review
│   └── FEATURES.md                    # Feature documentation
├── models/                            # Trained model weights (.gitignore'd except scalers)
│   ├── scaler_params.json             # StandardScaler mean/std for MLP standard
│   └── scaler_params_tuned.json       # StandardScaler for tuned MLP
├── results/
│   ├── logistic_regression_results.json
│   ├── knn_results.json
│   ├── random_forest_results.json
│   ├── lstm_results.json
│   ├── xgboost_results.json
│   ├── mlp_standard_results.json
│   ├── tuning_results.json            # All 50 Optuna trials + final metrics
│   ├── model_comparison.json          # All 7 models side-by-side
│   ├── model_evaluation.png           # ROC curves, confusion matrix, accuracy bars
│   ├── tuning_analysis.png            # Optuna progress, accuracy by depth
│   └── xgboost_feature_importance.png
├── notebooks/
│   └── data_exploration.ipynb
├── web/                               # Next.js web application
│   ├── app/
│   │   ├── page.tsx                   # Main UI with sliders and prediction
│   │   ├── predict.ts                 # Feature vector construction + ONNX inference
│   │   ├── layout.tsx                 # HTML layout
│   │   └── globals.css                # CS2-themed styling
│   ├── public/
│   │   ├── mlp_tuned.onnx            # Exported model (4.8MB)
│   │   ├── scaler_params.json         # Feature scaling parameters
│   │   └── feature_names.json         # 104 feature names in order
│   ├── __tests__/
│   │   └── predict.test.ts            # 54 unit tests
│   └── __mocks__/
│       └── onnxruntime-web.ts         # Mock for testing
└── data/                              # .gitignore'd (large files)
    ├── X_snapshots.npy                # Feature matrix (171K x 104)
    ├── y_snapshots.npy                # Labels (171K)
    └── feature_names.json             # Feature name order
```

---

## 10. Reproducing Results

### 10.1 Requirements

- Python 3.12+
- PyTorch 2.10+ with CUDA (for GPU training)
- scikit-learn, XGBoost, Optuna, Polars
- Node.js 18+ (for web app)
- Raw CS2 data at `D:\CS2_Data\csds\2023\12\10\`

### 10.2 Steps

```bash
# Step 0: Train baseline models
python scripts/0a_logistic_regression.py
python scripts/0b_knn.py
python scripts/0c_random_forest.py

# Step 1: Extract features (requires raw parquet data)
python scripts/1b_extract_snapshots.py

# Step 2-4: Train models
python scripts/2_train_lstm.py          # Uses basic 15-feature data
python scripts/3_train_xgboost.py       # Uses snapshot data
python scripts/4_train_mlp.py           # Uses snapshot data

# Step 5: Hyperparameter tuning (GPU recommended)
python scripts/5_tune_mlp.py --trials 50

# Step 6: Generate all plots and comparison
python scripts/6_generate_results.py

# Web app
cd web
npm install
npm run test    # 54 unit tests
npm run dev     # Local dev server at localhost:3000
```

### 10.3 Hardware Used

- **GPU:** NVIDIA RTX 3060 Ti (8GB VRAM)
- **Training time (MLP tuning):** ~5 minutes for 50 Optuna trials
- **Data extraction:** ~38 minutes for 530 matches

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Data Processing | Polars, PyArrow, NumPy |
| ML Models | scikit-learn (LogReg, KNN, RF), XGBoost |
| Deep Learning | PyTorch 2.10.0+cu128 |
| Tuning | Optuna (Bayesian optimization) |
| Model Export | ONNX (opset 18) |
| Web Framework | Next.js 16, React 19 |
| Browser Inference | ONNX Runtime Web |
| Styling | Tailwind CSS 4 |
| Testing | Jest, ts-jest |
| Hosting | Vercel (static) |
| Version Control | Git, GitHub |

---

*Live demo: [cs2predict.vercel.app](https://cs2predict.vercel.app)*
*Repository: [github.com/ratulsarker/CS2_RoundPrediction](https://github.com/ratulsarker/CS2_RoundPrediction)*
