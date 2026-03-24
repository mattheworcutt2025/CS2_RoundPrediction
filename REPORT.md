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

We built 7 models in a deliberate progression. Each model was chosen to answer a specific question about the data and to build on what the previous model taught us.

### 4.0 Why This Progression?

Our strategy followed a first-principles approach to classification:

1. **Start linear** — if a linear model works, the problem is simple. If not, we know we need non-linear methods.
2. **Try non-parametric** — KNN makes zero assumptions about the data. It tells us the theoretical ceiling if we can find the right patterns.
3. **Try ensemble trees** — Random Forest and XGBoost are the industry standard for tabular data. They handle non-linearity through recursive partitioning.
4. **Try deep learning** — neural networks learn arbitrary function mappings from data. If trees plateau, a neural network can find patterns trees structurally cannot.
5. **Tune the winner** — once we identify the best model family, systematic hyperparameter search squeezes out the remaining performance.

### Step 0a: Logistic Regression (Baseline)

**Architecture:** Linear classifier with 104 input features

**Why logistic regression first:** Logistic regression draws a single flat hyperplane through the 104-dimensional feature space. It computes P(CT wins) = σ(w₁·ct_alive + w₂·t_alive + ... + w₁₀₄·t_loss_streak + b), where σ is the sigmoid function. Every feature contributes independently and linearly — there are no interaction terms. This is the diagnostic baseline: if it gets 50%, our features are useless. If it gets 100%, the problem is trivially linear. Anything in between tells us exactly how much signal is linear vs non-linear.

**Result:** 76.08% accuracy, AUC 0.854

**What we learned:** 76% >> 50% means the features carry real signal. But 76% << 94% (KNN) means the decision boundary is not a flat hyperplane. In CS2 terms: a 4v5 with an AWP and planted bomb can beat a 5v4 with pistols — the interaction between player count, weapon type, and bomb status matters, and logistic regression cannot model interactions without explicit feature engineering.

### Step 0b: K-Nearest Neighbors (K=3)

**Architecture:** Instance-based classifier, K=3 (selected from {3, 5, 7, 9, 11, 15, 21, 31, 51})

**Why KNN:** KNN is the most assumption-free classifier possible. It stores the entire training set and at prediction time finds the K closest data points (by Euclidean distance in normalized feature space) and takes a majority vote. This is equivalent to asking: "Find the most similar game states in history and see who won." If the mapping from game state to outcome is smooth (similar states → similar outcomes), KNN will capture it regardless of the functional form. It is the non-parametric gold standard.

**Why K=3:** Smaller K means the model uses only the most local neighborhood. K=3 won because CS2 game states are highly specific — a 3v2 post-plant on Dust2 with AWP is very different from a 3v2 retake with rifles. Larger K averages over dissimilar situations, diluting the signal.

**Result:** 94.15% accuracy, AUC 0.980

**What we learned:** Similar game states DO reliably predict outcomes. The 94% accuracy set the performance ceiling that all other models target. However, KNN has fundamental limitations: (1) it stores all 129K training points in memory, (2) prediction time is O(n·d) per query, and (3) it cannot generalize beyond the exact training examples — it interpolates, not extrapolates.

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

**Why Random Forest:** Random Forest is a bagging ensemble — it trains 500 independent decision trees, each on a random bootstrap sample of the data and a random subset of features (√104 ≈ 10 features per split). The final prediction is a majority vote across all trees. This reduces variance (individual trees overfit, but the average does not) and provides a built-in feature importance ranking via mean decrease in Gini impurity.

We chose RF as our tree-based baseline because it is robust, rarely overfits catastrophically, and its feature importances would help us understand which game-state variables drive predictions.

**Result:** 89.78% accuracy, AUC 0.966

**What we learned:** RF confirmed `man_advantage`, `equipment_advantage`, and `health_advantage` as the top predictors. RF underperformed KNN (89.8% vs 94.1%) — this is significant. Trees partition the feature space into axis-aligned rectangles. Each split asks "is feature X above threshold T?" This means trees cannot naturally capture diagonal or curved decision boundaries. In CS2 terms: the interaction "3 alive + AWP + bomb planted = favored" requires multiple sequential splits to approximate what is really a smooth, continuous relationship.

### Step 2: LSTM (Round-Start Data)

**Architecture:** Bidirectional LSTM, 2 layers, 128 hidden units, dropout 0.3
- Input: sequences of 5 rounds, 15 features per round
- FC layers: 256 → 64 → 32 → 1
- 199,553 parameters

**Why LSTM:** Long Short-Term Memory networks are designed for sequential data. The hypothesis was that temporal patterns across rounds — momentum swings, economy cycles (win → buy → lose → eco → force), and map-side advantages — would carry predictive signal beyond what a single round's features show. LSTMs maintain a hidden state that can remember relevant information across time steps and forget irrelevant information via learned gating mechanisms.

**Result:** 53.21% accuracy — barely better than a coin flip.

**What we learned:** This was the most important failure of the project. It proved that round-start features (money, health, armor before the round begins) contain essentially no signal about who will win the gunfight. In CS2, a team with $16,000 routinely loses to a team with $2,000 — the pre-round economy tells you what weapons they can buy, but not who will hit their shots, use utility effectively, or hold the right angles. The LSTM architecture itself was not the problem; the input data was.

**Overfitting observed:** Train accuracy reached 82.5% while validation peaked at 58.9%. The model memorized noise in the training data because there was no real signal to learn. This gap (train >> val) is a classic diagnostic for "the model is powerful enough, but the data doesn't contain the answer."

This failure directly motivated the switch to mid-round snapshots — capturing who is alive *during* the round, not just who had money *before* it.

### Step 3: XGBoost (Gradient Boosted Trees)

**Architecture:** XGBClassifier, 300 estimators, max_depth=8, learning_rate=0.1

**Why XGBoost over Random Forest:** While Random Forest uses bagging (parallel, independent trees), XGBoost uses boosting (sequential, corrective trees). Each new tree in XGBoost is trained on the residual errors of all previous trees — it focuses on the examples the ensemble currently gets wrong. This typically yields better accuracy than RF because it directly optimizes the loss function via gradient descent in function space. XGBoost also includes L1/L2 regularization on the tree structure, which prevents overfitting better than RF's simple bootstrap approach.

We used XGBoost as the tree-based champion to benchmark against our deep learning models. If XGBoost matches or beats a neural network, there is no justification for the added complexity of deep learning.

**Result:** 89.39% accuracy, AUC 0.964, 5-fold CV 85.08% +/- 0.20%

**What we learned:** XGBoost matched Random Forest (~90%) but could not close the gap to KNN (94%). The tight CV variance (0.20%) confirmed the results are stable across folds. The 90% plateau of tree-based methods on this data suggested a structural limitation — trees partition the feature space into axis-aligned rectangles, and the true decision boundary in CS2 game-state space is smoother and more complex than rectangular partitions can efficiently approximate.

### Step 4: MLP Standard (Deep Learning)

**Architecture:** Feedforward MLP [512, 256, 128, 64], BatchNorm, ReLU, Dropout 0.3
- 219,009 parameters
- Optimizer: AdamW, lr=0.001
- Early stopping on validation accuracy

#### What Is an MLP? (Artificial Neural Network Fundamentals)

A Multi-Layer Perceptron (MLP) is the foundational type of Artificial Neural Network (ANN). At its core, an ANN is a computational model inspired by biological neurons. Here is what that means concretely:

**A single artificial neuron** computes: output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + bias). It takes a weighted sum of its inputs and passes the result through a non-linear activation function (in our case, ReLU: max(0, x)). The weights w₁...wₙ are the learned parameters — training the network means finding the weight values that minimize prediction error.

**An MLP stacks neurons into layers:**

```
Input (104 features)
  ↓  × 104→512 weights + 512 biases
Layer 1: 512 neurons → BatchNorm → ReLU → Dropout(30%)
  ↓  × 512→256 weights + 256 biases
Layer 2: 256 neurons → BatchNorm → ReLU → Dropout(30%)
  ↓  × 256→128 weights + 128 biases
Layer 3: 128 neurons → BatchNorm → ReLU → Dropout(30%)
  ↓  × 128→64 weights + 64 biases
Layer 4: 64 neurons → BatchNorm → ReLU → Dropout(30%)
  ↓  × 64→1 weight + 1 bias
Output: 1 neuron → Sigmoid → P(CT wins)
```

Each component serves a specific purpose:
- **Linear layers** (matrix multiplication + bias): the actual learned transformation. These are the 219,009 parameters the network learns.
- **ReLU activation** (max(0, x)): introduces non-linearity. Without this, stacking linear layers would just be one big linear layer — equivalent to logistic regression. ReLU is what gives the network its power to model curved decision boundaries.
- **BatchNorm**: normalizes the values between layers so training is faster and more stable. Without it, the distribution of values shifts as weights update (internal covariate shift), slowing convergence.
- **Dropout** (randomly zero 30% of neurons during training): forces the network to not rely on any single neuron. This is regularization — it prevents overfitting by making the network learn redundant representations.
- **Sigmoid** (squishes output to [0, 1]): converts the final value to a probability.

**Training** works via backpropagation and gradient descent:
1. Feed a batch of game-state snapshots through the network (forward pass)
2. Compare predictions to actual labels using Binary Cross-Entropy loss
3. Compute how much each weight contributed to the error (backward pass / backpropagation)
4. Adjust each weight by a small step in the direction that reduces the error (gradient descent via AdamW optimizer)
5. Repeat for all batches, for many epochs, until validation accuracy stops improving

**The Universal Approximation Theorem** guarantees that an MLP with a single hidden layer of sufficient width can approximate any continuous function to arbitrary precision. In practice, deeper networks (multiple layers) learn hierarchical representations more efficiently — earlier layers detect simple patterns (e.g., "is this team outnumbered?"), later layers combine them into complex patterns (e.g., "outnumbered but has AWP, bomb planted, and time advantage").

#### Why MLP Beats Trees (XGBoost, Random Forest)

The 7+ percentage point gap between MLP (96.7%) and XGBoost (89.4%) is not an accident — it reflects a fundamental structural difference in how these models partition the feature space:

1. **Axis-aligned vs arbitrary decision boundaries.** Decision trees split the feature space along one feature at a time: "is ct_alive > 3?" then "is equipment_advantage > 5000?" Each split is perpendicular to a single axis. To model a diagonal boundary like "CT wins when (2 × alive + weapon_value/1000) > threshold," a tree needs many sequential splits to approximate the diagonal with a staircase pattern. An MLP learns the diagonal directly as a single linear combination in the first layer.

2. **Smooth vs discontinuous predictions.** Trees produce piecewise-constant predictions — every data point in the same leaf gets the same prediction. The transition from "T favored" to "CT favored" is a hard step. MLPs produce smooth, continuous predictions — the probability changes gradually as features change. In CS2, the true win probability is smooth: going from 4v5 to 5v5 doesn't suddenly flip the outcome; it gradually shifts the odds. The MLP's smooth sigmoid output naturally captures this.

3. **Feature interactions.** XGBoost captures feature interactions through sequential splits (feature A at depth 1, feature B at depth 2). But each interaction requires additional tree depth, and the number of possible interactions grows exponentially with depth. An MLP captures all pairwise (and higher-order) interactions in a single layer through matrix multiplication — 104 input features × 512 neurons = 53,248 learned interaction weights in Layer 1 alone.

4. **Gradient-based end-to-end optimization.** XGBoost optimizes trees greedily — each split is locally optimal, but the ensemble is not globally optimal. An MLP optimizes all 219,009 parameters simultaneously via gradient descent, finding a globally coordinated solution where every weight works together.

5. **Generalization to unseen states.** Trees can only predict within regions they have seen training data for. MLPs learn a continuous function that interpolates and extrapolates smoothly — they can make reasonable predictions for game states that never appeared in training.

**Result:** 93.86% accuracy, AUC 0.989

**What we learned:** Deep learning matches KNN and significantly beats tree-based methods on this task. The smooth, continuous decision boundary of the MLP captures the underlying structure of CS2 game states better than the rectangular partitions of tree ensembles.

### Step 5: MLP Tuned (Optuna Hyperparameter Search)

**Architecture:** Found by Optuna — [896, 704, 448, 448], BatchNorm, ReLU, Dropout 0.1
- 1,248,001 parameters
- Optimizer: AdamW, lr=0.000721, weight_decay=0.000576
- Batch size: 1024
- 50 Optuna trials with Bayesian optimization

**Why tune:** The standard MLP used manually chosen hyperparameters (layer widths, dropout rate, learning rate). These choices significantly affect performance, and the search space is too large for manual exploration. Optuna uses Bayesian optimization (Tree-structured Parzen Estimator) — it learns from past trials which regions of hyperparameter space are promising and focuses future trials there, converging faster than grid search or random search.

**Result:** 96.71% accuracy, AUC 0.9964

**What we learned:**
- Wider networks (896 neurons) with low dropout (0.1) outperform narrower networks with high dropout. The data has enough signal that regularization should be light — the model needs capacity, not constraints.
- 4 layers is optimal — Optuna explored 2-5 layers and converged on 4. Deeper networks can represent more complex functions, but beyond 4 layers the returns diminish and training becomes harder.
- AdamW with weight decay is consistently preferred over plain Adam. Weight decay (L2 regularization on parameters) prevents any single weight from growing too large, improving generalization.
- Batch size 1024 enables better gradient estimates on GPU — the loss gradient averaged over 1024 samples is a more accurate estimate of the true gradient than over 64 samples, leading to more stable training.

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

### 8.3 Deep Learning vs Traditional ML — A First-Principles View

Each model family makes fundamentally different assumptions about the mapping from game state to outcome:

| Method | Decision Boundary | Best Accuracy | Core Limitation |
|--------|:-:|:---:|-----|
| **Logistic Regression** | Single hyperplane | 76.1% | Cannot model interactions — treats each feature independently |
| **Random Forest** | Axis-aligned rectangles (averaged) | 89.8% | Bagging smooths local patterns; each tree sees random feature subsets |
| **XGBoost** | Axis-aligned rectangles (sequential) | 89.4% | Boosting corrects errors iteratively, but still limited to axis-aligned splits |
| **KNN (K=3)** | Voronoi tessellation | 94.2% | Perfect local accuracy, but stores entire dataset and cannot extrapolate |
| **MLP** | Smooth, arbitrary surface | **96.7%** | Learns continuous function via gradient descent; generalizes to unseen states |

**Why do trees plateau at ~90%?** Every split in a decision tree asks "is feature X > threshold?" This creates rectangular regions. To approximate a diagonal boundary like "CT wins when (alive × 100 + equipment / 50) > 400," trees need dozens of small splits to build a staircase approximation. The MLP learns this diagonal in a single neuron: one weighted sum captures the relationship directly.

**Why does the MLP beat KNN?** KNN at K=3 memorizes the training set and looks up the 3 nearest neighbors. This gives excellent accuracy when the test point is close to training points, but it cannot generalize — it interpolates, never extrapolates. The MLP compresses 129K training examples into 1.2M learned parameters that encode the *rules* (not the examples) of what makes a game state favorable. When the MLP encounters a game state it has never seen (e.g., a novel weapon combination on a map with unusual rank distributions), it can still make a reasonable prediction by applying the learned rules. KNN has no answer for truly novel states.

### 8.4 The Neural Network Advantage — Why It Matters

The MLP's 7-point advantage over XGBoost (96.7% vs 89.4%) comes from three structural properties:

1. **Continuous feature interactions:** The first linear layer computes 512 different weighted combinations of all 104 features simultaneously. Each neuron captures a different "aspect" of the game state — one might encode "firepower advantage," another "utility pressure," another "time-bomb interaction." Trees would need hundreds of splits to approximate what a single layer does in one matrix multiplication.

2. **Hierarchical abstraction:** Layer 1 detects low-level patterns ("is this team outnumbered?"). Layer 2 combines them ("outnumbered but has AWP and bomb planted"). Layer 3 integrates game context ("outnumbered with AWP and bomb planted on CT side of Inferno with 20 seconds left"). Each layer builds on the previous one, creating increasingly abstract representations.

3. **Smooth probability surface:** The sigmoid output layer produces a continuous probability between 0 and 1. As game conditions gradually shift (e.g., a player takes damage, reducing health from 100 to 60), the predicted win probability shifts smoothly. Trees produce step functions — the probability jumps discontinuously when a feature crosses a split threshold. The smooth surface matches the real-world phenomenon better: losing 40 HP doesn't suddenly flip the round; it gradually shifts the odds.

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
