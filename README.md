# CS2 Round Prediction — Deep Learning

**Course:** MGTA621/622 (MMA Field Project)
**Team:** Ratul Sarker, Sam Matthew

Predicts which team (CT or T) will win a given round in Counter-Strike 2 using deep learning on mid-round game state snapshots.

---

## Results

| Step | Model | Test Accuracy | AUC-ROC |
|------|-------|--------------|---------|
| 0a | Logistic Regression | 76.08% | 0.854 |
| 0b | K-Nearest Neighbors (K=3) | 93.79% | 0.978 |
| 0c | Random Forest (500 trees) | 89.85% | 0.966 |
| 2 | LSTM (round-start, 15 features) | 53.21% | — |
| 3 | XGBoost (86 features) | 89.39% | 0.964 |
| 4 | MLP Standard (86 features) | 93.86% | 0.989 |
| 5 | **MLP Tuned (Optuna, 50 trials)** | **95.75%** | **0.9945** |

### Best Model Architecture
- **Type:** 5-layer MLP [896 → 832 → 576 → 512 → 192]
- **Parameters:** 1.7M
- **Optimizer:** AdamW (lr=7.2e-4, weight_decay=2.5e-5)
- **Dropout:** 0.1
- **Training:** 129 epochs (early stopped), GPU-accelerated

### Top Predictive Features
1. `man_advantage` (29.8%) — player count differential
2. `equipment_advantage` (8.4%)
3. `health_advantage` (4.7%)
4. `t_alive`, `kills_this_round_t`, armor/helmet counts

---

## Data

**Source:** Kaggle CS2 Dataset (December 10, 2023)
- **530 matches**, **9,746 rounds**, **161,271 snapshots**
- **86 features** extracted from mid-round game state
- **Class Balance:** 48.9% CT wins / 51.1% T wins

Features include player state, weapons, utility, economy, game state, map indicators, player skill, and derived advantages. See [`docs/FEATURES.md`](docs/FEATURES.md) for the full list.

---

## Project Structure

```
CS2_RoundPrediction/
├── README.md
├── scripts/
│   ├── 0a_logistic_regression.py  # Step 0: Logistic Regression baseline
│   ├── 0b_knn.py                  # Step 0: K-Nearest Neighbors baseline
│   ├── 0c_random_forest.py        # Step 0: Random Forest baseline
│   ├── 1a_preprocess.py           # Step 1: Basic feature extraction (15 features)
│   ├── 1b_extract_snapshots.py    # Step 1: Mid-round snapshot extraction (86 features)
│   ├── 2_train_lstm.py            # Step 2: LSTM model (round-start)
│   ├── 3_train_xgboost.py         # Step 3: XGBoost on snapshots
│   ├── 4_train_mlp.py             # Step 4: MLP deep learning model
│   ├── 5_tune_mlp.py              # Step 5: Optuna hyperparameter tuning
│   └── 6_generate_results.py      # Generate all evaluation artifacts
├── config/
│   ├── weapon_codes.json       # CS2 weapon ID mappings
│   └── weapon_mapping.json     # Weapon category classifications
├── docs/
│   ├── PLAN.md                 # Execution plan & phases
│   ├── RESEARCH.md             # Literature review & strategy
│   └── FEATURES.md             # Feature documentation
├── models/                     # Trained model weights (.gitignore'd)
│   ├── scaler_params.json      # Feature scaling parameters
│   └── scaler_params_tuned.json
├── results/
│   ├── logistic_regression_results.json
│   ├── knn_results.json
│   ├── random_forest_results.json
│   ├── lstm_results.json
│   ├── xgboost_results.json
│   ├── mlp_standard_results.json
│   ├── tuning_results.json
│   ├── model_comparison.json      # All 7 models side-by-side
│   ├── model_evaluation.png       # ROC, confusion matrix, accuracy bars
│   ├── tuning_analysis.png        # Optuna progress
│   └── xgboost_feature_importance.png
├── notebooks/
│   └── data_exploration.ipynb
├── data/                       # .gitignore'd (large files)
└── .gitignore
```

---

## Quick Start

```bash
# Step 0: Train baselines
python scripts/0a_logistic_regression.py
python scripts/0b_knn.py
python scripts/0c_random_forest.py

# Step 1: Data pipeline (requires raw parquet data on D:\CS2_Data)
python scripts/1b_extract_snapshots.py

# Step 2-4: Train models
python scripts/2_train_lstm.py
python scripts/3_train_xgboost.py
python scripts/4_train_mlp.py

# Step 5: Hyperparameter tuning (requires GPU)
python scripts/5_tune_mlp.py --trials 50

# Step 6: Generate all evaluation plots and comparison
python scripts/6_generate_results.py
```

**Requirements:** Python 3.12, PyTorch 2.10 (CUDA), Polars, scikit-learn, XGBoost, Optuna

---

## Tech Stack

- **Framework:** PyTorch 2.10.0+cu128
- **GPU:** NVIDIA RTX 3060 Ti (8GB)
- **Data Processing:** Polars + PyArrow
- **Tuning:** Optuna (Bayesian optimization)
- **Baselines:** XGBoost, scikit-learn
