# CS2 Round Prediction — Deep Learning

**Course:** MGTA621/622 (MMA Field Project)
**Team:** Ratul Sarker, Sam Matthew

Predicts which team (CT or T) will win a given round in Counter-Strike 2 using deep learning on mid-round game state snapshots.

---

## Results

| Model | Test Accuracy | AUC-ROC |
|-------|--------------|---------|
| LSTM (round-start, 15 features) | 53.2% | — |
| XGBoost (mid-round, 86 features) | 89.4% | 0.963 |
| MLP (mid-round, 86 features) | 93.2% | — |
| **MLP Tuned (Optuna, 50 trials)** | **95.75%** | **0.9945** |

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
│   ├── preprocess.py           # Basic feature extraction (15 features)
│   ├── extract_snapshots.py    # Mid-round snapshot extraction (86 features)
│   ├── train_lstm.py           # LSTM model (baseline)
│   ├── train_mlp.py            # MLP model
│   ├── train_xgboost.py        # XGBoost baseline
│   └── tune_mlp.py             # Optuna hyperparameter tuning
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
│   ├── tuning_results.json     # Optuna tuning output
│   ├── training_results.json   # LSTM training metrics
│   ├── xgboost_results.json    # XGBoost metrics
│   ├── training_curves.png     # Loss/accuracy plots
│   └── xgboost_feature_importance.png
├── notebooks/
│   └── data_exploration.ipynb
├── data/                       # .gitignore'd (large files)
└── .gitignore
```

---

## Quick Start

```bash
# 1. Extract snapshots from raw parquet data
python scripts/extract_snapshots.py

# 2. Train MLP
python scripts/train_mlp.py

# 3. Tune hyperparameters (requires GPU)
python scripts/tune_mlp.py --trials 50
```

**Requirements:** Python 3.12, PyTorch 2.10 (CUDA), Polars, scikit-learn, XGBoost, Optuna

---

## Tech Stack

- **Framework:** PyTorch 2.10.0+cu128
- **GPU:** NVIDIA RTX 3060 Ti (8GB)
- **Data Processing:** Polars + PyArrow
- **Tuning:** Optuna (Bayesian optimization)
- **Baselines:** XGBoost, scikit-learn
