"""
CS2 Round Prediction - Hyperparameter Tuning with Optuna

What this script does:
    Instead of guessing hyperparameters, Optuna automatically tries many combinations
    and finds the best ones. It's like a smart brute-force search.

    For each "trial", Optuna picks a random combination of:
        - Number of layers (2 to 5)
        - Neurons per layer (64 to 1024)
        - Dropout rate (0.1 to 0.5)
        - Learning rate (0.0001 to 0.01)
        - Batch size (64, 128, 256, 512)
        - Weight decay (0 to 0.01)
        - Optimizer (Adam vs AdamW)

    Then trains the model and checks validation accuracy.
    After many trials, it tells you which combination worked best.

    Optuna is smarter than grid search — it learns from past trials
    and focuses on promising regions of the search space (Bayesian optimization).

Usage:
    python 4_tune_mlp.py              # Run 50 trials (default)
    python 4_tune_mlp.py --trials 100 # Run 100 trials
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
from pathlib import Path
from datetime import datetime
import json
import argparse

# ============================================================
# PATHS
# ============================================================
DATA_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")
MODEL_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/models")
RESULTS_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/results")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training epochs per trial — kept lower to speed up search
# The final model will be retrained with more epochs
TUNING_EPOCHS = 30
TUNING_PATIENCE = 8


# ============================================================
# MODEL - same MLP structure but with variable architecture
# ============================================================
class TunableMLP(nn.Module):
    """MLP where the number of layers and neurons are configurable."""

    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD AND PREPARE DATA (done once, shared across all trials)
# ============================================================
def load_data():
    """Load and split data. Called once at the start. Pre-loads tensors to GPU."""
    print("Loading data...")
    X = np.load(DATA_PATH / "X_snapshots.npy")
    y = np.load(DATA_PATH / "y_snapshots.npy")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split: 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Pre-load ALL tensors to GPU once — avoids CPU→GPU transfer every batch
    print(f"Pre-loading tensors to {DEVICE}...")
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)
    print("Tensors loaded to GPU!")

    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t, scaler, X.shape[1]


# ============================================================
# OPTUNA OBJECTIVE - this is what Optuna optimizes
# ============================================================
def create_objective(X_train, X_val, y_train, y_val, input_dim):
    """
    Returns an objective function for Optuna.

    For each trial, Optuna:
        1. Picks hyperparameters from the ranges we define
        2. Builds and trains a model with those hyperparameters
        3. Returns the validation accuracy
        4. Optuna remembers this and picks smarter combinations next time
    """

    # Pre-compute shuffled indices on GPU for fast batching
    n_train = X_train.shape[0]

    def objective(trial):
        # --- OPTUNA PICKS HYPERPARAMETERS ---
        n_layers = trial.suggest_int('n_layers', 2, 5)

        hidden_dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f'hidden_dim_{i}', 64, 1024, step=64)
            hidden_dims.append(dim)
        hidden_dims.sort(reverse=True)

        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])

        # --- BUILD MODEL ---
        model = TunableMLP(input_dim, hidden_dims, dropout).to(DEVICE)

        # --- SET UP TRAINING ---
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        loss_fn = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        # --- TRAIN with manual batching (no DataLoader overhead) ---
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(TUNING_EPOCHS):
            model.train()
            # Shuffle indices on GPU
            perm = torch.randperm(n_train, device=DEVICE)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                batch_X = X_train[idx]
                batch_y = y_train[idx]
                optimizer.zero_grad()
                output = model(batch_X).squeeze()
                loss = loss_fn(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validate — single forward pass
            model.eval()
            with torch.no_grad():
                val_probs = model(X_val).squeeze()
                val_acc = ((val_probs > 0.5).float() == y_val).float().mean().item()
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TUNING_PATIENCE:
                break

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_acc

    return objective


# ============================================================
# RETRAIN BEST MODEL - train the winning config with full epochs
# ============================================================
def retrain_best(best_params, X_train, X_val, X_test, y_train, y_val, y_test, input_dim, scaler):
    """
    Take the best hyperparameters Optuna found and train properly
    with more epochs and full evaluation.
    """
    print(f"\n{'='*60}")
    print("RETRAINING BEST MODEL WITH FULL EPOCHS")
    print(f"{'='*60}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    # Reconstruct hidden dims from params (sorted descending, same as during tuning)
    n_layers = best_params['n_layers']
    hidden_dims = sorted([best_params[f'hidden_dim_{i}'] for i in range(n_layers)], reverse=True)

    model = TunableMLP(
        input_dim, hidden_dims, best_params['dropout']
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture: {hidden_dims}")
    print(f"Parameters: {num_params:,}")

    # Setup
    lr = best_params['learning_rate']
    wd = best_params['weight_decay']
    if best_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    loss_fn = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    batch_size = best_params['batch_size']
    n_train = X_train.shape[0]

    # Train with more epochs
    best_val_acc = 0
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(200):
        # Train with manual batching
        model.train()
        correct = 0
        total = 0
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = loss_fn(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            preds = (output.detach() > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total

        # Validate — single forward pass on GPU
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val).squeeze()
            val_acc = ((val_probs > 0.5).float() == y_val).float().mean().item()
        scheduler.step(val_acc)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH / "mlp_tuned.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

        if patience_counter >= 25:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Test evaluation — single forward pass on GPU
    model.load_state_dict(torch.load(MODEL_PATH / "mlp_tuned.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test).squeeze()
        test_preds = (test_probs > 0.5).float()

    test_probs_cpu = test_probs.cpu().numpy()
    test_labels_cpu = y_test.cpu().numpy()
    test_preds_cpu = test_preds.cpu().numpy()

    test_acc = accuracy_score(test_labels_cpu, test_preds_cpu)
    test_auc = roc_auc_score(test_labels_cpu, test_probs_cpu)

    print(f"\n{'='*60}")
    print(f"TUNED MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test AUC-ROC:  {test_auc:.4f}")

    # Save scaler for the tuned model
    scaler_params = {'mean': scaler.mean_.tolist(), 'std': scaler.scale_.tolist()}
    with open(MODEL_PATH / "scaler_params_tuned.json", 'w') as f:
        json.dump(scaler_params, f)

    return test_acc, test_auc, history


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()

    print("=" * 60)
    print("CS2 Round Prediction - Hyperparameter Tuning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Trials: {args.trials}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data once
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, input_dim = load_data()

    # Create Optuna study
    # "maximize" because we want the highest validation accuracy
    # MedianPruner stops bad trials early if they're below the median at that epoch
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Run the search
    objective = create_objective(X_train, X_val, y_train, y_val, input_dim)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Print results
    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val accuracy: {study.best_value:.4f} ({study.best_value*100:.2f}%)")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Show top 5 trials
    print(f"\nTop 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, trial in enumerate(trials_sorted[:5]):
        if trial.value:
            print(f"  #{trial.number}: {trial.value:.4f} ({trial.value*100:.2f}%)")

    # Retrain the best model with full epochs
    test_acc, test_auc, history = retrain_best(
        study.best_params, X_train, X_val, X_test,
        y_train, y_val, y_test, input_dim, scaler
    )

    # Save everything
    results = {
        'best_params': study.best_params,
        'best_val_accuracy': float(study.best_value),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'n_trials': args.trials,
        'architecture': sorted([study.best_params[f'hidden_dim_{i}']
                         for i in range(study.best_params['n_layers'])], reverse=True),
        'all_trials': [
            {
                'number': t.number,
                'value': float(t.value) if t.value else None,
                'params': t.params
            }
            for t in study.trials
        ]
    }

    with open(RESULTS_PATH / "tuning_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Compare with previous model
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  Original MLP (untuned):  93.21%")
    print(f"  XGBoost:                 89.39%")
    print(f"  Tuned MLP:               {test_acc*100:.2f}%")
    print(f"  Improvement:             {(test_acc - 0.9321)*100:+.2f}%")
    print(f"{'='*60}")

    print(f"\nResults saved to {RESULTS_PATH / 'tuning_results.json'}")
    print(f"Tuned model saved to {MODEL_PATH / 'mlp_tuned.pth'}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
