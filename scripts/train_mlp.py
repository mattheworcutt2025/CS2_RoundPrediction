"""
CS2 Round Prediction - MLP (Multi-Layer Perceptron) Deep Learning Model

What this script does:
    1. Loads the snapshot data (86 features per game-state snapshot)
    2. Normalizes the features (scales them so the model can learn better)
    3. Splits data into train/validation/test sets
    4. Trains a neural network to predict CT win (1) or T win (0)
    5. Saves the best model, results, and training plots

What is an MLP?
    A Multi-Layer Perceptron is the simplest type of neural network.
    It's just layers of neurons stacked on top of each other:

    Input (86 features) → Layer 1 (512 neurons) → Layer 2 (256) → Layer 3 (128) → Layer 4 (64) → Output (1 number: probability of CT win)

    Each layer does: output = activation(weights * input + bias)
    The model learns the weights and biases by seeing thousands of examples.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# ============================================================
# FILE PATHS
# ============================================================
DATA_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")
MODEL_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/models")
LOG_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/logs")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# HYPERPARAMETERS - these control how the model trains
# ============================================================
# NOTE: These were NOT extensively tuned. Things you could try:
#   - BATCH_SIZE: 64, 128, 256, 512 (smaller = more updates per epoch, noisier)
#   - LEARNING_RATE: 0.0001, 0.0005, 0.001, 0.005 (how big each weight update is)
#   - DROPOUT: 0.1, 0.2, 0.3, 0.5 (fraction of neurons randomly turned off to prevent overfitting)
#   - HIDDEN_DIMS: [256, 128], [512, 256, 128], [1024, 512, 256, 128] (size/depth of network)
#   - WEIGHT_DECAY: 0, 1e-5, 1e-4, 1e-3 (penalizes large weights to prevent overfitting)

BATCH_SIZE = 256        # How many samples the model sees before updating weights
LEARNING_RATE = 0.001   # How much to adjust weights each step (too high = unstable, too low = slow)
EPOCHS = 200            # Maximum number of times to go through the entire dataset
EARLY_STOP_PATIENCE = 25  # Stop if validation accuracy doesn't improve for 25 epochs
DROPOUT = 0.3           # Randomly turn off 30% of neurons during training (prevents overfitting)
HIDDEN_DIMS = [512, 256, 128, 64]  # Number of neurons in each hidden layer

# Use GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# MODEL DEFINITION
# ============================================================
class CS2RoundMLP(nn.Module):
    """
    A simple feedforward neural network (MLP).

    Architecture (with default settings):
        Input (86 features)
            ↓
        Linear(86 → 512) → BatchNorm → ReLU → Dropout(0.3)
            ↓
        Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.3)
            ↓
        Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.3)
            ↓
        Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)
            ↓
        Linear(64 → 1) → Sigmoid
            ↓
        Output: probability between 0 and 1 (0 = T wins, 1 = CT wins)

    What each component does:
        - Linear:    Multiplies input by learned weights + bias (the actual "learning" part)
        - BatchNorm: Normalizes values between layers (makes training faster and more stable)
        - ReLU:      Activation function — turns negative values to 0, keeps positive as-is
                     Without this, stacking linear layers would just be one big linear layer
        - Dropout:   Randomly zeroes out some neurons during training
                     Forces the network to not rely on any single neuron (prevents overfitting)
        - Sigmoid:   Squishes final output to be between 0 and 1 (so we can read it as a probability)
    """

    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()

        # Build layers dynamically based on hidden_dims list
        layers = []
        prev_dim = input_dim  # First layer takes the raw features (86)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))   # Weighted connections
            layers.append(nn.BatchNorm1d(hidden_dim))         # Normalize
            layers.append(nn.ReLU())                          # Activation
            layers.append(nn.Dropout(dropout))                # Regularization
            prev_dim = hidden_dim

        # Final layer: compress to single output + sigmoid for probability
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        # nn.Sequential chains all layers together so data flows through them in order
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Pass input through all layers and return prediction."""
        return self.network(x)


# ============================================================
# TRAINING FUNCTION - one pass through the entire training data
# ============================================================
def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """
    Train the model for one epoch (one full pass through training data).

    How training works:
        1. Feed a batch of snapshots into the model → get predictions
        2. Compare predictions to actual labels using the loss function
        3. Compute gradients (how much each weight contributed to the error)
        4. Update weights in the direction that reduces the error
        5. Repeat for all batches
    """
    model.train()  # Enable training mode (activates dropout and batch norm)
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_true_labels = []

    for batch_features, batch_labels in dataloader:
        # Move data to GPU/CPU
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # --- FORWARD PASS ---
        # Feed features through the network to get predictions
        predicted_probs = model(batch_features).squeeze()  # squeeze removes extra dimension

        # --- COMPUTE LOSS ---
        # Binary Cross-Entropy: measures how wrong our probabilities are
        # If true label is 1 (CT wins) and we predicted 0.9, loss is low
        # If true label is 1 (CT wins) and we predicted 0.1, loss is high
        loss = loss_function(predicted_probs, batch_labels.float())

        # --- BACKWARD PASS ---
        optimizer.zero_grad()  # Reset gradients from previous batch
        loss.backward()        # Compute gradients (how to adjust each weight)

        # Clip gradients to prevent them from exploding (becoming too large)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- UPDATE WEIGHTS ---
        optimizer.step()  # Adjust weights using the computed gradients

        # Track metrics
        total_loss += loss.item()
        probs = predicted_probs.detach().cpu().numpy()
        all_probabilities.extend(probs)
        all_predictions.extend((probs > 0.5).astype(int))  # Convert probability to 0 or 1
        all_true_labels.extend(batch_labels.cpu().numpy())

    # Calculate epoch-level metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_true_labels, all_predictions)
    try:
        auc = roc_auc_score(all_true_labels, all_probabilities)
    except:
        auc = 0.5

    return avg_loss, accuracy, auc


# ============================================================
# EVALUATION FUNCTION - test model without updating weights
# ============================================================
def evaluate_model(model, dataloader, loss_function, device):
    """
    Evaluate the model on validation or test data.
    Same as training but WITHOUT updating weights (no backward pass).
    """
    model.eval()  # Disable dropout and use running batch norm stats
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_true_labels = []

    # torch.no_grad() tells PyTorch not to track gradients (saves memory, faster)
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass only (no backward pass, no weight updates)
            predicted_probs = model(batch_features).squeeze()
            loss = loss_function(predicted_probs, batch_labels.float())

            total_loss += loss.item()
            probs = predicted_probs.cpu().numpy()
            all_probabilities.extend(probs)
            all_predictions.extend((probs > 0.5).astype(int))
            all_true_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_true_labels, all_predictions)
    try:
        auc = roc_auc_score(all_true_labels, all_probabilities)
    except:
        auc = 0.5

    return avg_loss, accuracy, auc, all_predictions, all_true_labels, all_probabilities


# ============================================================
# MAIN - ties everything together
# ============================================================
def main():
    print("=" * 60)
    print("CS2 Round Prediction - MLP Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ----------------------------------------------------------
    # STEP 1: Load the snapshot data
    # ----------------------------------------------------------
    # X_snapshots.npy: each row is a game-state snapshot (86 features)
    # y_snapshots.npy: each value is the label (0 = T won, 1 = CT won)
    print("\nLoading snapshot data...")
    X = np.load(DATA_PATH / "X_snapshots.npy")
    y = np.load(DATA_PATH / "y_snapshots.npy")

    with open(DATA_PATH / "feature_names.json", 'r') as f:
        feature_names = json.load(f)

    print(f"Loaded {len(y)} snapshots with {X.shape[1]} features")
    print(f"CT wins: {np.sum(y)} ({100*np.sum(y)/len(y):.1f}%)")
    print(f"T wins: {len(y) - np.sum(y)} ({100*(len(y)-np.sum(y))/len(y):.1f}%)")

    # ----------------------------------------------------------
    # STEP 2: Normalize features
    # ----------------------------------------------------------
    # StandardScaler transforms each feature to have mean=0 and std=1
    # Why? Features have very different scales:
    #   - ct_alive: 0-5
    #   - ct_money_total: 0-80000
    # Without scaling, the model would pay too much attention to large-valued features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler parameters so we can normalize new data the same way during inference
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    }
    with open(MODEL_PATH / "scaler_params.json", 'w') as f:
        json.dump(scaler_params, f)

    # ----------------------------------------------------------
    # STEP 3: Split data into train / validation / test
    # ----------------------------------------------------------
    # 80% train: model learns from this
    # 10% validation: used to check progress during training (for early stopping)
    # 10% test: final evaluation — model never sees this during training
    # stratify=y ensures each split has the same CT/T win ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # ----------------------------------------------------------
    # STEP 4: Create PyTorch DataLoaders
    # ----------------------------------------------------------
    # DataLoaders handle batching and shuffling automatically
    # Instead of feeding all 129K training samples at once, we feed them
    # in batches of 256 — this is faster and helps the model generalize
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True  # Shuffle training data each epoch
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=BATCH_SIZE
    )

    # ----------------------------------------------------------
    # STEP 5: Create the model
    # ----------------------------------------------------------
    model = CS2RoundMLP(
        input_dim=X.shape[1],     # 86 input features
        hidden_dims=HIDDEN_DIMS,  # [512, 256, 128, 64] neurons per layer
        dropout=DROPOUT           # 30% dropout
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")

    # ----------------------------------------------------------
    # STEP 6: Set up training components
    # ----------------------------------------------------------
    # Loss function: Binary Cross-Entropy
    # Measures how far our predicted probability is from the true label
    loss_function = nn.BCELoss()

    # Optimizer: AdamW
    # Adjusts weights based on gradients. AdamW adds weight decay (L2 regularization)
    # which penalizes large weights to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler: reduces learning rate when validation accuracy plateaus
    # After 10 epochs with no improvement, multiply learning rate by 0.5
    # This helps the model fine-tune when it stops making big improvements
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5
    )

    # ----------------------------------------------------------
    # STEP 7: Training loop
    # ----------------------------------------------------------
    # Each epoch: train on all batches → check validation accuracy → save if best
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    best_val_acc = 0
    patience_counter = 0

    print("\nTraining started...")
    for epoch in range(EPOCHS):
        # Train for one epoch
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, loss_function, optimizer, DEVICE
        )

        # Evaluate on validation set (without updating weights)
        val_loss, val_acc, val_auc, _, _, _ = evaluate_model(
            model, val_loader, loss_function, DEVICE
        )

        # Tell the scheduler about validation accuracy (it may reduce learning rate)
        scheduler.step(val_acc)

        # Record history for plotting later
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        # Save the model if this is the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH / "MLP_Standard_best.pth")
            patience_counter = 0  # Reset patience since we improved
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train: {train_acc:.4f} (AUC {train_auc:.4f}) | "
                  f"Val: {val_acc:.4f} (AUC {val_auc:.4f}) | "
                  f"Best: {best_val_acc:.4f}")

        # Early stopping: if no improvement for 25 epochs, stop training
        # This prevents overfitting — the model starts memorizing training data
        # instead of learning general patterns
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # ----------------------------------------------------------
    # STEP 8: Final evaluation on test set
    # ----------------------------------------------------------
    # Load the best model (not the last one — the last one may be overfit)
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(MODEL_PATH / "MLP_Standard_best.pth", weights_only=True))
    test_loss, test_acc, test_auc, test_preds, test_labels, test_probs = evaluate_model(
        model, test_loader, loss_function, DEVICE
    )

    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test AUC-ROC:  {test_auc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['T Wins', 'CT Wins']))

    # ----------------------------------------------------------
    # STEP 9: Save results
    # ----------------------------------------------------------
    results = {
        'model': 'MLP_Standard',
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'best_val_accuracy': float(best_val_acc),
        'epochs_trained': len(history['train_loss']),
        'hyperparameters': {
            'hidden_dims': HIDDEN_DIMS,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'weight_decay': 1e-4,
            'early_stop_patience': EARLY_STOP_PATIENCE
        },
        'num_features': X.shape[1],
        'num_samples': len(y),
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
    }

    with open(LOG_PATH / "mlp_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # ----------------------------------------------------------
    # STEP 10: Plot training curves
    # ----------------------------------------------------------
    # These plots help diagnose training:
    # - If train acc >> val acc: overfitting (model memorizes training data)
    # - If both are low: underfitting (model is too simple or needs more data)
    # - If both are high and close: good fit
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Time (lower = better)')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].axhline(y=0.88, color='r', linestyle='--', label='Target (88%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Over Time (higher = better)')
    axes[1].legend()
    axes[1].grid(True)

    # AUC curve
    axes[2].plot(history['train_auc'], label='Train AUC')
    axes[2].plot(history['val_auc'], label='Val AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC-ROC')
    axes[2].set_title('AUC Over Time (higher = better, 1.0 = perfect)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(LOG_PATH / "mlp_training_curves.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {LOG_PATH / 'mlp_results.json'}")
    print(f"Model saved to {MODEL_PATH / 'MLP_Standard_best.pth'}")
    print(f"Training curves saved to {LOG_PATH / 'mlp_training_curves.png'}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
