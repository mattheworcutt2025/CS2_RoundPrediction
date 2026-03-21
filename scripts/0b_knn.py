"""
CS2 Round Prediction - Step 0b: K-Nearest Neighbors Classification

KNN predicts by looking at the K closest training examples and voting.
For CS2: "find similar game states and see who won those."

K-Nearest Neighbors:
    - Non-parametric: no assumptions about data distribution
    - Intuitive: similar game states should have similar outcomes
    - Weakness: slow at prediction time, sensitive to feature scaling and K

Usage:
    python scripts/0b_knn.py
"""
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix)

# ============================================================
# PATHS
# ============================================================
DATA_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")
RESULTS_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("CS2 Round Prediction - K-Nearest Neighbors")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    X = np.load(DATA_PATH / "X_snapshots.npy")
    y = np.load(DATA_PATH / "y_snapshots.npy")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Normalize (critical for KNN — distance-based)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split: same as all other models
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Try different K values to find the best
    print("\nSearching for best K...")
    k_values = [3, 5, 7, 9, 11, 15, 21, 31, 51]
    k_results = {}

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, knn.predict(X_val))
        k_results[k] = val_acc
        print(f"  K={k:3d}: Val Accuracy = {val_acc:.4f} ({val_acc*100:.2f}%)")

    best_k = max(k_results, key=k_results.get)
    print(f"\nBest K: {best_k} (Val Accuracy: {k_results[best_k]:.4f})")

    # Train final model with best K
    print(f"\nTraining final KNN (K={best_k})...")
    model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["T Wins", "CT Wins"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test AUC-ROC:  {test_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["T Wins", "CT Wins"]))
    print(f"Confusion Matrix:")
    print(np.array(cm))

    # Save results
    results = {
        "model": "K-Nearest Neighbors",
        "script": "scripts/0b_knn.py",
        "type": "Instance-based classifier",
        "features": int(X.shape[1]),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "classification_report": report,
        "confusion_matrix": cm,
        "hyperparameters": {
            "best_k": best_k,
            "algorithm": "auto",
        },
        "k_search_results": {str(k): float(v) for k, v in k_results.items()},
    }

    with open(RESULTS_PATH / "knn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH / 'knn_results.json'}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
