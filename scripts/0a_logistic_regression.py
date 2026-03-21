"""
CS2 Round Prediction - Step 0a: Logistic Regression Baseline

The simplest classification model. If this works well, our features have signal.
If it doesn't, we need better features before trying complex models.

Logistic Regression:
    - Fits a linear decision boundary between classes
    - Fast to train, highly interpretable
    - Struggles with non-linear relationships

Usage:
    python scripts/0a_logistic_regression.py
"""
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    print("CS2 Round Prediction - Logistic Regression")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    X = np.load(DATA_PATH / "X_snapshots.npy")
    y = np.load(DATA_PATH / "y_snapshots.npy")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split: 80% train, 10% val, 10% test (same split as all other models)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Train logistic regression
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
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
        "model": "Logistic Regression",
        "script": "scripts/0a_logistic_regression.py",
        "type": "Linear classifier",
        "features": int(X.shape[1]),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "classification_report": report,
        "confusion_matrix": cm,
        "hyperparameters": {
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 1000
        },
        "num_coefficients": int(model.coef_.shape[1]),
    }

    with open(RESULTS_PATH / "logistic_regression_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH / 'logistic_regression_results.json'}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
