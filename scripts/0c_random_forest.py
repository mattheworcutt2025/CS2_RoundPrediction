"""
CS2 Round Prediction - Step 0c: Random Forest Classification

Random Forest builds many decision trees and averages their predictions.
Each tree sees a random subset of features and data — reduces overfitting.

Random Forest:
    - Ensemble of decision trees (bagging)
    - Handles non-linear relationships well
    - Built-in feature importance
    - Robust, rarely overfits badly

Usage:
    python scripts/0c_random_forest.py
"""
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    print("CS2 Round Prediction - Random Forest")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    X = np.load(DATA_PATH / "X_snapshots.npy")
    y = np.load(DATA_PATH / "y_snapshots.npy")

    with open(DATA_PATH / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Normalize (not strictly needed for RF, but keeps things consistent)
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

    # Train Random Forest
    print("\nTraining Random Forest (500 trees)...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["T Wins", "CT Wins"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature importance (top 20)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = {}
    for i in sorted_idx[:20]:
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        top_features[fname] = float(importances[i])

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test AUC-ROC:  {test_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["T Wins", "CT Wins"]))
    print(f"Confusion Matrix:")
    print(np.array(cm))
    print(f"\nTop 10 Features:")
    for i, (fname, imp) in enumerate(list(top_features.items())[:10]):
        print(f"  {i+1}. {fname}: {imp:.4f} ({imp*100:.2f}%)")

    # Validation accuracy
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"\nVal Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")

    # Save results
    results = {
        "model": "Random Forest",
        "script": "scripts/0c_random_forest.py",
        "type": "Ensemble (bagging)",
        "features": int(X.shape[1]),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "val_accuracy": float(val_acc),
        "classification_report": report,
        "confusion_matrix": cm,
        "hyperparameters": {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        "feature_importance": top_features,
    }

    with open(RESULTS_PATH / "random_forest_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH / 'random_forest_results.json'}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
