"""
Generate comprehensive results: evaluations, plots, model comparison.
Run after all models have been trained.
"""
import numpy as np
import torch
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")
MODEL_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/models")
RESULTS_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/results")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# MLP Classes (must match training scripts)
# ============================================================
class CS2RoundMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hd
        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TunableMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hd
        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def main():
    # Load and split data (same split as training)
    print("Loading data...")
    X = np.load(DATA_PATH / "X_snapshots.npy")
    y = np.load(DATA_PATH / "y_snapshots.npy")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    input_dim = X.shape[1]

    # ---- Evaluate MLP Standard ----
    print("Evaluating MLP Standard...")
    mlp_std = CS2RoundMLP(input_dim).to(DEVICE)
    mlp_std.load_state_dict(torch.load(MODEL_PATH / "mlp_standard.pth", weights_only=True, map_location=DEVICE))
    mlp_std.eval()
    with torch.no_grad():
        probs_std = mlp_std(X_test_t).squeeze().cpu().numpy()
    preds_std = (probs_std > 0.5).astype(int)
    std_acc = accuracy_score(y_test, preds_std)
    std_auc = roc_auc_score(y_test, probs_std)
    std_report = classification_report(y_test, preds_std, target_names=["T Wins", "CT Wins"], output_dict=True)
    std_cm = confusion_matrix(y_test, preds_std).tolist()

    mlp_std_results = {
        "model": "MLP_Standard",
        "script": "scripts/train_mlp.py",
        "model_file": "models/mlp_standard.pth",
        "architecture": [512, 256, 128, 64],
        "dropout": 0.3,
        "parameters": sum(p.numel() for p in mlp_std.parameters()),
        "test_accuracy": float(std_acc),
        "test_auc": float(std_auc),
        "classification_report": std_report,
        "confusion_matrix": std_cm,
    }
    with open(RESULTS_PATH / "mlp_standard_results.json", "w") as f:
        json.dump(mlp_std_results, f, indent=2)
    print(f"  MLP Standard: acc={std_acc:.4f}, auc={std_auc:.4f}")

    # ---- Evaluate Tuned MLP ----
    print("Evaluating MLP Tuned...")
    tuned_arch = [896, 832, 576, 512, 192]
    mlp_tuned = TunableMLP(input_dim, tuned_arch, 0.1).to(DEVICE)
    mlp_tuned.load_state_dict(torch.load(MODEL_PATH / "mlp_tuned.pth", weights_only=True, map_location=DEVICE))
    mlp_tuned.eval()
    with torch.no_grad():
        probs_tuned = mlp_tuned(X_test_t).squeeze().cpu().numpy()
    preds_tuned = (probs_tuned > 0.5).astype(int)
    tuned_acc = accuracy_score(y_test, preds_tuned)
    tuned_auc = roc_auc_score(y_test, probs_tuned)
    tuned_report = classification_report(y_test, preds_tuned, target_names=["T Wins", "CT Wins"], output_dict=True)
    tuned_cm = confusion_matrix(y_test, preds_tuned).tolist()

    # Update tuning_results.json
    with open(RESULTS_PATH / "tuning_results.json") as f:
        tuning_data = json.load(f)
    tuning_data["classification_report"] = tuned_report
    tuning_data["confusion_matrix"] = tuned_cm
    tuning_data["parameters"] = sum(p.numel() for p in mlp_tuned.parameters())
    tuning_data["script"] = "scripts/tune_mlp.py"
    tuning_data["model_file"] = "models/mlp_tuned.pth"
    with open(RESULTS_PATH / "tuning_results.json", "w") as f:
        json.dump(tuning_data, f, indent=2)
    print(f"  MLP Tuned: acc={tuned_acc:.4f}, auc={tuned_auc:.4f}")

    # ---- Model comparison summary ----
    print("Generating model comparison...")
    # Load baseline results
    baseline_results = {}
    for name, fname in [("logistic_regression", "logistic_regression_results.json"),
                        ("knn", "knn_results.json"),
                        ("random_forest", "random_forest_results.json")]:
        fpath = RESULTS_PATH / fname
        if fpath.exists():
            with open(fpath) as f:
                baseline_results[name] = json.load(f)

    comparison = {
        "models": [
            {
                "name": "Logistic Regression",
                "script": "scripts/0a_logistic_regression.py",
                "type": "Linear classifier",
                "features": 86,
                "data": "mid-round snapshots (161,271 snapshots)",
                "test_accuracy": baseline_results.get("logistic_regression", {}).get("test_accuracy", None),
                "test_auc": baseline_results.get("logistic_regression", {}).get("test_auc", None),
                "notes": "Step 0a - simplest baseline, linear decision boundary",
            },
            {
                "name": "K-Nearest Neighbors",
                "script": "scripts/0b_knn.py",
                "type": "Instance-based classifier",
                "features": 86,
                "data": "mid-round snapshots (161,271 snapshots)",
                "test_accuracy": baseline_results.get("knn", {}).get("test_accuracy", None),
                "test_auc": baseline_results.get("knn", {}).get("test_auc", None),
                "best_k": baseline_results.get("knn", {}).get("hyperparameters", {}).get("best_k", None),
                "notes": "Step 0b - finds similar game states and votes on winner",
            },
            {
                "name": "Random Forest",
                "script": "scripts/0c_random_forest.py",
                "type": "Ensemble (bagging)",
                "features": 86,
                "data": "mid-round snapshots (161,271 snapshots)",
                "test_accuracy": baseline_results.get("random_forest", {}).get("test_accuracy", None),
                "test_auc": baseline_results.get("random_forest", {}).get("test_auc", None),
                "notes": "Step 0c - ensemble of 500 decision trees",
            },
            {
                "name": "LSTM (round-start)",
                "script": "scripts/2_train_lstm.py",
                "model_file": "models/lstm_baseline.pth",
                "type": "Bidirectional LSTM",
                "features": 15,
                "data": "round-start only (9,746 rounds)",
                "test_accuracy": 0.5321,
                "test_auc": None,
                "parameters": 199553,
                "epochs_trained": 46,
                "notes": "Step 2 - first DL attempt, proved round-start features are insufficient",
            },
            {
                "name": "XGBoost",
                "script": "scripts/3_train_xgboost.py",
                "model_file": "models/xgboost.json",
                "type": "Gradient Boosted Trees",
                "features": 86,
                "data": "mid-round snapshots (161,271 snapshots)",
                "test_accuracy": 0.8939,
                "test_auc": 0.9635,
                "cv_accuracy": "85.08% +/- 0.20%",
                "notes": "Step 3 - gradient boosting baseline on snapshot features",
            },
            {
                "name": "MLP Standard",
                "script": "scripts/4_train_mlp.py",
                "model_file": "models/mlp_standard.pth",
                "type": "Feedforward MLP",
                "architecture": [512, 256, 128, 64],
                "features": 86,
                "data": "mid-round snapshots (161,271 snapshots)",
                "test_accuracy": float(std_acc),
                "test_auc": float(std_auc),
                "parameters": int(sum(p.numel() for p in mlp_std.parameters())),
                "notes": "Step 4 - deep learning model for course requirement",
            },
            {
                "name": "MLP Tuned (Optuna)",
                "script": "scripts/5_tune_mlp.py",
                "model_file": "models/mlp_tuned.pth",
                "type": "Feedforward MLP (Optuna-optimized)",
                "architecture": tuned_arch,
                "features": 86,
                "data": "mid-round snapshots (161,271 snapshots)",
                "test_accuracy": float(tuned_acc),
                "test_auc": float(tuned_auc),
                "parameters": int(sum(p.numel() for p in mlp_tuned.parameters())),
                "optuna_trials": 50,
                "best_val_accuracy": tuning_data["best_val_accuracy"],
                "notes": "Step 5 - best model, Optuna hyperparameter search",
            },
        ],
        "dataset": {
            "source": "Kaggle CS2 Dataset (December 10, 2023)",
            "total_matches": 530,
            "total_rounds": 9746,
            "total_snapshots": 161271,
            "features": 86,
            "splits": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
            "class_balance": {
                "ct_wins_pct": round(float(np.mean(y == 1)) * 100, 1),
                "t_wins_pct": round(float(np.mean(y == 0)) * 100, 1),
            },
        },
        "key_findings": [
            "Logistic Regression (76%) proves features have signal but relationships are non-linear",
            "KNN (93.8%) shows similar game states reliably predict outcomes",
            "Random Forest (89.9%) and XGBoost (89.4%) are strong tree-based baselines",
            "MLP (93.9%) matches KNN and captures non-linear patterns learned end-to-end",
            "Tuned MLP (95.75%) beats all models after Optuna hyperparameter search",
            "Round-start LSTM (53%) vs mid-round models proves snapshot features are critical",
        ],
    }
    with open(RESULTS_PATH / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # ---- Plot 1: Model evaluation ----
    print("Generating evaluation plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("CS2 Round Prediction - Model Evaluation", fontsize=16, fontweight="bold")

    # ROC curves
    ax = axes[0, 0]
    for name, probs_arr, color in [
        ("MLP Standard", probs_std, "#2196F3"),
        ("MLP Tuned", probs_tuned, "#4CAF50"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs_arr)
        auc_val = roc_auc_score(y_test, probs_arr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Confusion matrix (tuned)
    ax = axes[0, 1]
    cm = np.array(tuned_cm)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["T Wins", "CT Wins"])
    ax.set_yticklabels(["T Wins", "CT Wins"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Tuned MLP)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)

    # Model accuracy comparison
    ax = axes[1, 0]
    models = ["LSTM\n(round-start)", "XGBoost", "MLP\nStandard", "MLP\nTuned"]
    accs = [53.21, 89.39, std_acc * 100, tuned_acc * 100]
    colors = ["#f44336", "#FF9800", "#2196F3", "#4CAF50"]
    bars = ax.bar(models, accs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{acc:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.grid(True, axis="y", alpha=0.3)

    # Prediction confidence
    ax = axes[1, 1]
    correct_tuned = preds_tuned == y_test
    ax.hist(probs_tuned[correct_tuned], bins=50, alpha=0.7, color="#4CAF50", label="Correct", density=True)
    ax.hist(probs_tuned[~correct_tuned], bins=50, alpha=0.7, color="#f44336", label="Incorrect", density=True)
    ax.set_xlabel("Predicted Probability (CT Win)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence (Tuned MLP)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "model_evaluation.png", dpi=150, bbox_inches="tight")
    print("  Saved model_evaluation.png")

    # ---- Plot 2: Tuning analysis ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    trial_nums = [t["number"] for t in tuning_data["all_trials"] if t["value"] is not None]
    trial_vals = [t["value"] * 100 for t in tuning_data["all_trials"] if t["value"] is not None]
    best_so_far = []
    best = 0
    for v in trial_vals:
        best = max(best, v)
        best_so_far.append(best)

    ax = axes2[0]
    ax.scatter(trial_nums, trial_vals, alpha=0.6, color="#2196F3", s=30, label="Trial accuracy")
    ax.plot(trial_nums, best_so_far, color="#4CAF50", lw=2, label="Best so far")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Optuna Tuning Progress (50 Trials)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy by network depth
    ax = axes2[1]
    layers_vals = {}
    for t in tuning_data["all_trials"]:
        if t["value"] is not None:
            nl = t["params"]["n_layers"]
            if nl not in layers_vals:
                layers_vals[nl] = []
            layers_vals[nl].append(t["value"] * 100)

    positions = sorted(layers_vals.keys())
    data = [layers_vals[p] for p in positions]
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2196F3")
        patch.set_alpha(0.7)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy by Network Depth")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "tuning_analysis.png", dpi=150, bbox_inches="tight")
    print("  Saved tuning_analysis.png")

    print("\nDone! All results generated.")


if __name__ == "__main__":
    main()
