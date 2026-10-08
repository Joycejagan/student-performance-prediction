"""
train_model.py
--------------
Train, evaluate, compare, and save ML models for Student Performance Prediction.

Two classifiers are trained and compared:
  1. Decision Tree Classifier
  2. Random Forest Classifier

The better-performing model (by weighted F1-score on the test set) is saved to
models/best_model.pkl along with the label encoders.

Usage:
    python src/train_model.py
"""

import os
import sys
import warnings

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Project-local imports
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_data, preprocess, get_feature_names

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "reports")
MODEL_PATH   = os.path.join(MODELS_DIR, "best_model.pkl")


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    model_name: str,
) -> dict:
    """
    Evaluate a trained classifier and print a full report.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    y_pred = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'-' * 55}")
    print(f"  {model_name}")
    print(f"{'-' * 55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f} %)")
    print(f"  Precision : {prec:.4f}  (weighted avg)")
    print(f"  Recall    : {rec:.4f}  (weighted avg)")
    print(f"  F1-score  : {f1:.4f}  (weighted avg)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    model_name: str,
    filename: str,
) -> None:
    """Save a styled confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    fig.tight_layout()

    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


def plot_metrics_comparison(
    metrics: dict,      # {model_name: {metric_name: value}}
) -> None:
    """Bar chart comparing Accuracy / Precision / Recall / F1 across models."""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    labels      = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(metrics.keys())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3498db", "#e67e22"]

    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [metrics[name][k] for k in metric_keys]
        bars = ax.bar(x + i * width - width / 2, values, width, label=name, color=color, alpha=0.85, edgecolor="white")

        # annotate
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9
            )

    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Evaluation Metrics", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = os.path.join(REPORTS_DIR, "06_model_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Model comparison chart saved -> {path}")


def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str,
) -> None:
    """Horizontal bar chart of feature importances (trees only)."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i].replace("_", " ").title() for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_names[::-1], importances[indices][::-1], color="#3b82d4", edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = os.path.join(REPORTS_DIR, "07_feature_importances.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Feature importance chart saved -> {path}")


# ── main training routine ─────────────────────────────────────────────────────

def train_and_save() -> None:
    """
    Full pipeline:
      1. Load and preprocess data.
      2. Train Decision Tree and Random Forest.
      3. Evaluate both models and print metrics.
      4. Plot confusion matrices and comparison chart.
      5. Select the better model by weighted F1-score.
      6. Save the best model + label encoders to models/best_model.pkl.
    """
    _ensure_dirs()

    # ── 1. data preparation ───────────────────────────────────────────────
    print("\n" + "=" * 63)
    print("  Student Performance Prediction -- Model Training")
    print("=" * 63)

    print("\n[1/5] Loading and preprocessing data ...")
    df = load_data()
    X_train, X_test, y_train, y_test, label_encoders, feature_names = preprocess(df)

    # Retrieve original class labels in encoder order for display
    class_names = list(label_encoders["performance"].classes_)
    print(f"      Train size : {X_train.shape[0]} samples")
    print(f"      Test  size : {X_test.shape[0]} samples")
    print(f"      Features   : {len(feature_names)}")
    print(f"      Classes    : {class_names}")

    # ── 2. define models ──────────────────────────────────────────────────
    print("\n[2/5] Defining classifiers ...")

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,           # limit depth to reduce over-fitting
            min_samples_split=10,  # require at least 10 samples to split
            min_samples_leaf=5,    # leaf must contain at least 5 samples
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,      # 200 trees
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,             # use all CPU cores
        ),
    }

    # ── 3. train and evaluate ─────────────────────────────────────────────
    print("\n[3/5] Training and evaluating models ...")
    all_metrics: dict[str, dict] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        all_metrics[name] = evaluate_model(model, X_test, y_test, class_names, name)

    # ── 4. plots ──────────────────────────────────────────────────────────
    print("\n[4/5] Generating evaluation plots ...")
    filenames = {
        "Decision Tree": "06a_confusion_matrix_dt.png",
        "Random Forest": "06b_confusion_matrix_rf.png",
    }
    for name, model in models.items():
        plot_confusion_matrix(model, X_test, y_test, class_names, name, filenames[name])

    plot_metrics_comparison(all_metrics)

    # ── 5. select best model ──────────────────────────────────────────────
    print("\n[5/5] Selecting the best model ...")
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1"])
    best_model = models[best_name]
    best_f1    = all_metrics[best_name]["f1"]

    print(f"\n  Best model : {best_name}")
    print(f"     Weighted F1 : {best_f1:.4f}")

    # Feature importance chart for the best model
    plot_feature_importance(best_model, feature_names, best_name)

    # ── 6. save ───────────────────────────────────────────────────────────
    payload = {
        "model":          best_model,
        "label_encoders": label_encoders,
        "feature_names":  feature_names,
        "model_name":     best_name,
        "class_names":    class_names,
        "metrics":        all_metrics[best_name],
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\n  Model saved -> {MODEL_PATH}")
    print("\n  Training complete!")
    print("=" * 63 + "\n")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_save()
