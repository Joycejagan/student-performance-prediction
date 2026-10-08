"""
eda.py
------
Exploratory Data Analysis (EDA) for Student Performance Prediction.

This module produces and saves visualisation plots to help understand
the dataset before model training.

Usage (standalone):
    python src/eda.py

All plots are saved as PNG files inside a  reports/  sub-folder that is
created automatically at the project root.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Project-local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (
    load_data,
    inspect_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)

warnings.filterwarnings("ignore")

# ── output directory for saved plots ──────────────────────────────────────────
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def _save(fig: plt.Figure, filename: str) -> str:
    """Save a matplotlib figure to the reports directory and close it."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# ── individual plot helpers ────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> str:
    """
    Bar chart showing how many students fall into each performance category.
    """
    counts = df[TARGET_COL].value_counts().reindex(["Low", "Medium", "High"])

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)

    # annotate each bar with its count
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Student Performance Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance Category", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "01_target_distribution.png")


def plot_numerical_distributions(df: pd.DataFrame) -> str:
    """
    Histogram + KDE for every numerical feature, coloured by performance.
    """
    n_cols = 3
    n_rows = int(np.ceil(len(NUMERICAL_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#2ecc71"}

    for i, col in enumerate(NUMERICAL_FEATURES):
        ax = axes[i]
        for cat, color in palette.items():
            subset = df[df[TARGET_COL] == cat][col].dropna()
            ax.hist(subset, bins=20, alpha=0.55, color=color, label=cat, density=True)
            subset.plot.kde(ax=ax, color=color, linewidth=1.8)

        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # hide unused subplots
    for j in range(len(NUMERICAL_FEATURES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions by Performance", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "02_numerical_distributions.png")


def plot_categorical_distributions(df: pd.DataFrame) -> str:
    """
    Count-plot for every categorical feature, grouped by performance label.
    """
    n_cols = 2
    n_rows = int(np.ceil(len(CATEGORICAL_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 4))
    axes = axes.flatten()

    palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#2ecc71"}
    order = ["Low", "Medium", "High"]

    for i, col in enumerate(CATEGORICAL_FEATURES):
        ax = axes[i]
        sns.countplot(
            data=df,
            x=col,
            hue=TARGET_COL,
            hue_order=order,
            palette=palette,
            ax=ax,
            edgecolor="white",
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(title="Performance", fontsize=8, title_fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(CATEGORICAL_FEATURES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Features vs Student Performance", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "03_categorical_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """
    Pearson correlation heatmap for numerical features (including encoded target).
    """
    # quick label-encode the target for correlation purposes only
    target_encoded = df[TARGET_COL].map({"Low": 0, "Medium": 1, "High": 2})
    num_df = df[NUMERICAL_FEATURES].copy()
    num_df["performance_score"] = target_encoded

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Heatmap (Numerical Features + Performance)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_correlation_heatmap.png")


def plot_boxplots(df: pd.DataFrame) -> str:
    """
    Box-plots for key numerical features grouped by performance category.
    """
    key_features = ["attendance_pct", "study_hours_per_week", "previous_marks", "assignments_completed"]
    order = ["Low", "Medium", "High"]
    palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#2ecc71"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(key_features):
        sns.boxplot(
            data=df,
            x=TARGET_COL,
            y=col,
            order=order,
            palette=palette,
            ax=axes[i],
            linewidth=1.2,
        )
        axes[i].set_title(col.replace("_", " ").title(), fontsize=11)
        axes[i].set_xlabel("Performance", fontsize=9)
        axes[i].set_ylabel(col.replace("_", " ").title(), fontsize=9)
        axes[i].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Key Features vs Performance (Box Plots)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_boxplots.png")


def run_eda(df: pd.DataFrame | None = None) -> None:
    """
    Run all EDA steps:
      1. Print dataset summary to console.
      2. Generate and save all visualisation plots.

    Parameters
    ----------
    df : optional pre-loaded DataFrame.  If None, data is loaded from disk.
    """
    if df is None:
        df = load_data()

    print("\n-- Step 1 / 5 : Dataset Inspection ----------------------------------")
    inspect_data(df)

    print("\n-- Step 2 / 5 : Target Distribution ---------------------------------")
    plot_target_distribution(df)

    print("\n-- Step 3 / 5 : Numerical Distributions -----------------------------")
    plot_numerical_distributions(df)

    print("\n-- Step 4 / 5 : Categorical Distributions ---------------------------")
    plot_categorical_distributions(df)

    print("\n-- Step 5a / 5 : Correlation Heatmap --------------------------------")
    plot_correlation_heatmap(df)

    print("\n-- Step 5b / 5 : Box Plots ------------------------------------------")
    plot_boxplots(df)

    print("\nEDA complete.  All plots saved to the  reports/  folder.")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_eda()
