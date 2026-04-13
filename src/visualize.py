"""
Centralised plotting functions for the project.
Every plot is returned as a matplotlib Figure so callers can save via utils.save_figure.
"""

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


# ─── Style defaults ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


def plot_target_distribution(y: pd.Series, title: str = "Target Distribution") -> plt.Figure:
    """Histogram of raw rating values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = y.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title(title)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution") -> plt.Figure:
    """Bar chart of class label counts."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = y.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="coral", edgecolor="white")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> plt.Figure:
    """Full correlation heatmap of numeric columns."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importance",
    top_n: int = 20,
) -> plt.Figure:
    """Horizontal bar chart of feature importances (top_n)."""
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, len(idx) * 0.35)))
    ax.barh(
        [feature_names[i] for i in reversed(idx)],
        importances[list(reversed(idx))],
        color="teal", edgecolor="white",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
) -> plt.Figure:
    """Scatter plot of predicted vs actual values with diagonal reference."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
    mn, mx = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
) -> plt.Figure:
    """Residual plot: predicted on x-axis, residuals on y-axis."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
) -> plt.Figure:
    """ROC curve for binary classification."""
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_permutation_importance(
    perm_result,
    feature_names: List[str],
    title: str = "Permutation Importance",
    top_n: int = 20,
) -> plt.Figure:
    """Box-plot style permutation importance."""
    means = perm_result.importances_mean
    idx = np.argsort(means)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, len(idx) * 0.35)))
    ax.boxplot(
        perm_result.importances[list(reversed(idx))].T,
        vert=False,
        labels=[feature_names[i] for i in reversed(idx)],
    )
    ax.set_xlabel("Decrease in score")
    ax.set_title(title)
    fig.tight_layout()
    return fig
