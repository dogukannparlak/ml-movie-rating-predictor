"""
Classification evaluation: compute metrics, generate best-model plots only.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger, save_figure, save_table, timer
from src.visualize import plot_confusion_matrix, plot_roc_curve


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba=None,
    n_classes: int = 2,
) -> dict:
    """Compute classification metrics for one model."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Log-loss (requires probabilities)
    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except Exception:
            metrics["log_loss"] = np.nan
    else:
        metrics["log_loss"] = np.nan

    # ROC-AUC for binary only
    if n_classes == 2 and y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


@timer
def evaluate_classification_holdout(
    holdout_results: List[dict],
    task_name: str,
) -> pd.DataFrame:
    """
    Compute metrics for every holdout result and save comparison table.
    Does NOT generate plots — call plot_best_classification separately.
    """
    logger = get_logger()
    rows = []

    for res in holdout_results:
        y_true = res["y_test"]
        y_pred = res["y_pred"]
        y_proba = res.get("y_proba")
        model_name = res["model"]
        full_task = res["task"]
        n_classes = len(np.unique(y_true))

        metrics = _compute_metrics(y_true, y_pred, y_proba, n_classes)
        row = {"task": full_task, "model": model_name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows)
    save_table(df, f"classification_{task_name}_results.csv", index=False)
    logger.info(f"Saved classification_{task_name}_results.csv ({len(df)} rows)")
    return df


def plot_best_classification(
    holdout_results: List[dict],
    metrics_df: pd.DataFrame,
    task_name: str,
) -> None:
    """
    Generate confusion matrix (and ROC for binary) for the best baseline
    model only. Best is determined by f1_macro on baseline features.
    """
    logger = get_logger()

    baseline_mask = metrics_df["task"].str.endswith("_baseline")
    if baseline_mask.sum() == 0:
        logger.warning(f"No baseline results found for {task_name}")
        return

    baseline_df = metrics_df[baseline_mask]
    best_model = baseline_df.loc[baseline_df["f1_macro"].idxmax(), "model"]

    for res in holdout_results:
        if res["model"] == best_model and res["task"].endswith("_baseline"):
            y_true = res["y_test"]
            y_pred = res["y_pred"]
            y_proba = res.get("y_proba")
            n_classes = len(np.unique(y_true))
            labels = [str(l) for l in sorted(np.unique(np.concatenate([y_true, y_pred])))]

            # Confusion matrix
            fig = plot_confusion_matrix(
                y_true, y_pred, labels=labels,
                title=f"Confusion Matrix — {best_model} ({task_name})",
            )
            save_figure(fig, f"cm_{task_name}_{best_model}.png")
            logger.info(f"Saved CM for best model: {best_model} ({task_name})")

            # ROC for binary only
            if n_classes == 2 and y_proba is not None:
                try:
                    fig_roc = plot_roc_curve(
                        y_true, y_proba[:, 1],
                        title=f"ROC Curve — {best_model} ({task_name})",
                    )
                    save_figure(fig_roc, f"roc_{task_name}_{best_model}.png")
                    logger.info(f"Saved ROC for best model: {best_model} ({task_name})")
                except Exception:
                    pass
            break


def format_cv_results(cv_results: List[dict]) -> pd.DataFrame:
    """Convert CV results list to DataFrame. Combined save happens in run_all."""
    return pd.DataFrame(cv_results)
