"""
Regression evaluation: compute MAE / RMSE / R², save comparison table,
generate plots only for the best model.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import get_logger, save_figure, save_table, timer
from src.visualize import plot_predicted_vs_actual, plot_residuals


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for one model."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


@timer
def evaluate_regression_holdout(holdout_results: List[dict]) -> pd.DataFrame:
    """
    Compute metrics for every holdout result and save comparison table.
    Does NOT generate plots — call plot_best_regression separately.
    """
    logger = get_logger()
    rows = []

    for res in holdout_results:
        y_true = res["y_test"]
        y_pred = res["y_pred"]
        model_name = res["model"]
        exp_name = res["experiment"]

        metrics = _compute_metrics(y_true, y_pred)
        row = {"experiment": exp_name, "model": model_name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows)
    save_table(df, "regression_results.csv", index=False)
    logger.info(f"Saved regression_results.csv ({len(df)} rows)")
    return df


def plot_best_regression(holdout_results: List[dict], metrics_df: pd.DataFrame) -> None:
    """
    Generate predicted-vs-actual and residual plots for the best baseline
    model only. Best is determined by lowest MAE on baseline features.
    """
    logger = get_logger()

    baseline_mask = metrics_df["experiment"] == "baseline"
    if baseline_mask.sum() == 0:
        logger.warning("No baseline results found for regression")
        return

    baseline_df = metrics_df[baseline_mask]
    best_model = baseline_df.loc[baseline_df["MAE"].idxmin(), "model"]

    for res in holdout_results:
        if res["model"] == best_model and res["experiment"] == "baseline":
            y_true = res["y_test"]
            y_pred = res["y_pred"]

            fig = plot_predicted_vs_actual(
                y_true, y_pred,
                title=f"Predicted vs Actual — {best_model}",
            )
            save_figure(fig, f"reg_pred_vs_actual_{best_model}.png")

            fig_res = plot_residuals(
                y_true, y_pred,
                title=f"Residuals — {best_model}",
            )
            save_figure(fig_res, f"reg_residuals_{best_model}.png")

            logger.info(f"Saved regression plots for best model: {best_model}")
            break


def format_regression_cv(cv_results: List[dict]) -> pd.DataFrame:
    """Save regression CV summary table."""
    logger = get_logger()
    df = pd.DataFrame(cv_results)
    save_table(df, "regression_cv_summary.csv", index=False)
    logger.info("Saved regression_cv_summary.csv")
    return df
