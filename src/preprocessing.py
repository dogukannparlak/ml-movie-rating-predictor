"""
Data preprocessing: cleaning, EDA outputs, and descriptive statistics.
"""

import pandas as pd
import numpy as np

from src.config import TARGET_COL, FEATURE_COLS, NUMERIC_COLS, GENRE_COLS
from src.utils import get_logger, save_table, save_figure, timer
from src.visualize import plot_target_distribution, plot_correlation_heatmap


@timer
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in feature columns.
    - Numeric columns: fill with median.
    - Binary genre columns: fill with 0 (absent).

    Returns cleaned DataFrame.
    """
    logger = get_logger()
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {n_miss} missing in '{col}' with median={median_val}")

    for col in GENRE_COLS:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                df[col].fillna(0, inplace=True)
                logger.info(f"Filled {n_miss} missing in '{col}' with 0")

    return df


@timer
def generate_eda(df: pd.DataFrame) -> None:
    """
    Generate and save all EDA artifacts:
    - dataset_summary.csv
    - missing_values.csv
    - target_distribution.csv
    - correlation_with_target.csv
    - histogram of raw ratings (figure)
    - correlation heatmap (figure)
    """
    logger = get_logger()

    # ── Descriptive statistics ──────────────────────────────────────
    desc = df[FEATURE_COLS + [TARGET_COL]].describe().T
    save_table(desc, "dataset_summary.csv")
    logger.info("Saved dataset_summary.csv")

    # ── Missing values ──────────────────────────────────────────────
    miss = df[FEATURE_COLS + [TARGET_COL]].isnull().sum().rename("missing_count").to_frame()
    miss["missing_pct"] = (miss["missing_count"] / len(df) * 100).round(2)
    save_table(miss, "missing_values.csv")
    logger.info("Saved missing_values.csv")

    # ── Target distribution ─────────────────────────────────────────
    target_dist = df[TARGET_COL].value_counts().sort_index().rename("count").to_frame()
    target_dist["pct"] = (target_dist["count"] / target_dist["count"].sum() * 100).round(2)
    save_table(target_dist, "target_distribution.csv")
    logger.info("Saved target_distribution.csv")

    # ── Correlation with target ─────────────────────────────────────
    numeric_df = df[FEATURE_COLS + [TARGET_COL]].select_dtypes(include=[np.number])
    corr_target = numeric_df.corr()[TARGET_COL].drop(TARGET_COL).sort_values(
        key=abs, ascending=False
    ).rename("correlation").to_frame()
    save_table(corr_target, "correlation_with_target.csv")
    logger.info("Saved correlation_with_target.csv")

    # ── Figures ─────────────────────────────────────────────────────
    fig_hist = plot_target_distribution(df[TARGET_COL], title="Distribution of Your Rating")
    save_figure(fig_hist, "rating_histogram.png")
    logger.info("Saved rating_histogram.png")

    fig_corr = plot_correlation_heatmap(
        numeric_df, title="Feature Correlation Matrix"
    )
    save_figure(fig_corr, "correlation_heatmap.png")
    logger.info("Saved correlation_heatmap.png")


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the feature columns from a DataFrame."""
    return df[FEATURE_COLS].copy()
