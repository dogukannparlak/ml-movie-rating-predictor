"""
Load and validate the movies dataset.
"""

import pandas as pd

from src.config import DATA_PATH, TARGET_COL, FEATURE_COLS
from src.utils import get_logger


def load_data(path: str = None) -> pd.DataFrame:
    """
    Load the CSV dataset robustly.

    Parameters
    ----------
    path : str, optional
        Path to CSV file. Defaults to config DATA_PATH.

    Returns
    -------
    pd.DataFrame
        The loaded and lightly validated dataset.
    """
    logger = get_logger()
    csv_path = path or str(DATA_PATH)
    logger.info(f"Loading data from {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate essential columns
    missing_cols = [c for c in [TARGET_COL] + FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")

    # Basic type coercion
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Drop rows where target is NaN
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    if len(df) < n_before:
        logger.warning(f"Dropped {n_before - len(df)} rows with missing target")

    logger.info(f"Dataset ready: {df.shape}")
    return df
