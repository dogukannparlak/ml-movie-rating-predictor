"""
Central configuration for the ML project.
All seeds, paths, model parameters, and task definitions live here.
"""

import os
from pathlib import Path

# ─── Reproducibility ────────────────────────────────────────────────
RANDOM_STATE = 42

# ─── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "movies.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# ─── Data columns ──────────────────────────────────────────────────
TARGET_COL = "Your Rating"
DROP_COLS = ["Original Title", "Genres"]  # text columns not used directly

GENRE_COLS = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western",
]

NUMERIC_COLS = ["Runtime (mins)", "Year"]

FEATURE_COLS = NUMERIC_COLS + GENRE_COLS  # 22 features total

# ─── Evaluation settings ───────────────────────────────────────────
TEST_SIZE = 0.20
CV_FOLDS = 5

# ─── Classification target definitions ─────────────────────────────
BINARY_THRESHOLD = 6  # low: <6, high: >=6

STRICT_3CLASS_BINS = [0, 4, 6, 10]       # edges: (0,4], (4,6], (6,10]
STRICT_3CLASS_LABELS = ["low", "medium", "high"]

# Balanced 3-class and 4-class use quantile-based binning (computed at runtime)

# ─── XGBoost availability ──────────────────────────────────────────
try:
    import xgboost  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ─── Hyperparameter grids (kept small for speed) ───────────────────
PARAM_GRIDS = {
    "RandomForest_clf": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
    },
    "GradientBoosting_clf": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
    },
    "RandomForest_reg": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
    },
    "GradientBoosting_reg": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
    },
}
