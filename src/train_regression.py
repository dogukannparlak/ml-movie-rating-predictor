"""
Regression model training pipeline.
Builds sklearn Pipelines (with scaling where needed), runs hold-out + CV.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import (
    CV_FOLDS,
    PARAM_GRIDS,
    RANDOM_STATE,
    TEST_SIZE,
    XGBOOST_AVAILABLE,
)
from src.utils import get_logger, timer

if XGBOOST_AVAILABLE:
    from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─── Model factory ──────────────────────────────────────────────────

def _get_regression_models() -> Dict[str, Pipeline]:
    """Return a dict of {name: sklearn Pipeline} for regression models."""
    models = {}

    models["RandomForest"] = Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    models["GradientBoosting"] = Pipeline([
        ("model", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE,
        )),
    ])

    models["SVR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf")),
    ])

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("model", XGBRegressor(
                n_estimators=200, learning_rate=0.1,
                random_state=RANDOM_STATE, verbosity=0,
            )),
        ])

    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=RANDOM_STATE, early_stopping=True,
            validation_fraction=0.15,
        )),
    ])

    return models


# ─── Hold-out evaluation ────────────────────────────────────────────

def _holdout_evaluate(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    exp_name: str,
) -> Tuple[List[dict], Dict[str, Pipeline]]:
    """Train on hold-out train, predict on test. Return result dicts + fitted models."""
    logger = get_logger()
    results = []
    fitted_models = {}

    for name, pipe in models.items():
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            fitted_models[name] = pipe
            results.append({
                "experiment": exp_name,
                "model": name,
                "eval_type": "holdout",
                "y_test": y_test.values,
                "y_pred": y_pred,
            })
            logger.info(f"  regression | {name} holdout ({exp_name}) — done")
        except Exception as e:
            logger.warning(f"  regression | {name} holdout ({exp_name}) FAILED: {e}")

    return results, fitted_models


# ─── Cross-validation ───────────────────────────────────────────────

def _cv_evaluate(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    exp_name: str,
) -> List[dict]:
    """Run k-fold CV and return per-model summary."""
    logger = get_logger()
    results = []
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]

    for name, pipe in models.items():
        try:
            scores = cross_validate(
                pipe, X, y, cv=kf, scoring=scoring, n_jobs=-1,
                return_train_score=False, error_score="raise",
            )
            row = {"experiment": exp_name, "model": name, "eval_type": "cv"}
            for metric in scoring:
                key = f"test_{metric}"
                vals = scores[key]
                # Negate negative metrics
                if metric.startswith("neg_"):
                    vals = -vals
                clean = metric.replace("neg_", "")
                row[f"cv_{clean}_mean"] = vals.mean()
                row[f"cv_{clean}_std"] = vals.std()
            results.append(row)
            logger.info(f"  regression | {name} CV ({exp_name}) — done")
        except Exception as e:
            logger.warning(f"  regression | {name} CV ({exp_name}) FAILED: {e}")

    return results


# ─── Hyperparameter tuning (light) ─────────────────────────────────

def _tune_model(
    pipe: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Light GridSearchCV tuning. Returns best estimator."""
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid, cv=kf, scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


# ─── Main entry point ──────────────────────────────────────────────

@timer
def run_regression_task(
    X: pd.DataFrame,
    y: pd.Series,
    feature_sets: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[dict], List[dict], Dict[str, Pipeline]]:
    """
    Full regression experiment:
    - baseline features + all models (holdout + CV)
    - selected features + all models (holdout + CV)

    Returns
    -------
    all_holdout : list of result dicts
    all_cv : list of result dicts
    best_fitted : dict of fitted models (baseline)
    """
    logger = get_logger()
    logger.info("═══ Regression task ═══")

    all_holdout = []
    all_cv = []
    best_fitted = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    # Feature-set experiments
    experiments = {"baseline": X.columns.tolist()}
    if feature_sets:
        for fs_name, fs_cols in feature_sets.items():
            if fs_cols:
                experiments[f"fs_{fs_name}"] = fs_cols

    for exp_name, cols in experiments.items():
        logger.info(f"── Feature set: {exp_name} ({len(cols)} features) ──")
        X_tr = X_train[cols]
        X_te = X_test[cols]
        X_full = X[cols]

        models = _get_regression_models()

        # Light tuning
        if exp_name == "baseline":
            for mname, grid_key in [("RandomForest", "RandomForest_reg"),
                                     ("GradientBoosting", "GradientBoosting_reg")]:
                if mname in models and grid_key in PARAM_GRIDS:
                    try:
                        models[mname] = _tune_model(
                            models[mname], PARAM_GRIDS[grid_key], X_tr, y_train,
                        )
                        logger.info(f"  Tuned {mname} for regression")
                    except Exception as e:
                        logger.warning(f"  Tuning {mname} failed: {e}")

        # Hold-out
        ho_results, fitted = _holdout_evaluate(
            models, X_tr, X_te, y_train, y_test, exp_name,
        )
        all_holdout.extend(ho_results)

        if exp_name == "baseline":
            best_fitted = fitted

        # CV
        cv_results = _cv_evaluate(models, X_full, y, exp_name)
        all_cv.extend(cv_results)

    return all_holdout, all_cv, best_fitted
