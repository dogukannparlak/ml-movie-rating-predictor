"""
Classification model training pipeline.
Builds sklearn Pipelines (with scaling where needed), runs hold-out + CV,
optional hyperparameter tuning, and returns results DataFrames.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    CV_FOLDS,
    PARAM_GRIDS,
    RANDOM_STATE,
    TEST_SIZE,
    XGBOOST_AVAILABLE,
)
from src.utils import get_logger, timer

if XGBOOST_AVAILABLE:
    from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─── Model factory ──────────────────────────────────────────────────

def _get_classification_models(n_classes: int = 2) -> Dict[str, Pipeline]:
    """
    Return a dict of {name: sklearn Pipeline} for classification models.
    Scaling is embedded in pipelines for models that need it.
    """
    models = {}

    # Logistic Regression (needs scaling)
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE,
            class_weight="balanced", solver="lbfgs",
        )),
    ])

    # Random Forest (no scaling)
    models["RandomForest"] = Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1,
        )),
    ])

    # Gradient Boosting (no scaling)
    models["GradientBoosting"] = Pipeline([
        ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE,
        )),
    ])

    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("model", XGBClassifier(
                n_estimators=200, learning_rate=0.1,
                random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric="mlogloss", verbosity=0,
            )),
        ])

    # SVM (needs scaling)
    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    # MLP (needs scaling)
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=RANDOM_STATE, early_stopping=True,
            validation_fraction=0.15,
        )),
    ])

    # Stacking: RF + GB + SVM as base, LogReg meta
    base_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1)),
        ("gb", GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE)),
        ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, random_state=RANDOM_STATE,
                        class_weight="balanced")),
        ])),
    ]
    models["Stacking"] = Pipeline([
        ("model", StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced",
            ),
            cv=3, n_jobs=-1,
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
    task_name: str,
) -> Tuple[List[dict], Dict[str, Pipeline]]:
    """
    Train each model on the hold-out train set, predict on test.
    Returns list of result dicts and dict of fitted models.
    """
    logger = get_logger()
    results = []
    fitted_models = {}

    for name, pipe in models.items():
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = None
            if hasattr(pipe, "predict_proba"):
                try:
                    y_proba = pipe.predict_proba(X_test)
                except Exception:
                    pass

            fitted_models[name] = pipe
            results.append({
                "task": task_name,
                "model": name,
                "eval_type": "holdout",
                "y_test": y_test.values,
                "y_pred": y_pred,
                "y_proba": y_proba,
            })
            logger.info(f"  {task_name} | {name} holdout — done")
        except Exception as e:
            logger.warning(f"  {task_name} | {name} holdout FAILED: {e}")

    return results, fitted_models


# ─── Cross-validation ───────────────────────────────────────────────

def _cv_evaluate(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    task_name: str,
) -> List[dict]:
    """
    Run stratified k-fold cross-validation and return per-model summary.
    """
    logger = get_logger()
    cv_results = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]

    for name, pipe in models.items():
        try:
            scores = cross_validate(
                pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1,
                return_train_score=False, error_score="raise",
            )
            row = {
                "task": task_name,
                "model": name,
                "eval_type": "cv",
            }
            for metric in scoring:
                key = f"test_{metric}"
                row[f"cv_{metric}_mean"] = scores[key].mean()
                row[f"cv_{metric}_std"] = scores[key].std()
            cv_results.append(row)
            logger.info(f"  {task_name} | {name} CV — done")
        except Exception as e:
            logger.warning(f"  {task_name} | {name} CV FAILED: {e}")

    return cv_results


# ─── Hyperparameter tuning (light) ─────────────────────────────────

def _tune_model(
    pipe: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Light GridSearchCV tuning. Returns best estimator."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid, cv=skf, scoring="f1_macro",
        n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


# ─── Main entry point ──────────────────────────────────────────────
@timer
def run_classification_task(
    X: pd.DataFrame,
    y: pd.Series,
    task_name: str,
    feature_sets: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Pipeline]]:
    """
    Full classification experiment for one task:
    - baseline features + all models (holdout + CV)
    - selected features + all models (holdout + CV)

    Parameters
    ----------
    X : feature DataFrame
    y : integer class labels
    task_name : e.g. "binary", "3class_balanced"
    feature_sets : dict with 'filter'/'wrapper'/'embedded' keys → list of feature names
                   If None, only baseline is run.

    Returns
    -------
    holdout_df : results from hold-out evaluation
    cv_df : results from cross-validation
    best_fitted : dict of best fitted models
    """
    logger = get_logger()
    n_classes = y.nunique()
    logger.info(f"═══ Classification task: {task_name} (classes={n_classes}) ═══")

    all_holdout = []
    all_cv = []
    best_fitted = {}

    # Prepare hold-out split (shared across feature sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # Define feature-set experiments
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

        models = _get_classification_models(n_classes=n_classes)

        # Light tuning on RF and GB for baseline only
        if exp_name == "baseline":
            for mname, grid_key in [("RandomForest", "RandomForest_clf"),
                                     ("GradientBoosting", "GradientBoosting_clf")]:
                if mname in models and grid_key in PARAM_GRIDS:
                    try:
                        models[mname] = _tune_model(
                            models[mname], PARAM_GRIDS[grid_key], X_tr, y_train,
                        )
                        logger.info(f"  Tuned {mname} for {task_name}")
                    except Exception as e:
                        logger.warning(f"  Tuning {mname} failed: {e}")

        # Hold-out
        ho_results, fitted = _holdout_evaluate(
            models, X_tr, X_te, y_train, y_test, f"{task_name}_{exp_name}",
        )
        all_holdout.extend(ho_results)

        if exp_name == "baseline":
            best_fitted = fitted

        # Cross-validation
        cv_results = _cv_evaluate(models, X_full, y, f"{task_name}_{exp_name}")
        all_cv.extend(cv_results)

    return all_holdout, all_cv, best_fitted
