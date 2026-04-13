"""
Feature selection: filter, wrapper, and embedded methods.
Also provides permutation importance analysis.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance as sklearn_perm_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, XGBOOST_AVAILABLE
from src.utils import get_logger, save_table, save_figure, timer
from src.visualize import plot_feature_importance, plot_permutation_importance

if XGBOOST_AVAILABLE:
    from xgboost import XGBClassifier, XGBRegressor


# ─── Filter method ──────────────────────────────────────────────────
def filter_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str = "classification",
    corr_threshold: float = 0.05,
    mi_top_k: int = 15,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Filter-based feature selection:
    1. Remove features with absolute correlation to target below threshold.
    2. Rank remaining features by mutual information; keep top_k.

    Parameters
    ----------
    task : str
        "classification" or "regression" — determines MI estimator.

    Returns
    -------
    selected : list of str
    ranking_df : pd.DataFrame with correlation and MI scores
    """
    logger = get_logger()

    # Correlation with target
    corr = X_train.corrwith(y_train).abs().rename("abs_correlation")

    # Mutual information
    mi_func = mutual_info_classif if task == "classification" else mutual_info_regression
    mi_scores = mi_func(X_train, y_train, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=X_train.columns, name="mutual_info")

    ranking = pd.concat([corr, mi_series], axis=1).sort_values("mutual_info", ascending=False)
    ranking["above_corr_threshold"] = ranking["abs_correlation"] >= corr_threshold

    # Select: correlation above threshold AND top-k MI
    above_corr = ranking[ranking["above_corr_threshold"]].index.tolist()
    top_mi = ranking.head(mi_top_k).index.tolist()
    selected = list(dict.fromkeys(above_corr + top_mi))  # union, preserving order

    ranking["selected"] = ranking.index.isin(selected)
    logger.info(f"Filter selection ({task}): {len(selected)}/{len(X_train.columns)} features")
    return selected, ranking


# ─── Wrapper method ─────────────────────────────────────────────────
def wrapper_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str = "classification",
    n_features: int = 10,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Wrapper-based selection using Recursive Feature Elimination (RFE).
    Uses LogisticRegression for classification, RandomForestRegressor for regression.
    """
    logger = get_logger()

    if task == "classification":
        base = LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs",
            class_weight="balanced",
        )
        # Scale for logistic regression
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                columns=X_train.columns, index=X_train.index)
    else:
        base = RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
        )
        X_scaled = X_train

    n_features_to_select = min(n_features, X_train.shape[1])
    rfe = RFE(estimator=base, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X_scaled, y_train)

    ranking = pd.DataFrame({
        "feature": X_train.columns,
        "rfe_ranking": rfe.ranking_,
        "selected": rfe.support_,
    }).sort_values("rfe_ranking")

    selected = ranking[ranking["selected"]]["feature"].tolist()
    logger.info(f"Wrapper/RFE selection ({task}): {len(selected)}/{len(X_train.columns)} features")
    return selected, ranking


# ─── Embedded method ────────────────────────────────────────────────
def embedded_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str = "classification",
    top_k: int = 12,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Embedded feature selection using tree-based feature importance.
    Uses RandomForest + XGBoost (if available), averages importances.
    """
    logger = get_logger()

    importances = {}

    if task == "classification":
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
                                    class_weight="balanced")
        rf.fit(X_train, y_train)
        importances["rf"] = rf.feature_importances_

        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                use_label_encoder=False, eval_metric="mlogloss",
                                verbosity=0)
            xgb.fit(X_train, y_train)
            importances["xgb"] = xgb.feature_importances_
    else:
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances["rf"] = rf.feature_importances_

        if XGBOOST_AVAILABLE:
            xgb = XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0)
            xgb.fit(X_train, y_train)
            importances["xgb"] = xgb.feature_importances_

    # Average importances
    imp_df = pd.DataFrame(importances, index=X_train.columns)
    imp_df["avg_importance"] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values("avg_importance", ascending=False)
    imp_df["selected"] = False
    imp_df.iloc[:top_k, imp_df.columns.get_loc("selected")] = True

    selected = imp_df[imp_df["selected"]].index.tolist()
    logger.info(f"Embedded selection ({task}): {len(selected)}/{len(X_train.columns)} features")

    # Plot
    fig = plot_feature_importance(
        imp_df["avg_importance"].values,
        imp_df.index.tolist(),
        title=f"Embedded Feature Importance ({task})",
    )
    save_figure(fig, f"feature_importance_embedded_{task}.png")

    return selected, imp_df


# ─── Run all feature selection methods ──────────────────────────────
@timer
def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str = "classification",
) -> Dict[str, List[str]]:
    """
    Run all three feature selection methods and save outputs.

    Returns dict with keys 'filter', 'wrapper', 'embedded' mapping to selected feature lists.
    """
    logger = get_logger()

    results = {}

    # Filter
    sel_filter, rank_filter = filter_selection(X_train, y_train, task=task)
    results["filter"] = sel_filter

    # Wrapper
    sel_wrapper, rank_wrapper = wrapper_selection(X_train, y_train, task=task)
    results["wrapper"] = sel_wrapper

    # Embedded
    sel_embedded, rank_embedded = embedded_selection(X_train, y_train, task=task)
    results["embedded"] = sel_embedded

    # ── Build ONE consolidated feature selection table ───────────────
    consolidated = pd.DataFrame(index=X_train.columns)
    consolidated.index.name = "feature"

    # Filter columns
    consolidated["abs_correlation"] = rank_filter["abs_correlation"]
    consolidated["mutual_info"] = rank_filter["mutual_info"]
    consolidated["filter_selected"] = rank_filter["selected"]

    # Wrapper columns
    wrapper_indexed = rank_wrapper.set_index("feature")
    consolidated["rfe_ranking"] = wrapper_indexed["rfe_ranking"]
    consolidated["wrapper_selected"] = wrapper_indexed["selected"]

    # Embedded columns
    consolidated["tree_importance"] = rank_embedded["avg_importance"]
    consolidated["embedded_selected"] = rank_embedded["selected"]

    # Summary column
    consolidated["methods_selected"] = (
        consolidated["filter_selected"].astype(int)
        + consolidated["wrapper_selected"].astype(int)
        + consolidated["embedded_selected"].astype(int)
    )
    consolidated = consolidated.sort_values("methods_selected", ascending=False)

    save_table(consolidated, f"feature_selection_{task}.csv")

    logger.info(f"Feature selection ({task}) complete — "
                f"filter={len(sel_filter)}, wrapper={len(sel_wrapper)}, "
                f"embedded={len(sel_embedded)}")

    return results


# ─── Permutation importance ─────────────────────────────────────────
def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "classification",
    model_name: str = "BestModel",
) -> None:
    """Compute and save permutation importance for a fitted model."""
    logger = get_logger()

    scoring = "accuracy" if task == "classification" else "neg_mean_absolute_error"
    perm = sklearn_perm_importance(
        model, X_test, y_test, n_repeats=15, random_state=RANDOM_STATE,
        scoring=scoring, n_jobs=-1,
    )

    fig = plot_permutation_importance(
        perm, X_test.columns.tolist(),
        title=f"Permutation Importance — {model_name} ({task})",
    )
    save_figure(fig, f"permutation_importance_{task}.png")
    logger.info(f"Permutation importance plot saved for {model_name} ({task})")
