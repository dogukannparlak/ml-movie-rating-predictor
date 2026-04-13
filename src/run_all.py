"""
Main orchestrator — run the full ML pipeline end-to-end.

Usage:
    python -m src.run_all              # run everything
    python -m src.run_all --task cls   # classification only
    python -m src.run_all --task reg   # regression only
"""

import argparse
import sys
import traceback
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLS,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    XGBOOST_AVAILABLE,
)
from src.data_loading import load_data
from src.evaluate_classification import (
    evaluate_classification_holdout,
    format_cv_results,
    plot_best_classification,
)
from src.evaluate_regression import (
    evaluate_regression_holdout,
    format_regression_cv,
    plot_best_regression,
)
from src.feature_selection import (
    compute_permutation_importance,
    run_feature_selection,
)
from src.preprocessing import generate_eda, get_feature_matrix, handle_missing
from src.target_building import build_all_targets
from src.train_classification import run_classification_task
from src.train_regression import run_regression_task
from src.utils import clean_output_dirs, ensure_dirs, get_logger, save_table, save_text, timer


# ─── Helpers ─────────────────────────────────────────────────────────

def _best_model_from_df(df: pd.DataFrame, metric_col: str, higher_is_better: bool = True) -> dict:
    """Return the row with the best metric value."""
    if higher_is_better:
        idx = df[metric_col].idxmax()
    else:
        idx = df[metric_col].idxmin()
    return df.loc[idx].to_dict()


# ─── Classification pipeline ────────────────────────────────────────

@timer
def pipeline_classification(
    X: pd.DataFrame, y_raw: pd.Series, targets: dict,
    feat_sel: Dict[str, List[str]],
) -> pd.DataFrame:
    """Run all classification tasks and return combined results."""
    logger = get_logger()

    all_holdout_dfs = []
    all_cv_dfs = []
    best_models_rows = []
    permutation_done = False  # only do permutation importance once

    for task_name, y_cls in targets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"CLASSIFICATION TASK: {task_name}")
        logger.info(f"{'='*60}")

        holdout_results, cv_results, fitted = run_classification_task(
            X, y_cls, task_name, feature_sets=feat_sel,
        )

        # Evaluate hold-out (metrics table only)
        ho_df = evaluate_classification_holdout(holdout_results, task_name)
        all_holdout_dfs.append(ho_df)

        # Plot confusion matrix + ROC for best baseline model only
        plot_best_classification(holdout_results, ho_df, task_name)

        # Collect CV results (no per-task CSV)
        cv_df = format_cv_results(cv_results)
        all_cv_dfs.append(cv_df)

        # Permutation importance for best model (once only, on first task)
        if fitted and not permutation_done:
            baseline_mask = ho_df["task"].str.endswith("_baseline")
            baseline_df = ho_df[baseline_mask] if baseline_mask.sum() > 0 else ho_df
            best_name = baseline_df.loc[baseline_df["f1_macro"].idxmax(), "model"] if len(baseline_df) > 0 else None
            if best_name and best_name in fitted:
                try:
                    from sklearn.model_selection import train_test_split
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y_cls, test_size=TEST_SIZE,
                        random_state=RANDOM_STATE, stratify=y_cls,
                    )
                    compute_permutation_importance(
                        fitted[best_name], X_te, y_te,
                        task="classification", model_name=f"{task_name}_{best_name}",
                    )
                    permutation_done = True
                except Exception as e:
                    logger.warning(f"Permutation importance failed for {task_name}: {e}")

        # Save best model joblib
        if fitted and len(ho_df) > 0:
            best_name_acc = ho_df.loc[ho_df["accuracy"].idxmax(), "model"]
            if best_name_acc in fitted:
                try:
                    path = MODELS_DIR / f"best_clf_{task_name}.joblib"
                    joblib.dump(fitted[best_name_acc], path)
                    logger.info(f"Saved best classification model: {path.name}")
                except Exception:
                    pass

            best_row = ho_df.loc[ho_df["f1_macro"].idxmax()]
            best_models_rows.append({
                "task_type": "classification",
                "task_name": task_name,
                "best_model": best_row.get("model", ""),
                "accuracy": best_row.get("accuracy", np.nan),
                "f1_macro": best_row.get("f1_macro", np.nan),
                "f1_weighted": best_row.get("f1_weighted", np.nan),
            })

    # Combine all classification results
    combined_ho = pd.concat(all_holdout_dfs, ignore_index=True) if all_holdout_dfs else pd.DataFrame()
    combined_cv = pd.concat(all_cv_dfs, ignore_index=True) if all_cv_dfs else pd.DataFrame()

    if not combined_cv.empty:
        save_table(combined_cv, "classification_cv_summary.csv", index=False)

    return pd.DataFrame(best_models_rows) if best_models_rows else pd.DataFrame()


# ─── Regression pipeline ────────────────────────────────────────────

@timer
def pipeline_regression(
    X: pd.DataFrame, y: pd.Series,
    feat_sel: Dict[str, List[str]],
) -> pd.DataFrame:
    """Run regression task and return best-model summary row."""
    logger = get_logger()

    logger.info(f"\n{'='*60}")
    logger.info("REGRESSION TASK")
    logger.info(f"{'='*60}")

    holdout_results, cv_results, fitted = run_regression_task(
        X, y, feature_sets=feat_sel,
    )

    # Evaluate hold-out (metrics table only)
    ho_df = evaluate_regression_holdout(holdout_results)

    # Plot pred-vs-actual + residuals for best baseline model only
    plot_best_regression(holdout_results, ho_df)

    # CV summary
    cv_df = format_regression_cv(cv_results)

    # Permutation importance for best model
    if fitted:
        best_name = ho_df.loc[ho_df["MAE"].idxmin(), "model"] if len(ho_df) > 0 else None
        if best_name and best_name in fitted:
            try:
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                )
                compute_permutation_importance(
                    fitted[best_name], X_te, y_te,
                    task="regression", model_name=f"regression_{best_name}",
                )
            except Exception as e:
                logger.warning(f"Permutation importance failed for regression: {e}")

        # Save best model joblib
        best_name_r2 = ho_df.loc[ho_df["R2"].idxmax(), "model"] if len(ho_df) > 0 else None
        if best_name_r2 and best_name_r2 in fitted:
            try:
                path = MODELS_DIR / "best_reg.joblib"
                joblib.dump(fitted[best_name_r2], path)
                logger.info(f"Saved best regression model: {path.name}")
            except Exception:
                pass

    best_rows = []
    if len(ho_df) > 0:
        best_row = ho_df.loc[ho_df["MAE"].idxmin()]
        best_rows.append({
            "task_type": "regression",
            "task_name": "regression",
            "best_model": best_row.get("model", ""),
            "MAE": best_row.get("MAE", np.nan),
            "RMSE": best_row.get("RMSE", np.nan),
            "R2": best_row.get("R2", np.nan),
        })

    return pd.DataFrame(best_rows) if best_rows else pd.DataFrame()


# ─── Summary generation ─────────────────────────────────────────────

def _generate_summaries(
    best_clf_df: pd.DataFrame,
    best_reg_df: pd.DataFrame,
) -> None:
    """Write markdown summary files to outputs/logs/."""
    logger = get_logger()

    # best_models_summary.csv
    parts = []
    if not best_clf_df.empty:
        parts.append(best_clf_df)
    if not best_reg_df.empty:
        parts.append(best_reg_df)
    if parts:
        summary = pd.concat(parts, ignore_index=True)
        save_table(summary, "best_models_summary.csv", index=False)

    # experiment_notes.md
    notes = [
        "# Experiment Notes\n",
        f"- Random state: {RANDOM_STATE}",
        f"- Test size: {TEST_SIZE}",
        f"- XGBoost available: {XGBOOST_AVAILABLE}",
        f"- Number of features (baseline): {len(FEATURE_COLS)}",
        f"- Features used: {FEATURE_COLS}",
        "",
        "## Classification tasks",
        "- binary: rating < 6 → low, >= 6 → high",
        "- 3class_balanced: quantile-based 3 equal-frequency bins",
        "- 3class_strict: low (1-4), medium (5-6), high (7-10)",
        "- 4class: quantile-based 4 equal-frequency bins",
        "",
        "## Regression task",
        "- Predict raw 'Your Rating' (1-10)",
        "",
        "## Models used",
        "### Classification:",
        "- LogisticRegression, RandomForest, GradientBoosting, "
        + ("XGBoost, " if XGBOOST_AVAILABLE else "")
        + "SVM, MLP, StackingClassifier",
        "### Regression:",
        "- RandomForest, GradientBoosting, SVR, "
        + ("XGBoost, " if XGBOOST_AVAILABLE else "")
        + "MLP",
        "",
        "## Feature selection methods",
        "- Filter: correlation threshold + mutual information",
        "- Wrapper: RFE with LogisticRegression / RandomForest",
        "- Embedded: RandomForest"
        + (" + XGBoost" if XGBOOST_AVAILABLE else "")
        + " feature importances",
        "",
        "## Evaluation",
        "- Hold-out: 80/20 split",
        "- Cross-validation: 5-fold (stratified for classification, standard for regression)",
    ]
    save_text("\n".join(notes), "experiment_notes.md")

    # important_observations.md
    obs = [
        "# Important Observations\n",
        "This file is auto-generated. Review the tables and figures for detailed results.",
        "",
    ]

    if not best_clf_df.empty:
        obs.append("## Best Classification Models (by F1 Macro on hold-out)")
        for _, row in best_clf_df.iterrows():
            obs.append(
                f"- **{row.get('task_name', '')}**: {row.get('best_model', '')} "
                f"(Acc={row.get('accuracy', 0):.3f}, F1_macro={row.get('f1_macro', 0):.3f})"
            )
        obs.append("")

    if not best_reg_df.empty:
        obs.append("## Best Regression Model (by MAE on hold-out)")
        for _, row in best_reg_df.iterrows():
            obs.append(
                f"- **{row.get('task_name', '')}**: {row.get('best_model', '')} "
                f"(MAE={row.get('MAE', 0):.3f}, RMSE={row.get('RMSE', 0):.3f}, "
                f"R²={row.get('R2', 0):.3f})"
            )
        obs.append("")

    obs.append("## Notes")
    obs.append(f"- Dataset size: 200 samples → results may vary with different splits")
    obs.append(f"- XGBoost {'was' if XGBOOST_AVAILABLE else 'was NOT'} available")
    obs.append("- Check class_mapping.md for target definitions and class imbalance info")
    obs.append("- Review confusion matrices in outputs/figures/ for per-class performance")

    save_text("\n".join(obs), "important_observations.md")
    logger.info("Saved experiment_notes.md and important_observations.md")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML Project — IMDb Rating Prediction")
    parser.add_argument(
        "--task", choices=["cls", "reg", "all"], default="all",
        help="Which task to run: cls (classification), reg (regression), all (default)",
    )
    args = parser.parse_args()

    ensure_dirs()
    clean_output_dirs()
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("ML PROJECT — IMDb Rating Prediction Pipeline")
    logger.info("=" * 70)

    try:
        # ── 1. Load data ────────────────────────────────────────────
        df = load_data()
        X = get_feature_matrix(df)
        y = df[TARGET_COL].copy()

        # ── 2. Preprocessing & EDA ──────────────────────────────────
        df_clean = handle_missing(df)
        X = get_feature_matrix(df_clean)
        y = df_clean[TARGET_COL].copy()
        generate_eda(df_clean)

        # ── 3. Build targets ────────────────────────────────────────
        targets = build_all_targets(y)

        # ── 4. Feature selection (on binary target for cls, raw for reg)
        logger.info("\n── Feature Selection ──")
        from sklearn.model_selection import train_test_split

        # Classification feature selection (use binary target)
        X_tr_cls, _, y_tr_cls, _ = train_test_split(
            X, targets["binary"], test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=targets["binary"],
        )
        feat_sel_cls = run_feature_selection(X_tr_cls, y_tr_cls, task="classification")

        # Regression feature selection
        X_tr_reg, _, y_tr_reg, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
        feat_sel_reg = run_feature_selection(X_tr_reg, y_tr_reg, task="regression")

        # ── 5. Run tasks ────────────────────────────────────────────
        best_clf_df = pd.DataFrame()
        best_reg_df = pd.DataFrame()

        if args.task in ("cls", "all"):
            best_clf_df = pipeline_classification(X, y, targets, feat_sel_cls)

        if args.task in ("reg", "all"):
            best_reg_df = pipeline_regression(X, y, feat_sel_reg)

        # ── 6. Summaries ────────────────────────────────────────────
        _generate_summaries(best_clf_df, best_reg_df)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE — check outputs/ for results")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
