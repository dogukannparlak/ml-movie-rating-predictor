# IMDb Rating Prediction — ML Project

Machine learning project that predicts personal IMDb ratings using both **regression** and **classification** approaches. Built for the CSE315 Machine Learning course assignment.

## Dataset

- **Location**: `data/movies.csv`
- **Size**: 200 movies
- **Target**: `Your Rating` (1–10 scale)
- **Features**: Runtime (mins), Year, 20 binary genre indicator columns (Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Family, Fantasy, History, Horror, Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, Western)

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
# Run the full pipeline (EDA + feature selection + classification + regression)
python -m src.run_all

# Run only classification tasks
python -m src.run_all --task cls

# Run only regression task
python -m src.run_all --task reg
```

All outputs are saved automatically under `outputs/`.

## Tasks

### Regression
Predict the raw `Your Rating` score (1–10).

**Models**: RandomForest, GradientBoosting, SVR, XGBoost*, MLP

**Metrics**: MAE, RMSE, R²

### Classification
Four formulations derived from the rating:

| Task | Classes | Rule |
|------|---------|------|
| **Binary** | 2 | low (< 6) vs high (≥ 6) |
| **3-class balanced** | 3 | Quantile-based equal-frequency bins |
| **3-class strict** | 3 | low (1–4), medium (5–6), high (7–10) |
| **4-class** | 4 | Quantile-based equal-frequency bins |

**Models**: LogisticRegression, RandomForest, GradientBoosting, XGBoost*, SVM, MLP, StackingClassifier

**Metrics**: Accuracy, F1 (macro/weighted), Precision, Recall, Log Loss, ROC-AUC (binary only)

*\*XGBoost is optional — the pipeline skips it gracefully if not installed.*

## Feature Selection

Three methods implemented and compared:

1. **Filter**: Correlation threshold + Mutual Information ranking
2. **Wrapper**: Recursive Feature Elimination (RFE) with LogisticRegression / RandomForest
3. **Embedded**: Feature importance from RandomForest + XGBoost

Permutation importance is computed for the best model of each task type.

## Outputs

### Tables (`outputs/tables/`)
- `dataset_summary.csv` — descriptive statistics
- `missing_values.csv` — missing value report
- `target_distribution.csv` — rating value counts
- `correlation_with_target.csv` — feature-target correlations
- `classification_{task}_results.csv` — hold-out metrics per classification task
- `classification_cv_{task}_summary.csv` — cross-validation results
- `classification_cv_summary.csv` — combined CV summary
- `regression_results.csv` — hold-out regression metrics
- `regression_cv_summary.csv` — regression CV results
- `feature_selection_filter_{task}.csv` — filter method rankings
- `feature_selection_wrapper_{task}.csv` — RFE rankings
- `feature_selection_embedded_{task}.csv` — embedded method rankings
- `best_models_summary.csv` — best model per task

### Figures (`outputs/figures/`)
- Rating histogram
- Class distribution plots (4 classification tasks)
- Correlation heatmap
- Feature importance plots
- Confusion matrices for all models × tasks
- ROC curves (binary classification)
- Predicted vs Actual scatter plots (regression)
- Residual plots (regression)
- Permutation importance plots

### Logs (`outputs/logs/`)
- `experiment_notes.md` — experiment configuration summary
- `class_mapping.md` — target label definitions and class counts
- `important_observations.md` — auto-generated best-model highlights
- `run.log` — full execution log

## Project Structure

```
.
├── data/
│   └── movies.csv
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── target_building.py
│   ├── feature_selection.py
│   ├── train_classification.py
│   ├── evaluate_classification.py
│   ├── train_regression.py
│   ├── evaluate_regression.py
│   ├── visualize.py
│   └── run_all.py
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   └── logs/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## Methodology

Follows the **CRISP-DM** process:

1. **Data Understanding** — EDA, descriptive stats, correlation analysis
2. **Data Preparation** — missing value handling, target encoding, feature selection
3. **Modeling** — multiple algorithms with sklearn Pipelines (scaling inside pipeline to prevent data leakage)
4. **Evaluation** — hold-out (80/20) + k-fold cross-validation (k=5, stratified for classification)

All preprocessing (scaling, feature selection) is applied only on training data to prevent data leakage.

## Reproducibility

- Fixed `random_state=42` throughout
- All results are deterministic given the same seed
- Pipeline saves all intermediate tables and figures for inspection
