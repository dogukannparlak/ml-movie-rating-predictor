# Experiment Notes

- Random state: 42
- Test size: 0.2
- XGBoost available: True
- Number of features (baseline): 22
- Features used: ['Runtime (mins)', 'Year', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']

## Classification tasks
- binary: rating < 6 → low, >= 6 → high
- 3class_balanced: quantile-based 3 equal-frequency bins
- 3class_strict: low (1-4), medium (5-6), high (7-10)
- 4class: quantile-based 4 equal-frequency bins

## Regression task
- Predict raw 'Your Rating' (1-10)

## Models used
### Classification:
- LogisticRegression, RandomForest, GradientBoosting, XGBoost, SVM, MLP, StackingClassifier
### Regression:
- RandomForest, GradientBoosting, SVR, XGBoost, MLP

## Feature selection methods
- Filter: correlation threshold + mutual information
- Wrapper: RFE with LogisticRegression / RandomForest
- Embedded: RandomForest + XGBoost feature importances

## Evaluation
- Hold-out: 80/20 split
- Cross-validation: 5-fold (stratified for classification, standard for regression)