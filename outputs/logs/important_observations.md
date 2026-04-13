# Important Observations

This file is auto-generated. Review the tables and figures for detailed results.

## Best Classification Models (by F1 Macro on hold-out)
- **binary**: GradientBoosting (Acc=0.753, F1_macro=0.752)
- **3class_balanced**: RandomForest (Acc=0.631, F1_macro=0.598)
- **3class_strict**: GradientBoosting (Acc=0.665, F1_macro=0.587)
- **4class**: RandomForest (Acc=0.631, F1_macro=0.598)

## Best Regression Model (by MAE on hold-out)
- **regression**: SVR (MAE=0.934, RMSE=1.224, R²=0.299)

## Notes
- Dataset size: 200 samples → results may vary with different splits
- XGBoost was available
- Check class_mapping.md for target definitions and class imbalance info
- Review confusion matrices in outputs/figures/ for per-class performance