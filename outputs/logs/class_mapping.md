# Classification Target Mappings

## binary
- Number of classes: 2
- Value counts:
  - Class 0: 657 samples (50.1%)
  - Class 1: 655 samples (49.9%)
- Rule: low = rating < 6, high = rating >= 6

## 3class_balanced
- Number of classes: 3
- Value counts:
  - Class 0: 657 samples (50.1%)
  - Class 1: 392 samples (29.9%)
  - Class 2: 263 samples (20.0%)
- Rule: quantile-based 3 equal-frequency bins
- Bin edges: [np.float64(1.0), np.float64(5.0), np.float64(6.0), np.float64(10.0)]

## 3class_strict
- Number of classes: 3
- Value counts:
  - Class 0: 312 samples (23.8%)
  - Class 1: 737 samples (56.2%)
  - Class 2: 263 samples (20.0%)
- Rule: low = 1-4, medium = 5-6, high = 7-10

## 4class
- Number of classes: 3
- Value counts:
  - Class 0: 657 samples (50.1%)
  - Class 1: 392 samples (29.9%)
  - Class 2: 263 samples (20.0%)
- Rule: quantile-based 4 equal-frequency bins
- Bin edges: [np.float64(1.0), np.float64(5.0), np.float64(6.0), np.float64(10.0)]
