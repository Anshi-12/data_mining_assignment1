# Demo datasets

These synthetic datasets are bundled only to demonstrate application behavior. The application itself is dataset-agnostic.

- **classification_churn.csv** — mixed numeric/categorical customer-style data with an explicit binary `churn` target. Includes a small amount of predictor missingness so the leakage-safe imputation path can be discussed.
- **regression_housing.csv** — mixed numeric/categorical housing-style data with continuous `price` target. Useful for showing MAE/RMSE/R² and residual diagnostics.
- **skip_no_features.csv** — an ID-like unique column plus constants. Structural preparation leaves no useful analytical feature set, intentionally demonstrating graceful skip behavior.
