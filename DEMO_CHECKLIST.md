# Demo Checklist — Assignment 1, Part 1

Use this as a camera-ready click path. The clean classification demo exercises the entire pipeline; the regression and skip demos show adaptive behavior.

## Before recording

1. Activate the project virtual environment.
2. Run `pytest` and keep the passing result available if you want to show QA.
3. Start the app with `streamlit run app.py`.
4. Optional for full-fidelity PDF charts: confirm Chrome/Chromium is installed. If not, the fallback PDF path is still a valid demo.
5. Keep `demo_data/classification_churn.csv` ready in the file picker.

## Main end-to-end demo — classification

### 1. Upload

- Click **Upload**.
- Upload `demo_data/classification_churn.csv`.
- Show successful validation, dimensions, parsing metadata, preview, and column overview.
- Mention that invalid uploads clear old downstream results.

### 2. Understanding

- Click **Understanding**.
- Point out **KNOWN** vs **INFERRED** objective text.
- Show per-column profiles, missingness/cardinality, target candidates, identifier/leakage review flags, and data-quality warnings.
- Point out `churn` as a suggested target but emphasize it is not automatically selected.

### 3. Preparation

- Click **Preparation**.
- Show **Applied now — structural** vs **Deferred to modeling — learned**.
- Point out that any missing predictor remains missing in the structural preview.
- Say: “Imputation/scaling/encoding have not been fit globally.”

### 4. EDA

- Click **EDA**.
- Show at least one computed finding and its matching interactive Plotly chart.
- Hover a chart.
- Show missingness and one relationship/outlier diagnostic.
- Mention unsupported analysis is recorded as a skip rather than fabricated.

### 5. Clustering

- Click **Clustering**.
- If accepted: show chosen `k`, silhouette, candidate diagnostics, PCA scatter, profiles, and data-derived segment descriptions.
- If skipped: show the explicit meaningfulness reason. A skip is a valid result.

### 6. Modeling

- Click **Modeling**.
- First show that the target selector says **— Select a target —**.
- Select `churn` explicitly.
- Show detected classification task, stratified split, Dummy baseline, Logistic Regression, Random Forest, HistGradientBoosting, test metrics, CV mean ± std, and divergence flags.
- State: “The split happens before preprocessing is fit; CV refits the full Pipeline inside each fold.”

### 7. Evaluation

- Click **Evaluation**.
- Show selected model or the explicit no-defensible-model state.
- Show baseline improvement and the stability/overfitting/complexity reasoning.
- If a model is selected, show held-out permutation importance and confusion matrix.
- Read the five recommendation headings: what learned, reliability, action, limitations, next steps.

### 8. Report

- Click **Report**.
- Show **Download HTML Report** and **Download PDF Report**.
- Download/open HTML and show an interactive chart.
- Download/open PDF.
- If Chrome is installed, point out static chart equivalents.
- If Chrome is absent, point out the friendly PDF fidelity warning and chart placeholders; emphasize that the PDF still succeeds.

## Quick regression demo

1. Upload `demo_data/regression_housing.csv`.
2. Run Understanding → Preparation → EDA.
3. In Modeling, explicitly select `price`.
4. Show `Regression`, DummyRegressor, Ridge, Random Forest, HistGradientBoosting, MAE/RMSE/R².
5. In Evaluation, show residual diagnostics instead of a confusion matrix.

## Honest-skip demo

1. Upload `demo_data/skip_no_features.csv`.
2. Show that ingestion/understanding still work.
3. In Preparation, point out that the ID-like and constant columns are removed from the analytical feature set.
4. EDA/Clustering should explain that usable analysis/features are unavailable.
5. Modeling has no defensible feature/target path to force.
6. Emphasize: “The application records unsupported work explicitly instead of manufacturing a result.”

## Final QA proof (optional on camera)

```powershell
pytest
python scripts/smoke_test.py
python scripts/hardening_matrix.py
```

In the smoke output, point to the final line:

```text
Deferred transformer is unfitted after reporting: True
```
