# Architecture

## Data flow

CRISP-DM Studio uses explicit immutable/structured results between phases. UI pages are renderers and orchestration boundaries; analytical logic lives in the domain modules.

```text
CSV upload
   ↓
ingest_csv()
   ↓
IngestedDataset
   ↓
profile_dataset()
   ↓
UnderstandingResult
   ↓
prepare_dataset()
   ↓
PreparationResult
   ├── structural_dataframe
   └── unfitted PreprocessingSpec
          │
          ├──────────────→ run_eda() → EDAResult
          │
          ├──────────────→ run_clustering() → ClusteringResult
          │
          └──────────────→ run_modeling(target) → ModelingResult
                                               ↓
                                      evaluate_modeling()
                                               ↓
                                        EvaluationResult
                                               ↓
                                          build_report()
                                               ↓
                                           ReportResult
```

The Report stage consumes results only. It does not receive raw analysis instructions and does not refit/recompute the earlier phases.

## Leakage boundary

Phase 3 separates two classes of transformation.

### Safe structural cleaning — may run before splitting

- exact deduplication;
- constant-column removal;
- identifier exclusion;
- header normalization performed at ingestion;
- lossless numeric coercion.

These are deterministic structural operations and do not estimate population parameters for supervised prediction.

### Learned supervised transformations — deferred

- median / mode imputation;
- IQR outlier bounds;
- numeric scaling;
- one-hot category vocabulary and infrequent-category handling.

They are represented by `PreprocessingSpec` but are **not fitted** in Phase 3.

```text
structural_dataframe + explicit target
              ↓
       remove missing target rows
              ↓
         SPLIT FIRST
       ┌──────┴──────┐
       ↓             ↓
    X_train         X_test
       ↓
Fresh preprocessing transformer
       +
Estimator
       ↓
sklearn Pipeline.fit(X_train, y_train)
       ↓
Evaluate on untouched X_test
```

For cross-validation, sklearn clones and fits the **entire Pipeline** independently inside each training fold. The imputer, IQR bounds, scaler, categorical vocabulary, and estimator therefore never learn from that fold's validation rows.

The original Phase 3 `PreprocessingSpec` remains unfitted throughout the application. `scripts/smoke_test.py` asserts this after clustering, modeling, evaluation, and reporting.

## Why clustering is different

Clustering is an unsupervised analysis with no target/holdout outcome. Its independent clustering preprocessor may be fit on the full unlabeled clustering dataset because that dataset is the population being segmented. It never uses or mutates the Phase 3 supervised transformer.

## Analyze once / render many ways

Analysis engines return reusable result objects. Charts are stored as Plotly `Figure` objects inside `EDAChart`, `ClusteringChart`, or `EvaluationChart`.

```text
stored Figure
   ├── Streamlit → interactive chart
   ├── HTML report → interactive Plotly fragment
   └── PDF → static image of the same Figure
                  └── fallback placeholder if static renderer is unavailable
```

Text follows the same pattern: EDA findings, preparation decisions, segment descriptions, model metrics, and the final recommendation are computed once, stored, and rendered verbatim later.

## Failure and state boundaries

Each Streamlit page has a stage-level exception boundary. An unexpected failure logs the technical exception, shows a friendly message, and invalidates the failed stage plus dependent state. `app.py` adds a final last-resort boundary.

A new upload or failed upload clears all downstream state. This prevents a report/model result from a previous dataset from remaining visible after the active dataset changes.

## Resource controls

Configured limits bound expensive work:

- upload: 50 MB;
- ingestion: 500,000 rows / 500 columns;
- EDA: 100,000-row working cap and bounded relationship dimensions;
- clustering: 20,000-row working cap;
- supervised modeling: 50,000-row working cap before the split;
- permutation importance: 5,000 held-out rows, 5 repeats.

Sampling uses a deterministic random seed when required and is surfaced to the user through resource notes.
