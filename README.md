# Assignment 1 — Part 1

# CRISP-DM Studio

A production-style, dataset-agnostic Streamlit application that turns a reasonably sized CSV into an auditable end-to-end CRISP-DM analysis: understanding, structural preparation, exploratory analysis, clustering, supervised modeling with a baseline, reasoned evaluation, plain-English recommendation, and downloadable HTML/PDF reports.

The application is designed around two headline principles:

> **Analyze once, render many ways.** Analysis engines return structured result objects. The UI and reports render those same results and the same stored Plotly chart objects rather than recomputing analysis.

> **Split first, fit second.** Supervised preprocessing is never fit on the full dataset. Imputation, outlier bounds, scaling, and categorical vocabularies live inside a single scikit-learn `Pipeline` with the estimator and are fit only on training rows / inside each CV fold.

## Project overview

CRISP-DM Studio accepts an arbitrary CSV, validates it, profiles its structure, records preparation decisions, runs adaptive EDA, attempts clustering only when meaningful, lets the user explicitly confirm a supervised target, compares baseline and candidate models, evaluates the fitted pipelines, and produces a canonical report in HTML and PDF.

The application is deliberately conservative. Unsupported analyses are recorded as explicit skips; weak clustering is rejected; supervised modeling is never auto-started without an explicitly confirmed target; and evaluation can conclude that no candidate model is defensible.

## CRISP-DM approach and the 10 implementation phases

| Build phase | CRISP-DM responsibility | What the application does |
| --- | --- | --- |
| 1. Foundation & ingestion | Data acquisition / initial understanding | Validates CSVs, encoding, delimiter, dimensions, headers, and safety limits. |
| 2. Business & Data Understanding | Business Understanding + Data Understanding | Profiles columns, types, missingness, cardinality, identifiers, leakage risks, target candidates, quality warnings, and separates known facts from inferred objectives. |
| 3. Data Preparation | Data Preparation | Applies only safe structural cleaning globally; stores learned transformations as an unfitted preprocessing specification. |
| 4. EDA | Data Understanding | Computes adaptive descriptive statistics, relationships, missingness, outlier diagnostics, and reusable charts/findings. |
| 5. Clustering | Modeling (unsupervised) | Selects usable features, performs unsupervised preprocessing, searches K-Means candidates, applies a meaningfulness gate, profiles accepted segments, and otherwise records a skip. |
| 6. Modeling + baseline | Modeling | Requires an explicit target, splits first, fits preprocessing + estimator pipelines only on training data, compares Dummy + candidate models, and runs leakage-safe CV. |
| 7. Evaluation + recommendation | Evaluation | Quantifies improvement over baseline, stability, divergence, imbalance, complexity, held-out permutation importance, errors, and produces a structured recommendation. |
| 8. HTML + PDF reporting | Deployment / communication | Reuses all stored results and chart objects in one canonical Jinja report, exported to standalone interactive HTML and robust PDF. |
| 9. Hardening | Deployment readiness | Adds hostile-input integration tests, resource caps, failure isolation, progress states, stale-state invalidation, and page-level safety boundaries. |
| 10. Final QA & release | Deployment readiness | Final tests, documentation, demos, release packaging, and startup verification. |

## Architecture highlights

### Analyze once / render many ways

```text
CSV bytes
   ↓
IngestedDataset
   ↓
UnderstandingResult
   ↓
PreparationResult
   ├───────────────┐
   ↓               ↓
EDAResult      ClusteringResult
   │               │
   └──────┬────────┘
          ↓
     ModelingResult
          ↓
     EvaluationResult
          ↓
       ReportContext
       ┌─────┴─────┐
       ↓           ↓
 Streamlit UI   Jinja report
                   ├── interactive HTML
                   └── static/fallback PDF
```

`EDAChart`, `ClusteringChart`, and `EvaluationChart` retain their Plotly `Figure` objects. The UI renders them interactively; the report exporter serializes those same objects. Findings, preparation decisions, segment descriptions, metrics, and recommendations are likewise stored once and reused.

### Split-first / leakage-safe supervised learning

```text
Structurally cleaned dataframe
          ↓
     confirm target
          ↓
 remove rows missing target
          ↓
  TRAIN / TEST SPLIT FIRST
          ↓
 fresh copy of unfitted Phase 3 preprocessing spec
          ↓
┌─────────────────────────────────┐
│ sklearn Pipeline                │
│                                 │
│ imputation                      │
│ → IQR bounds                    │
│ → scaling / categorical encode  │
│ → estimator                     │
└─────────────────────────────────┘
          ↓
.fit(X_train, y_train) only
          ↓
CV clones/refits entire pipeline inside each fold
```

The original Phase 3 transformer is never fitted. The smoke test checks this invariant after preparation, clustering, modeling, evaluation, and reporting.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed data flow and leakage boundary.

## Features

- Drag-and-drop / file-picker CSV upload.
- Friendly validation for empty files, header-only files, type/extension mismatches, encoding, delimiter/readability, duplicate/blank headers, and configured file/row/column limits.
- Dataset profiling with inferred semantic feature types, missingness, cardinality, uniqueness, constants, near-constants, numeric/categorical distributions, possible timestamps, identifiers, leakage review signals, and target candidates.
- Honest generated objective split into **KNOWN** facts and **INFERRED** analytical objectives.
- Auditable preparation ledger: what changed → why → rows/columns affected.
- Explicit distinction between structural cleaning applied now and learned preprocessing deferred to supervised training.
- Adaptive Plotly EDA for numeric-only, categorical-only, and mixed datasets.
- Missingness, outlier diagnostics, Pearson correlations, numeric↔categorical effect sizes, and categorical associations.
- K-Means search with silhouette, inertia, Calinski-Harabasz, Davies-Bouldin, PCA display, cluster profiles, and a meaningfulness gate.
- Explicit target confirmation; nothing is auto-selected.
- Classification: DummyClassifier, Logistic Regression, Random Forest, HistGradientBoosting.
- Regression: DummyRegressor, Ridge, Random Forest, HistGradientBoosting.
- Holdout metrics plus leakage-safe cross-validation mean ± standard deviation.
- Evaluation that considers baseline improvement, CV stability, train/test divergence, class imbalance, and model complexity.
- Held-out permutation importance and confusion-matrix / residual diagnostics.
- Structured recommendation: what was learned, reliability, reasonable action, limitations, and next steps.
- Canonical CRISP-DM report with standalone interactive HTML and robust PDF fallback.
- Deterministic resource caps and progress states for expensive operations.
- Page-level and final app-level exception boundaries with stale-state invalidation.

## Tech stack and rationale

- **Python 3.11+** — one language across data engineering, ML, testing, and UI.
- **Streamlit** — fast local demo, drag/drop uploads, stateful multipage workflow, and download controls without a separate frontend build.
- **pandas / NumPy / SciPy / statsmodels** — tabular processing and statistical analysis.
- **scikit-learn** — composable `Pipeline` / `ColumnTransformer`, baselines, clustering, supervised estimators, CV, metrics, and permutation importance.
- **Plotly** — interactive browser charts and a shared figure representation that can also be exported statically.
- **Jinja2** — one canonical report template.
- **WeasyPrint** — styled HTML → PDF when available.
- **Kaleido** — Plotly static image export for full-fidelity PDFs when Chrome/Chromium is available.
- **pytest** — unit, integration, hostile-input, leakage, and reporting tests.
- **Ruff / Black** — pinned developer tooling for linting/formatting.

The project intentionally favors a pure-Python analytical architecture over React/FastAPI for this assignment because local setup, iteration speed, and ML integration matter more than frontend framework flexibility.

## Windows setup — fresh machine

### Prerequisites

- Windows 10/11.
- Python **3.11, 3.12, 3.13, or 3.14** available through the `py` launcher.
- Internet access for the first dependency installation.
- Optional: Chrome/Chromium for full-fidelity PDF charts.

### From ZIP to running app

Unzip the release, open **PowerShell** inside the `crispdm-studio` folder, then run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
streamlit run app.py
```

After installation, the single documented application run command is:

```powershell
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` automatically.

If PowerShell blocks virtual-environment activation, you can either allow locally signed scripts for your user or use `.venv\Scripts\python.exe -m streamlit run app.py` without activating the environment.

## How to use the app end to end

1. **Upload** — choose a CSV and confirm validation + dataset overview.
2. **Understanding** — inspect known facts, inferred objective, column profiles, identifier/leakage flags, target candidates, and quality warnings.
3. **Preparation** — review applied-now structural cleaning versus deferred learned transformations.
4. **EDA** — inspect computed findings, adaptive interactive charts, and explicit skip reasons for unsupported analyses.
5. **Clustering** — inspect candidate `k` diagnostics, accepted segments, or the honest skip reason.
6. **Modeling** — explicitly select a target. The selector defaults to nothing. Confirm task type, baseline/model metrics, CV mean ± std, and divergence flags.
7. **Evaluation** — review reasoned model selection, baseline improvement, stability, imbalance, held-out permutation importance, error diagnostic, and recommendation.
8. **Report** — download the standalone interactive HTML report and the PDF report.

Changing or failing a new upload invalidates all previous downstream results so stale analyses cannot remain visible.

## Testing and hardening

Run the full automated suite:

```powershell
pytest
```

Run the end-to-end smoke test:

```powershell
python scripts/smoke_test.py
```

Generate the hostile-input hardening matrix:

```powershell
python scripts/hardening_matrix.py
```

Run lint locally after installing `[dev]` dependencies:

```powershell
python -m ruff check src tests app.py scripts
```

Optional formatter check:

```powershell
python -m black --check src tests app.py scripts
```

The release suite contains 71 automated tests spanning ingestion, profiling, preparation, EDA, clustering, modeling, evaluation, reporting, leakage guarantees, and cross-phase hostile-input paths.

## Static PDF dependency note

The **HTML report always works without Chrome, Kaleido runtime support, or WeasyPrint system libraries** because it is rendered as standalone HTML with embedded Plotly JavaScript.

For a full-fidelity PDF with static chart images:

1. Install Google Chrome or Chromium.
2. If Kaleido cannot locate Chrome, run:

```powershell
plotly_get_chrome
```

3. Restart Streamlit and regenerate the PDF.

If static chart rendering is unavailable, the app still creates a styled PDF with clearly labeled chart placeholders. If WeasyPrint itself cannot render, the app falls back to a valid text-only PDF and displays installation guidance rather than crashing.

## Bundled demo datasets

| Dataset | Purpose | Suggested target |
| --- | --- | --- |
| `demo_data/classification_churn.csv` | Clean mixed-type classification demo with missing numeric values, a useful signal, EDA, clustering potential, and supervised modeling. | `churn` |
| `demo_data/regression_housing.csv` | Continuous-target regression demo with numeric + categorical predictors. | `price` |
| `demo_data/skip_no_features.csv` | Intentionally leaves no usable analytical features after structural cleaning; demonstrates honest EDA/clustering/model skip behavior. | None |

See [demo_data/README.md](demo_data/README.md) and [DEMO_CHECKLIST.md](DEMO_CHECKLIST.md).

## Project structure

```text
crispdm-studio/
├── app.py
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md
├── LIMITATIONS.md
├── HARDENING_MATRIX.md
├── DEMO_CHECKLIST.md
├── demo_data/
│   ├── README.md
│   ├── classification_churn.csv
│   ├── regression_housing.csv
│   └── skip_no_features.csv
├── docs/
│   └── screenshots/
│       └── README.md
├── assets/
│   └── report.css
├── templates/
│   └── report.html.j2
├── scripts/
│   ├── smoke_test.py
│   └── hardening_matrix.py
├── src/
│   └── crispdm_studio/
│       ├── ingestion/
│       ├── understanding/
│       ├── preparation/
│       ├── eda/
│       ├── clustering/
│       ├── modeling/
│       ├── evaluation/
│       ├── reporting/
│       └── ui/
└── tests/
    ├── fixtures/
    └── test_*.py
```

## Known limitations

This is a generic analytical studio, not a substitute for domain knowledge. Important limitations include heuristic target/type inference, no causal interpretation, K-Means geometry assumptions, bounded generic feature engineering for text/time fields, dataset-size/resource caps, and the fact that a statistically good model is not automatically a safe business decision. Full details are in [LIMITATIONS.md](LIMITATIONS.md).

## Screenshot placeholders

Before publishing/submitting, capture these views and place them under `docs/screenshots/`:

| Placeholder | Suggested capture |
| --- | --- |
| `01-upload.png` | Successful CSV upload and dataset overview |
| `02-understanding.png` | KNOWN vs INFERRED objective + profiles |
| `03-preparation.png` | Structural vs deferred transformation ledger |
| `04-eda.png` | Interactive distributions / relationship chart |
| `05-clustering.png` | k diagnostics + PCA scatter or skip state |
| `06-modeling.png` | Explicit target + baseline/model comparison |
| `07-evaluation.png` | Selected/no-defensible model + diagnostics |
| `08-report.png` | HTML/PDF download controls + PDF fidelity notice |

The repository intentionally includes placeholders rather than fabricated screenshots; capture them from the final app on your own machine.

## Submission note

This repository is organized as **Assignment 1 — Part 1** and is intended to be directly runnable from a clean checkout/ZIP after dependency installation. The application code lives under `src/crispdm_studio/`; Streamlit pages are presentation-only wrappers over reusable engines.
