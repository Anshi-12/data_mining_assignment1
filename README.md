# Assignment 1 — Part 2

## Anomaly Detection Studio — replication + improvements

This project replicates and improves the `06_anomaly_detection` example from [`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples/tree/main/06_anomaly_detection). The original project demonstrates full-stack anomaly detection for high-dimensional server telemetry. This version preserves that core idea while correcting documentation inconsistencies and replacing demonstration-only shortcuts with leakage-safe training, real persisted-model inference, measured hyperparameter search, and genuine retraining.

> **Source credit:** original concept and reference implementation by `dlmastery/data_science_examples/06_anomaly_detection`. This repository is a coursework replication/improvement, not the original project.

## What the application does

The system synthesizes reproducible 10-dimensional server telemetry, injects four known anomaly archetypes, benchmarks five unsupervised anomaly detectors, selects an Isolation Forest production configuration through a real measured search, persists the fitted production artifacts, exposes them through FastAPI, and visualizes them in a React analyst dashboard.

The ten telemetry dimensions are:

- inbound network bytes/sec
- outbound network bytes/sec
- CPU utilization
- memory pressure
- latency
- error rate
- request velocity
- authentication failures
- entropy score
- disk IOPS

The four injected anomaly archetypes are volumetric DDoS, credential-stuffing/infiltration, resource exhaustion/memory leak, and CPU-vs-disk subspace correlation breakdown.

The dataset is **locally synthesized** by `ml/data_loader.py`. Nothing is downloaded from Kaggle.

## Headline engineering decisions

### 1. Split first / leakage-safe

```text
10-D feature matrix X
        │
        ▼
   SPLIT FIRST
   /         \
train       validation
  │              │
  ▼              │
fit NEW           │
RobustScaler      │
TRAIN ONLY        │
  │              │
  ├── detector.fit(X_train_scaled)
  │
  └──────────────► scaler.transform(X_valid)
                         │
                         ▼
                   anomaly scores
                         │
                         ▼
                 use is_anomaly ONLY HERE
                   for validation metrics
```

`is_anomaly` and `archetype` are structurally separate evaluation labels. They never enter the scaler or unsupervised detector fitting.

### 2. Live scoring uses the actual fitted model

The original demo's live scoring logic was separate from its trained Isolation Forest path. This project fixes that.

```text
incoming telemetry
      ↓
manifest feature validation/order
      ↓
robust_scaler.joblib.transform
      ↓
isolation_forest.joblib
      ↓
raw anomaly score
      ↓
score_calibration.json
      ↓
0–100 threat index + threshold verdict
```

The robust-IQR per-feature deviations are returned as a **separate explanation**. They do not generate or override the model verdict.

### 3. Real measured AutoResearch + real retraining

The leaderboard is not fabricated. `search_history.json` contains 12 actual Isolation Forest fits over a bounded grid of contamination, tree count and max-sample settings. Every stored row is marked `measured: true` and contains measured held-out metrics and timing.

`POST /api/retrain` genuinely regenerates data, splits, refits a new scaler, reruns the measured search, recalibrates scores, creates a complete temporary artifact set, validates it, and atomically replaces the live set. Failed activation rolls back; simultaneous retraining is rejected with HTTP 409.

## Architecture

```text
                 OFFLINE / RETRAINING

locally synthesized telemetry
        │
        ├── X: 10 feature columns
        └── y: evaluation-only labels
        │
        ▼
 split-first training pipeline
        │
        ├── five detector benchmark
        ├── 12-run measured IF search
        ├── training-only score calibration
        ├── train-fitted PCA
        └── top anomaly extraction
        │
        ▼
server/artifacts/
        │
        ├── robust_scaler.joblib
        ├── isolation_forest.joblib
        ├── score_calibration.json
        ├── search_history.json
        ├── benchmarks.json
        ├── manifold.json
        ├── top_anomalies.json
        └── manifest.json

                   LIVE APPLICATION

React 18 + Vite :5179
        │
        │ /api proxy
        ▼
FastAPI :8006
        │
        ├── artifact validation/runtime
        ├── exact persisted-model scoring
        ├── stored artifact APIs
        └── real atomic retraining
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the detailed leakage boundary and artifact lifecycle.

## Tech stack

### Python / ML

- Python 3.11+
- NumPy / pandas for deterministic synthetic data and numerical work
- scikit-learn for RobustScaler, Isolation Forest, LOF, One-Class SVM, EllipticEnvelope, metrics and PCA
- joblib for exact fitted-object persistence
- a hand-written NumPy autoencoder for parity with the reference project's neural reconstruction detector without requiring a heavyweight deep-learning runtime

### API

- FastAPI on **8006**
- Pydantic request/response contracts
- explicit artifact manifest validation
- clean `ml_ready:false` degradation instead of startup crashes

### Frontend

- React 18
- Vite on **5179**
- Vite `/api` proxy → `http://127.0.0.1:8006`
- dependency-light SVG/CSS visualization

## Final committed 10,000-row benchmark

The release artifacts were generated with:

```powershell
python -m ml.train --rows 10000 --seed 42 --artifact-dir server/artifacts
```

Split: **7,500 training rows / 2,500 validation rows**.

| Detector | ROC-AUC | PR-AUC | Precision | Recall | F1 | Fit (s) | Inference ms/row |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.9999 | 0.9971 | 0.960 | 0.969 | 0.964 | 0.223 | 0.015 |
| Local Outlier Factor | 0.3088 | 0.0703 | 0.053 | 0.051 | 0.052 | 0.290 | 0.074 |
| One-Class SVM | 0.9286 | 0.8894 | 0.869 | 0.878 | 0.873 | 0.071 | 0.014 |
| Robust Mahalanobis | 1.0000 | 1.0000 | 0.970 | 1.000 | 0.985 | 1.024 | 0.000 |
| NumPy Autoencoder | 0.9995 | 0.9882 | 0.929 | 0.939 | 0.934 | 0.324 | 0.000 |

These are **10K release numbers**, not the earlier 2K development benchmark.

The production Isolation Forest search selected:

```json
{
  "contamination": 0.035,
  "n_estimators": 160,
  "max_samples": "auto"
}
```

Selection reason: highest measured validation PR-AUC (`0.9971`), with F1 (`0.9645`) and ROC-AUC (`0.9999`) as tie-breakers.

> Important: several injected anomalies are intentionally separable. These results validate the workflow on synthetic ground truth and should **not** be interpreted as expected real-world security performance.

## CRISP-DM mapping

| Phase | Implementation |
|---|---|
| **Business Understanding** | Detect abnormal server telemetry associated with security/operational threats while limiting false alarms. |
| **Data Understanding** | Deterministic local 10-D telemetry synthesis, normal operating distributions, 3.5% anomaly injection, four known archetypes. |
| **Data Preparation** | Feature/label separation, split first, fit a fresh RobustScaler on training features only. |
| **Modeling** | Isolation Forest, LOF, One-Class SVM, robust Mahalanobis, NumPy autoencoder, plus measured Isolation Forest search. |
| **Evaluation** | Held-out ROC-AUC, PR-AUC, precision, recall, F1, fit time and inference latency. Ground truth is used only here/for visualization. |
| **Deployment** | Versioned artifacts, FastAPI runtime, React analyst dashboard, exact fitted-model scoring and atomic real retraining. |

## Fresh Windows setup

### Prerequisites

- Python 3.11
- Node.js 18+ and npm
- Git optional

### 1. Extract and enter the project

```powershell
cd path\to\06_anomaly_detection
```

### 2. Create/install Python environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The release ZIP already contains the final 10K artifacts. You do not need to retrain before launching.

### Terminal 1 — FastAPI

```powershell
cd path\to\06_anomaly_detection
.venv\Scripts\Activate.ps1
python -m uvicorn server.main:app --host 127.0.0.1 --port 8006 --reload
```

Verify:

```text
http://127.0.0.1:8006/api/health
```

### Terminal 2 — React/Vite

```powershell
cd path\to\06_anomaly_detection\client
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5179
```

Optional frontend build check:

```powershell
npm run build
```

## Dashboard walkthrough

1. **Threat Scorer** — controls come from live `/api/metadata`; submit telemetry and inspect the real model threat index/verdict. Keep the separate IQR explanation distinction clear.
2. **PCA Manifold** — stored PCA coordinates from `/api/manifold`.
3. **Model Tournament** — final 10K metrics from `/api/benchmarks`.
4. **AutoResearch** — all 12 measured search runs from `/api/autoresearch/history`.
5. **Anomalies** — production top anomalies from `/api/anomalies/top`.
6. **CRISP-DM** — six-phase narrative, honest local-synthesis wording and synthetic-separability caveat.
7. **Retrain** — run a bounded 1K–2K demo retrain; on completion all dependent dashboard views refresh.

See [`DEMO_CHECKLIST.md`](DEMO_CHECKLIST.md) for a camera-ready version.

## API reference

### Health

```powershell
curl.exe http://127.0.0.1:8006/api/health
```

Expected shape:

```json
{
  "status": "ok",
  "service": "Anomaly Detection Studio",
  "phase": 6,
  "ml_ready": true,
  "ml_reason": null,
  "scorer": "persisted-robustscaler-isolationforest",
  "model_type": "IsolationForest",
  "selected_hyperparameters": {
    "contamination": 0.035,
    "n_estimators": 160,
    "max_samples": "auto"
  },
  "artifact_schema_version": "1.0"
}
```

### Feature/runtime metadata

```powershell
curl.exe http://127.0.0.1:8006/api/metadata
```

### Score one telemetry vector

```powershell
curl.exe -X POST http://127.0.0.1:8006/api/anomaly/score `
  -H "Content-Type: application/json" `
  -d "{\"network_bytes_in\":285886.73,\"network_bytes_out\":232866.03,\"cpu_utilization\":46.38,\"memory_pressure\":20.0,\"latency_ms\":32.56,\"error_rate\":0.000443,\"request_velocity\":137.38,\"auth_failures\":0,\"entropy_score\":0.6957,\"disk_iops\":481.28}"
```

Response fields include `raw_anomaly_score`, `threat_index`, `raw_threshold`, `threat_threshold`, `is_anomaly`, `verdict`, `model_type`, and the separately labeled `attribution` array.

### Stored benchmark/model artifacts

```powershell
curl.exe http://127.0.0.1:8006/api/benchmarks
curl.exe http://127.0.0.1:8006/api/manifold
curl.exe http://127.0.0.1:8006/api/anomalies/top
curl.exe http://127.0.0.1:8006/api/autoresearch/history
```

### Retrain options

```powershell
curl.exe http://127.0.0.1:8006/api/retrain/options
```

### Real demo retrain

```powershell
curl.exe -X POST http://127.0.0.1:8006/api/retrain `
  -H "Content-Type: application/json" `
  -d "{\"rows\":1000,\"contamination\":0.035,\"seed\":42,\"validation_fraction\":0.25}"
```

The interactive endpoint is capped at 5,000 rows for local/demo responsiveness. A second simultaneous request receives HTTP 409.

## Improvement proofs

### Improvement A — offline score equals API score

A regression test loads the exact persisted Phase 3 scaler/model/calibration and computes a fixed-vector score offline, then sends the same vector to `/api/anomaly/score`. Both the raw anomaly score and calibrated threat index must match to `1e-12` tolerance.

This is the evidence that live scoring uses the persisted fitted pipeline.

### Improvement B — measured search + real retraining

The final `search_history.json` contains exactly 12 real candidate fits, all with `"measured": true`. `/api/retrain` calls the real training/search pipeline, creates a complete temporary artifact directory, validates it, atomically activates it and reloads the live runtime. Tests cover successful replacement, rollback on invalid replacement, and the HTTP 409 concurrency guard.

See [`PARITY_AUDIT.md`](PARITY_AUDIT.md) for the original→replication mapping and before→after corrections.

## Testing

### Python/API tests

```powershell
pytest
```

Current release result:

```text
32 passed
```

### Re-run final 10K training

```powershell
python -m ml.train --rows 10000 --seed 42 --artifact-dir server/artifacts
```

### Phase 2 deterministic data validation

```powershell
python -m ml.generate --rows 2000 --seed 42 --output data/generated/telemetry_2000.csv
```

### Ruff

```powershell
python -m ruff check ml server tests
```

Ruff is included in the project's `[dev]` dependencies. The build container used for this release did not have Ruff installed globally, so no lint pass is claimed here.

### Frontend build

```powershell
cd client
npm install
npm run build
```

## Artifact set

```text
server/artifacts/
├── manifest.json
├── robust_scaler.joblib
├── isolation_forest.joblib
├── score_calibration.json
├── search_history.json
├── benchmarks.json
├── manifold.json
└── top_anomalies.json
```

The final committed manifest states:

- source: locally synthesized deterministic server telemetry
- dataset rows: 10,000
- training rows: 7,500
- validation rows: 2,500
- seed: 42
- scaler fit scope: training features only
- production model: IsolationForest

## Project structure

```text
06_anomaly_detection/
├── README.md
├── ARCHITECTURE.md
├── LIMITATIONS.md
├── PARITY_AUDIT.md
├── DEMO_CHECKLIST.md
├── pyproject.toml
├── ml/
├── server/
│   └── artifacts/
├── client/
├── data/
│   └── generated/
└── tests/
```

## Known limitations

The main limitations are synthetic/separable anomalies, synthetic ground truth, a single live production detector, empirical-not-probabilistic threat calibration, heuristic explanation rather than causal attribution, a bounded 12-run search and a 5,000-row interactive retrain cap. See [`LIMITATIONS.md`](LIMITATIONS.md) for details.

## Screenshots

Add final screenshots before publishing the submission:

- `[SCREENSHOT PLACEHOLDER]` Live Threat Scorer + separate attribution panel
- `[SCREENSHOT PLACEHOLDER]` 2D PCA Manifold
- `[SCREENSHOT PLACEHOLDER]` 10K Model Tournament
- `[SCREENSHOT PLACEHOLDER]` real AutoResearch leaderboard
- `[SCREENSHOT PLACEHOLDER]` Anomaly Explorer
- `[SCREENSHOT PLACEHOLDER]` CRISP-DM view
- `[SCREENSHOT PLACEHOLDER]` real Retrain controls/result

## Attribution

Replication source: <https://github.com/dlmastery/data_science_examples/tree/main/06_anomaly_detection>

This implementation intentionally uses `server/` and `client/` because those are the actual component names, and describes the data as locally synthesized because that is what the executable reference generator does.
