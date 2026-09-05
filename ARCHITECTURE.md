# Architecture — Assignment 1, Part 2

## System overview

```text
Locally synthesized 10-D telemetry
        │
        ├── features X ───────────────────────────────┐
        │                                             │
        └── ground truth labels                      │
             is_anomaly / archetype                  │
             (evaluation only)                       │
                                                      ▼
                                               SPLIT FIRST
                                               /          \
                                        X_train          X_valid
                                           │                │
                                           ▼                │
                                Fresh RobustScaler.fit      │
                                  TRAINING FEATURES ONLY    │
                                           │                │
                                           ▼                ▼
                                    transform train    transform valid
                                           │                │
                                           ▼                ▼
                                    detector.fit(X)     anomaly scores
                                                            │
                                                            ▼
                                                labels first used here
                                                            │
                                                            ▼
                                                     validation metrics
```

No fitting method receives `is_anomaly` or `archetype`.

## Offline training and artifacts

```text
ml.train
  │
  ├── deterministic data generator
  ├── feature-only split
  ├── train-only RobustScaler
  ├── five detector benchmarks
  ├── 12-run measured Isolation Forest search
  ├── selected production Isolation Forest
  ├── training-score percentile calibration
  ├── train-fitted PCA projection
  └── top anomaly extraction
        │
        ▼
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

The artifact manifest is the compatibility contract: schema version, ordered feature names, feature metadata, dataset/split metadata, selected hyperparameters and artifact filenames.

## Live scoring

```text
React :5179
   │ POST /api/anomaly/score
   ▼
FastAPI :8006
   │
   ├── validate telemetry request
   ├── reorder according to manifest.feature_order
   ├── persisted RobustScaler.transform
   ├── persisted IsolationForest decision_function
   ├── convert to higher-is-more-anomalous raw score
   ├── persisted empirical percentile calibration → 0–100
   └── threshold verdict

Separate branch:
raw input + scaler center/scale → robust-IQR deviations → explanation only
```

The IQR explanation never changes the detector score or verdict.

## Artifact lifecycle and real retraining

```text
POST /api/retrain
      │
      ├── acquire single-retrain lock
      ├── real bounded training/search
      ├── write NEW artifacts to temporary directory
      ├── validate schema/files/feature compatibility
      │
      ├── invalid ──► discard temporary set; live model untouched
      │
      └── valid
             │
             ▼
        swap live directory
             │
             ▼
        reload ModelRuntime
             │
             └── rollback old directory if activation fails
```

The frontend refreshes health, metadata, benchmarks, manifold, search history and top anomalies after successful activation.

## Runtime readiness

FastAPI does not crash when artifacts are absent or incompatible. `/api/health` reports `ml_ready:false` plus `ml_reason`. Analytical views display that reason, while retrain controls remain available via `/api/retrain/options` so the user can recover.

## Ports

- FastAPI: `127.0.0.1:8006`
- React/Vite: `127.0.0.1:5179`
- Vite proxies `/api/*` to FastAPI.
