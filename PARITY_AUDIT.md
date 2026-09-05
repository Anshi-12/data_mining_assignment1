# Parity Audit — Assignment 1, Part 2

This document maps the original `dlmastery/data_science_examples/06_anomaly_detection` project to this replication. The goal was not a pixel-for-pixel clone: preserve the original analytical/product intent, correct documented inconsistencies, and replace demonstration-only shortcuts with measured ML behavior.

Original source: <https://github.com/dlmastery/data_science_examples/tree/main/06_anomaly_detection>

## Major capability mapping

| Original element | Our implementation | Status | Notes |
|---|---|---|---|
| Synthetic 10-D server telemetry | `ml/data_loader.py` deterministic local generator | Replicated + corrected docs | 10,000 rows / 3.5% contamination default; no Kaggle download claim. |
| Four injected anomaly archetypes | DDoS, credential infiltration, resource exhaustion, CPU↔disk correlation breakdown | Replicated | Ground truth is structurally separate from detector fitting features. |
| Isolation Forest | scikit-learn IsolationForest | Replicated + improved | Production config selected by a real measured 12-run search. |
| Local Outlier Factor | scikit-learn LocalOutlierFactor with novelty scoring | Replicated | Benchmarked on the held-out validation split. |
| One-Class SVM | scikit-learn OneClassSVM | Replicated | Unsupervised fit; labels only used after scoring. |
| Robust Mahalanobis | EllipticEnvelope | Replicated | Strongest benchmark on the final separable 10K synthetic run. |
| Deep/tabular autoencoder | hand-written NumPy autoencoder | Replicated | Reconstruction error used as anomaly score. |
| ROC-AUC | stored validation metric | Replicated | Computed from held-out labels only. |
| PR-AUC | stored validation metric | Replicated | Primary rare-event search-selection metric. |
| Precision / Recall / F1 | stored validation metrics | Replicated | Stored once in `benchmarks.json` / `search_history.json`. |
| Runtime comparison | fit time + inference ms/row | Replicated | Measured, not hardcoded. |
| PCA manifold | train-fitted PCA projected onto all rows | Replicated | Stored in `manifold.json`; frontend does not refit PCA. |
| IQR attribution | robust per-feature deviation panel | Replicated + clarified | Explanation only; it does not generate the model verdict. |
| Live threat scorer | React scorer → FastAPI `/api/anomaly/score` | Improved | Uses the exact persisted RobustScaler + IsolationForest + calibration from offline training. |
| Model tournament | React view backed by `/api/benchmarks` | Replicated | All displayed metrics come from stored artifacts. |
| AutoResearch leaderboard | React view backed by `/api/autoresearch/history` | Improved | Every row is a genuine measured Isolation Forest search run. |
| Anomaly explorer | React table backed by `/api/anomalies/top` | Replicated | Uses stored production rankings. |
| CRISP-DM report/view | analyst-facing six-phase view | Replicated + corrected | States local synthesis and synthetic-separability limitation explicitly. |
| Retrain controls | React → `/api/retrain` | Improved | Actually retrains, recalibrates, validates and atomically activates artifacts. |
| FastAPI backend on 8006 | `server/` FastAPI | Replicated | Clean artifact readiness state and validation added. |
| React/Vite frontend on 5179 | `client/` React 18 + Vite | Replicated | `/api` proxy targets `127.0.0.1:8006`. |

## CRISP-DM parity

| CRISP-DM phase | Replication |
|---|---|
| Business Understanding | Detect abnormal telemetry patterns representing security/operational threats while controlling false positives. |
| Data Understanding | Ten telemetry dimensions, reproducible local synthesis, 3.5% injected anomalies, four known archetypes. |
| Data Preparation | Split first; fit a fresh RobustScaler on training features only. Labels never enter preprocessing or detector fitting. |
| Modeling | Five unsupervised detector families plus a bounded measured Isolation Forest search. |
| Evaluation | Held-out ROC-AUC, PR-AUC, precision, recall, F1 and runtime metrics; synthetic labels are evaluation-only. |
| Deployment | Persisted versioned artifacts, FastAPI runtime, live React dashboard, real retraining with atomic activation. |

## Explicit corrections and improvements

| Topic | Original → Our version |
|---|---|
| Folder names | README says `backend/` + `frontend/` → actual and documented folders are consistently `server/` + `client/`. |
| Dataset description | README calls it Kaggle server telemetry → code and docs state the truth: telemetry is synthesized locally and reproducibly. |
| Live scoring | IQR-based heuristic threat scoring → exact persisted `RobustScaler.transform` + `IsolationForest` score + persisted calibration. |
| IQR diagnostics | Heuristic score and explanation can be conflated → IQR deviations are a separate explanation field and UI panel only. |
| AutoResearch | predefined/static leaderboard values → 12 genuine measured hyperparameter fits stored in `search_history.json`. |
| Retraining | threshold update + hardcoded estimated metrics → actual regenerate/split/scale/search/refit/calibrate pipeline with measured metrics. |
| Artifact activation | direct/demo-style state update → complete temporary artifact set is validated then atomically swapped with rollback protection. |
| Artifact compatibility | implicit assumptions → manifest schema version + exact feature order + metadata validation gate. |

## Improvement A proof — live API uses the fitted model

The automated API test computes a score in two independent ways for the same fixed raw telemetry vector:

1. Phase 3 offline round trip loads `robust_scaler.joblib`, `isolation_forest.joblib`, and `score_calibration.json` and calculates the raw anomaly score + calibrated threat index.
2. Phase 4 sends that same raw vector to `POST /api/anomaly/score`.

The test asserts equality to `1e-12` tolerance for both the raw anomaly score and threat index. This is direct evidence that the API runs the persisted production model path, not an IQR scoring heuristic.

## Improvement B proof — measured search and real retraining

`search_history.json` in the final 10K artifact set contains exactly 12 candidate runs. Every row carries `"measured": true` and real validation metrics/timings from an actual fit. The selected production configuration is one of those measured rows.

`POST /api/retrain` executes the real training pipeline on a bounded demo-sized dataset, writes a complete new artifact set into a temporary directory, validates it, then swaps it into the live artifact location. A concurrency lock rejects a second simultaneous retrain with HTTP 409. Tests also remove a required artifact from a prepared replacement and prove that the invalid set is rejected without corrupting the live model.

## Final 10K release result

The committed artifacts were regenerated with:

```powershell
python -m ml.train --rows 10000 --seed 42 --artifact-dir server/artifacts
```

The final split is 7,500 training rows / 2,500 validation rows. See the project README for the measured benchmark table.
