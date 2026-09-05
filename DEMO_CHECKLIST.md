# Demo Checklist — Assignment 1, Part 2

Use this as the camera click-path for the final 10K release artifacts.

## Before recording

1. Extract the release ZIP.
2. Start FastAPI on port 8006.
3. Start React/Vite on port 5179.
4. Open `http://127.0.0.1:5179`.
5. Confirm the header says the production model is online and health reports a 10,000-row artifact set.

## 1. Headline architecture

Say: "This replicates the original anomaly-detection project, but I corrected four important shortcuts: the data is described honestly as locally synthesized, folder names match the actual code, live scoring runs the fitted RobustScaler plus Isolation Forest, and AutoResearch/retrain are real measured experiments."

## 2. Threat Scorer

1. Open **Threat Scorer**.
2. Point out the ten controls are generated from `/api/metadata`, including units, bounds and training-derived defaults.
3. Score the default vector.
4. Point to the **model threat index** and verdict.
5. Point separately to **Why it looks unusual** and say the IQR deviations explain unusual raw dimensions but do not produce the model verdict.
6. Move traffic, CPU, latency, request velocity and error rate toward high values and score again.

## 3. PCA Manifold

1. Click **PCA Manifold**.
2. Show the stored 2-D projection and hover points.
3. Explain PCA was fitted using the training preprocessing path and these coordinates are served from stored artifacts, not recomputed in React.

## 4. Model Tournament

1. Click **Model Tournament**.
2. Show all five detectors.
3. State the visible numbers are the final 10K held-out measurements.
4. Note that the synthetic benchmark is deliberately separable and that LOF performs much worse at 10K, demonstrating why the project reports measured results rather than assuming one algorithm always wins.

## 5. AutoResearch

1. Click **AutoResearch**.
2. Point out the **Real measured search runs** label.
3. Show the 12 rows and selected configuration.
4. Explain production selection prioritizes validation PR-AUC, then F1 and ROC-AUC.

## 6. Anomaly Explorer

1. Click **Anomalies**.
2. Show rank, row ID, threat score, raw model score and synthetic ground-truth archetype.
3. Explain ground truth is present for evaluation because anomalies were injected; it is not used to fit the detector.

## 7. CRISP-DM

1. Click **CRISP-DM**.
2. Walk through Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.
3. Highlight the two honest statements: data is locally synthesized, and high scores on separable synthetic anomalies are not production accuracy claims.

## 8. Real retrain

1. Click **Retrain**.
2. For demo speed, select 1,000–2,000 rows.
3. Start retraining.
4. Point out the disabled button/progress state.
5. When complete, show measured selected metrics/config.
6. Return to AutoResearch/health and show values refreshed from the newly activated artifact set.
7. Explain activation is temporary-write → validate → swap, with rollback protection; a second concurrent retrain is rejected with HTTP 409.

## 9. Close

Conclude: "The key improvements are integrity, not just UI polish: split-first preprocessing, evaluation-only labels, exact offline/live model equivalence, measured search history, and real atomic retraining."

## Restore final 10K artifacts after a demo retrain

A browser demo retrain intentionally replaces the live artifacts with a smaller run. Before committing/submitting again, restore the final artifacts with:

```powershell
python -m ml.train --rows 10000 --seed 42 --artifact-dir server/artifacts
```
