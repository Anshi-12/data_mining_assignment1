# Limitations — Assignment 1, Part 2

## Synthetic data

The project intentionally reproduces the reference code's locally generated telemetry. It is not a benchmark on a downloaded Kaggle production dataset. Synthetic data is useful for deterministic testing and known ground truth, but it cannot reproduce all drift, dependency structure, attack diversity, missingness, instrumentation failures, seasonality or adversarial behavior found in real infrastructure telemetry.

## Separable anomaly archetypes

Several injected anomaly archetypes are deliberately far from normal operating behavior. As a result, some validation ROC-AUC/PR-AUC values are extremely high. These numbers demonstrate correct mechanics on this synthetic benchmark; they must not be interpreted as expected production security performance.

The final 10K release illustrates this sensitivity: Robust Mahalanobis achieves perfect validation ranking while Local Outlier Factor performs poorly. Model ranking can change substantially when density, sample size or anomaly geometry changes.

## Ground-truth labels are synthetic evaluation aids

`is_anomaly` and `archetype` are known because the generator injects them. They are used only for held-out evaluation/visualization and never for unsupervised fitting. In a real deployment, comparable labels may be missing, delayed or noisy.

## Single live production detector

Five detectors are benchmarked, but live inference intentionally deploys one production model: Isolation Forest. Robust Mahalanobis may outperform it on this specific synthetic release benchmark, but Isolation Forest is retained for parity with the reference project's production direction and because Improvement B is implemented as an Isolation Forest search/retrain lifecycle.

## Calibration is empirical, not probabilistic

The 0–100 threat index is the empirical percentile of the raw model anomaly score relative to training scores. It is not a calibrated probability that an event is malicious. The threshold reflects the selected Isolation Forest contamination setting.

## Explanation is diagnostic, not causal

Per-feature robust-IQR deviation describes which raw inputs are far from their training centers. It does not decompose the Isolation Forest prediction, prove causality, or identify a true attack root cause.

## Demo retraining caps

The interactive `/api/retrain` endpoint is deliberately bounded to 500–5,000 rows to keep a browser demo responsive and protect local compute. The committed final artifacts are produced offline with the 10,000-row CLI training command.

## Search scope

The real AutoResearch replacement is intentionally bounded to 12 Isolation Forest configurations. It proves a measured experiment loop rather than claiming exhaustive optimization.

## PCA visualization

PCA is a linear two-dimensional projection. Separation or overlap in the plot is descriptive and can omit structure present in the original ten-dimensional space.

## Local demonstration deployment

The application is designed for coursework/local demonstration. It does not include authentication, authorization, distributed job execution, a model registry, production observability, durable retraining queues, TLS termination or horizontal scaling.
