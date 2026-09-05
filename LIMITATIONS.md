# Known Limitations

CRISP-DM Studio is intentionally dataset-agnostic, so it makes conservative generic choices. These limitations should be read alongside any result or generated report.

## Business context is not contained in a CSV

The application can describe observable properties of a file, but it cannot infer the true business decision, costs, causal mechanism, data collection process, or deployment environment. Business objectives generated from the file are explicitly marked **INFERRED**, not known facts.

## Type and target inference are heuristic

Semantic feature types, identifier detection, leakage review signals, and target candidates are based on observed values and naming/structure heuristics. A domain expert can know that a column is valid/invalid in ways the CSV alone cannot establish. The app therefore never auto-confirms a supervised target.

## Leakage warnings are review signals

Name/uniqueness heuristics can identify suspicious columns but cannot prove temporal or operational leakage without knowing when a feature becomes available in the real process.

## Generic preprocessing is deliberately bounded

The baseline supervised transformer supports numeric and categorical predictors. Free text and rich datetime feature engineering are not generically invented. Depending on the domain, specialized NLP, time-series, geospatial, image, or hierarchical methods may be more appropriate.

## Outlier handling is generic

The IQR clipper is a generic robust baseline. Extreme values may be valid and important. Domain-specific transformations should replace it when appropriate.

## Clustering assumptions

K-Means favors approximately convex/spherical groups under the chosen representation and Euclidean geometry. A strong silhouette does not prove business relevance. A weak solution is skipped, but an accepted solution still requires domain validation.

## Supervised model set is intentionally small

The project compares a defensible baseline with a compact group of common models; it is not an AutoML system and does not perform extensive hyperparameter optimization. More sophisticated tuning must remain leakage-safe and should be justified by the problem.

## Metrics do not encode business cost

Classification and regression metrics are generic. Thresholding, asymmetric error costs, fairness, calibration, expected value, and operational constraints require additional domain-specific evaluation.

## Small samples remain uncertain

Cross-validation is used where feasible, but tiny datasets can still produce unstable estimates. The evaluation engine surfaces small holdouts/CV constraints rather than treating point estimates as definitive.

## Resource caps trade completeness for responsiveness

Very large inputs are bounded for profiling/EDA/clustering/modeling/permutation importance. Deterministic sampling protects local demo reliability but can miss rare patterns. Resource notes are surfaced when caps apply.

## PDF fidelity depends on optional local runtimes

Standalone HTML reporting has no Chrome requirement. Full-fidelity PDF chart rendering needs Kaleido plus Chrome/Chromium, and styled HTML→PDF uses WeasyPrint. Missing optional prerequisites trigger chart placeholders or text-only fallback instead of a crash.

## No persistence / authentication / multi-user backend

This assignment is a local Streamlit application. Results live in the active Streamlit session and are not a secure multi-user datastore. It is not designed for sensitive production data without additional security, access control, audit, retention, and deployment work.

## No causal claims

Correlations, associations, feature importance, clusters, and predictive performance are descriptive/predictive evidence. They do not establish causal effects.
