"""Phase 3 leakage-safe modeling, measured search, and artifact persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.data_loader import FEATURE_API_FIELDS, FEATURE_CATALOG, FEATURE_INPUT_BOUNDS, FEATURE_NAMES, TelemetryDataset, generate_anomaly_dataset
from ml.models import SklearnDetector, build_detectors, fit_and_time
from ml.preprocessing import build_robust_scaler
from server.artifact_schema import ArtifactLayout


@dataclass(frozen=True, slots=True)
class SplitMetadata:
    seed: int
    validation_fraction: float
    training_rows: int
    validation_rows: int
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DetectorMetrics:
    model_name: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    fit_seconds: float
    inference_ms_per_row: float


@dataclass(frozen=True, slots=True)
class SearchRun:
    run_id: int
    params: dict[str, Any]
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    fit_seconds: float
    inference_ms_per_row: float
    measured: bool = True


@dataclass(frozen=True, slots=True)
class Calibration:
    method: str
    percentiles: tuple[float, ...]
    score_quantiles: tuple[float, ...]
    raw_threshold: float
    threat_threshold: float
    contamination: float

    def threat_index(self, raw_scores: np.ndarray | float) -> np.ndarray:
        values = np.asarray(raw_scores, dtype=float)
        pct = np.interp(values, self.score_quantiles, self.percentiles, left=0.0, right=1.0)
        return np.clip(pct * 100.0, 0.0, 100.0)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    dataset_rows: int
    contamination: float
    split: SplitMetadata
    benchmark_results: tuple[DetectorMetrics, ...]
    search_history: tuple[SearchRun, ...]
    selected_params: dict[str, Any]
    selection_reason: str
    calibration: Calibration
    artifact_dir: str
    manifold_rows: int
    top_anomaly_rows: int


def split_features_only(
    features: pd.DataFrame,
    *,
    validation_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitMetadata]:
    """Split features deterministically without consulting ground-truth labels."""
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5.")
    n_rows = len(features)
    if n_rows < 20:
        raise ValueError("At least 20 rows are required for Phase 3 modeling.")
    rng = np.random.RandomState(random_state)
    order = rng.permutation(n_rows)
    n_validation = max(1, int(round(n_rows * validation_fraction)))
    validation_idx = order[:n_validation]
    training_idx = order[n_validation:]
    X_train = features.iloc[training_idx].copy()
    X_validation = features.iloc[validation_idx].copy()
    metadata = SplitMetadata(
        seed=random_state,
        validation_fraction=validation_fraction,
        training_rows=len(X_train),
        validation_rows=len(X_validation),
        training_indices=tuple(int(i) for i in training_idx),
        validation_indices=tuple(int(i) for i in validation_idx),
    )
    return X_train, X_validation, metadata


def _metrics(
    *, model_name: str, y_true: np.ndarray, scores: np.ndarray, predictions: np.ndarray,
    fit_seconds: float, inference_ms_per_row: float,
) -> DetectorMetrics:
    return DetectorMetrics(
        model_name=model_name,
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        fit_seconds=float(fit_seconds),
        inference_ms_per_row=float(inference_ms_per_row),
    )


def benchmark_detectors(
    X_train_scaled: np.ndarray,
    X_validation_scaled: np.ndarray,
    y_validation: np.ndarray,
    *, contamination: float,
    random_state: int,
) -> tuple[DetectorMetrics, ...]:
    results: list[DetectorMetrics] = []
    for name, detector in build_detectors(
        contamination=contamination,
        random_state=random_state,
        input_dim=X_train_scaled.shape[1],
    ).items():
        timed = fit_and_time(detector, X_train_scaled, X_validation_scaled)
        results.append(
            _metrics(
                model_name=name,
                y_true=y_validation,
                scores=timed.scores,
                predictions=timed.predictions,
                fit_seconds=timed.fit_seconds,
                inference_ms_per_row=timed.inference_ms_per_row,
            )
        )
    return tuple(results)


def isolation_forest_search_grid(base_contamination: float) -> tuple[dict[str, Any], ...]:
    """A bounded 9-run grid suitable for development compute."""
    contaminations = sorted({round(base_contamination * 0.8, 4), base_contamination, round(base_contamination * 1.2, 4)})
    configs: list[dict[str, Any]] = []
    for contamination in contaminations:
        for n_estimators in (100, 160):
            for max_samples in (0.75, "auto"):
                configs.append(
                    {
                        "contamination": float(contamination),
                        "n_estimators": n_estimators,
                        "max_samples": max_samples,
                    }
                )
    return tuple(configs)


def run_isolation_forest_search(
    X_train_scaled: np.ndarray,
    X_validation_scaled: np.ndarray,
    y_validation: np.ndarray,
    *, base_contamination: float,
    random_state: int,
) -> tuple[tuple[SearchRun, ...], dict[str, Any], str]:
    history: list[SearchRun] = []
    for run_id, params in enumerate(isolation_forest_search_grid(base_contamination), start=1):
        detector = SklearnDetector(
            IsolationForest(
                **params,
                random_state=random_state,
                n_jobs=-1,
            )
        )
        timed = fit_and_time(detector, X_train_scaled, X_validation_scaled)
        metric = _metrics(
            model_name="Isolation Forest Search",
            y_true=y_validation,
            scores=timed.scores,
            predictions=timed.predictions,
            fit_seconds=timed.fit_seconds,
            inference_ms_per_row=timed.inference_ms_per_row,
        )
        history.append(
            SearchRun(
                run_id=run_id,
                params=params,
                roc_auc=metric.roc_auc,
                pr_auc=metric.pr_auc,
                precision=metric.precision,
                recall=metric.recall,
                f1=metric.f1,
                fit_seconds=metric.fit_seconds,
                inference_ms_per_row=metric.inference_ms_per_row,
            )
        )

    # PR-AUC is primary because anomalies are rare; F1 is a thresholded tie-breaker,
    # then ROC-AUC. Lower latency wins only if the measured quality tuple is equal.
    best = max(
        history,
        key=lambda r: (r.pr_auc, r.f1, r.roc_auc, -r.inference_ms_per_row),
    )
    reason = (
        f"Run {best.run_id} selected from {len(history)} measured configurations: "
        f"highest validation PR-AUC ({best.pr_auc:.4f}); F1 ({best.f1:.4f}) and "
        f"ROC-AUC ({best.roc_auc:.4f}) were used as tie-breakers."
    )
    return tuple(history), dict(best.params), reason


def build_calibration(train_raw_scores: np.ndarray, contamination: float) -> Calibration:
    """Build empirical-CDF calibration using training scores only.

    A raw anomaly score is mapped to its empirical training-score percentile and then
    multiplied by 100. The anomaly threshold is the (1-contamination) training-score
    quantile, expressed both as a raw score and a 0-100 threat index.
    """
    percentiles = np.linspace(0.0, 1.0, 101)
    score_quantiles = np.quantile(train_raw_scores, percentiles)
    raw_threshold = float(np.quantile(train_raw_scores, 1.0 - contamination))
    threat_threshold = float(
        np.interp(raw_threshold, score_quantiles, percentiles, left=0.0, right=1.0) * 100.0
    )
    return Calibration(
        method="training_empirical_percentile",
        percentiles=tuple(float(x) for x in percentiles),
        score_quantiles=tuple(float(x) for x in score_quantiles),
        raw_threshold=raw_threshold,
        threat_threshold=threat_threshold,
        contamination=contamination,
    )


def _serialize_calibration(calibration: Calibration) -> dict[str, Any]:
    return {
        **asdict(calibration),
        "formula": "threat_index = 100 * empirical_percentile(raw_anomaly_score among training scores)",
        "threshold_formula": "raw_threshold = training_quantile(1 - contamination)",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_phase3(
    *,
    rows: int = 2_000,
    contamination: float = 0.035,
    seed: int = 42,
    validation_fraction: float = 0.25,
    artifact_dir: str | Path = "server/artifacts",
    top_n: int = 100,
) -> TrainingResult:
    dataset: TelemetryDataset = generate_anomaly_dataset(
        n_samples=rows,
        contamination=contamination,
        random_state=seed,
    )

    # CRITICAL: split the feature matrix before ANY learned preprocessing.
    X_train, X_validation, split = split_features_only(
        dataset.features,
        validation_fraction=validation_fraction,
        random_state=seed,
    )

    # A fresh scaler is fit ONLY on training features. No labels are accepted here.
    scaler = build_robust_scaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy(dtype=float))
    X_validation_scaled = scaler.transform(X_validation.to_numpy(dtype=float))

    # Ground truth is first read here, after all fitting inputs are already formed.
    y_validation = dataset.labels.iloc[list(split.validation_indices)]["is_anomaly"].to_numpy(dtype=int)
    if len(np.unique(y_validation)) < 2:
        raise RuntimeError("Validation split contains only one class; metrics are not defensible.")

    benchmark_results = benchmark_detectors(
        X_train_scaled,
        X_validation_scaled,
        y_validation,
        contamination=contamination,
        random_state=seed,
    )
    search_history, selected_params, selection_reason = run_isolation_forest_search(
        X_train_scaled,
        X_validation_scaled,
        y_validation,
        base_contamination=contamination,
        random_state=seed,
    )

    production_model = IsolationForest(
        **selected_params,
        random_state=seed,
        n_jobs=-1,
    )
    production_model.fit(X_train_scaled)
    train_raw_scores = -production_model.decision_function(X_train_scaled)
    production_contamination = float(selected_params["contamination"])
    calibration = build_calibration(train_raw_scores, production_contamination)

    # Visualization projection is also learned from training rows only, then applied
    # to every row through the already training-fitted scaler.
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(X_train_scaled)
    X_all_scaled = scaler.transform(dataset.features.to_numpy(dtype=float))
    pca_coords = pca.transform(X_all_scaled)
    all_raw_scores = -production_model.decision_function(X_all_scaled)
    all_threat = calibration.threat_index(all_raw_scores)

    layout = ArtifactLayout(Path(artifact_dir))
    layout.root.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, layout.scaler)
    joblib.dump(production_model, layout.detector)
    _write_json(layout.calibration, _serialize_calibration(calibration))
    _write_json(
        layout.search_history,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
            "validation_rows": split.validation_rows,
            "selection_metric": "validation_pr_auc",
            "runs": [asdict(run) for run in search_history],
            "selected_params": selected_params,
            "selection_reason": selection_reason,
        },
    )

    manifold = pd.DataFrame(
        {
            "row_id": np.arange(rows, dtype=int),
            "pc1": pca_coords[:, 0],
            "pc2": pca_coords[:, 1],
            "raw_anomaly_score": all_raw_scores,
            "threat_index": all_threat,
            "is_anomaly": dataset.labels["is_anomaly"].to_numpy(dtype=int),
            "archetype": dataset.labels["archetype"].astype(str).to_numpy(),
        }
    )
    manifold_path = layout.root / "manifold.json"
    _write_json(
        manifold_path,
        {
            "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "points": manifold.to_dict(orient="records"),
        },
    )

    top_n = min(top_n, rows)
    top_idx = np.argsort(-all_raw_scores)[:top_n]
    top_records: list[dict[str, Any]] = []
    combined = dataset.combined_frame()
    for rank, idx in enumerate(top_idx, start=1):
        record = {name: float(combined.iloc[idx][name]) for name in FEATURE_NAMES}
        record.update(
            {
                "rank": rank,
                "row_id": int(idx),
                "raw_anomaly_score": float(all_raw_scores[idx]),
                "threat_index": float(all_threat[idx]),
                "is_anomaly": int(combined.iloc[idx]["is_anomaly"]),
                "archetype": str(combined.iloc[idx]["archetype"]),
            }
        )
        top_records.append(record)
    _write_json(layout.root / "top_anomalies.json", {"rows": top_records})

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "locally synthesized deterministic server telemetry",
        "training_rows": split.training_rows,
        "validation_rows": split.validation_rows,
        "dataset_rows": rows,
        "seed": seed,
        "feature_order": list(FEATURE_NAMES),
        "feature_metadata": [
            {
                **next(item for item in FEATURE_CATALOG if item["feature"] == feature),
                **FEATURE_INPUT_BOUNDS[feature],
                "api_field": FEATURE_API_FIELDS[feature],
                "default": float(scaler.center_[index]),
            }
            for index, feature in enumerate(FEATURE_NAMES)
        ],
        "ground_truth_columns": ["is_anomaly", "archetype"],
        "ground_truth_usage": "validation metrics and visualization only; never detector fitting",
        "scaler_fit_scope": "training features only",
        "production_model": "IsolationForest",
        "selected_params": selected_params,
        "files": {
            "scaler": layout.scaler.name,
            "detector": layout.detector.name,
            "calibration": layout.calibration.name,
            "search_history": layout.search_history.name,
            "benchmarks": "benchmarks.json",
            "manifold": "manifold.json",
            "top_anomalies": "top_anomalies.json",
        },
    }
    _write_json(layout.manifest, manifest)

    _write_json(
        layout.root / "benchmarks.json",
        {
            "rows": rows,
            "validation_rows": split.validation_rows,
            "results": [asdict(result) for result in benchmark_results],
        },
    )

    return TrainingResult(
        dataset_rows=rows,
        contamination=contamination,
        split=split,
        benchmark_results=benchmark_results,
        search_history=search_history,
        selected_params=selected_params,
        selection_reason=selection_reason,
        calibration=calibration,
        artifact_dir=str(layout.root),
        manifold_rows=rows,
        top_anomaly_rows=top_n,
    )


def load_production_artifacts(artifact_dir: str | Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    layout = ArtifactLayout(Path(artifact_dir))
    scaler = joblib.load(layout.scaler)
    model = joblib.load(layout.detector)
    calibration = json.loads(layout.calibration.read_text(encoding="utf-8"))
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    return scaler, model, calibration, manifest


def score_fixed_vector(
    vector: np.ndarray,
    *, scaler: Any,
    model: Any,
    calibration_payload: dict[str, Any],
) -> tuple[float, float]:
    x = np.asarray(vector, dtype=float).reshape(1, -1)
    scaled = scaler.transform(x)
    raw = float(-model.decision_function(scaled)[0])
    threat = float(
        np.interp(
            raw,
            calibration_payload["score_quantiles"],
            calibration_payload["percentiles"],
            left=0.0,
            right=1.0,
        )
        * 100.0
    )
    return raw, threat
