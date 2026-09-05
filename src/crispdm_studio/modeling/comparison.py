"""Leakage-safe supervised model comparison engine."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from crispdm_studio.modeling.classification import classification_estimators
from crispdm_studio.modeling.regression import regression_estimators
from crispdm_studio.modeling.target import TargetAssessment, assess_target
from crispdm_studio.preparation.preprocessing import PreparationResult, PreprocessingSpec
from crispdm_studio.understanding.profiler import UnderstandingResult
from crispdm_studio.understanding.task_inference import TaskType
from crispdm_studio.hardening import CAPS, deterministic_cap


@dataclass(frozen=True, slots=True)
class CVMetric:
    name: str
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class ModelResult:
    name: str
    is_baseline: bool
    fitted_pipeline: Pipeline
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    cv_metrics: tuple[CVMetric, ...]
    overfitting_flag: bool
    overfitting_reason: str | None


@dataclass(frozen=True, slots=True)
class SplitInfo:
    train_rows: int
    test_rows: int
    train_indices: tuple[Any, ...]
    test_indices: tuple[Any, ...]
    stratified: bool
    test_size: float
    random_state: int


@dataclass(frozen=True, slots=True)
class ModelingResult:
    target: str | None
    task_type: TaskType | None
    target_assessment: TargetAssessment
    skipped: bool
    skip_reason: str | None
    split: SplitInfo | None
    feature_columns: tuple[str, ...]
    model_results: tuple[ModelResult, ...]
    cv_folds: int
    primary_metric: str | None
    leakage_note: str
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None
    resource_notes: tuple[str, ...] = ()
    model_failures: tuple[str, ...] = ()

    @property
    def baseline(self) -> ModelResult | None:
        return next((item for item in self.model_results if item.is_baseline), None)


def _target_excluded_spec(spec: PreprocessingSpec, target: str) -> PreprocessingSpec:
    """Return a fresh immutable spec with the target removed from every predictor list."""
    remove = lambda cols: tuple(c for c in cols if c != target)
    return replace(
        spec,
        numeric_columns=remove(spec.numeric_columns),
        categorical_columns=remove(spec.categorical_columns),
        high_cardinality_columns=remove(spec.high_cardinality_columns),
        excluded_columns=tuple(dict.fromkeys((*spec.excluded_columns, target))),
    )


def _classification_metrics(y_true, y_pred, pipeline: Pipeline, X) -> dict[str, float]:
    labels = np.unique(y_true)
    average = "binary" if len(labels) == 2 else "weighted"
    positive = labels[-1] if len(labels) == 2 else None
    kwargs = {"average": average, "zero_division": 0}
    if positive is not None:
        kwargs["pos_label"] = positive
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, **kwargs)),
        "f1": float(f1_score(y_true, y_pred, **kwargs)),
    }
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X)
            if proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1], labels=pipeline.classes_))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=pipeline.classes_))
    except (ValueError, AttributeError):
        pass
    return metrics


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _cv_configuration(task: TaskType, y: pd.Series):
    n = len(y)
    if task is TaskType.CLASSIFICATION:
        min_class = int(y.value_counts().min())
        folds = min(5, min_class, max(0, n // 8))
        if folds < 2:
            return 0, None, None
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scoring = {
            "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
            "precision": make_scorer(precision_score, average="weighted", zero_division=0),
            "recall": make_scorer(recall_score, average="weighted", zero_division=0),
            "f1": make_scorer(f1_score, average="weighted", zero_division=0),
        }
        if y.nunique() == 2:
            scoring["roc_auc"] = "roc_auc"
        else:
            scoring["roc_auc"] = "roc_auc_ovr_weighted"
        return folds, cv, scoring
    folds = min(5, max(0, n // 8))
    if folds < 2:
        return 0, None, None
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    return folds, cv, scoring


def _summarize_cv(scores: dict[str, np.ndarray], scoring: dict[str, str]) -> tuple[CVMetric, ...]:
    out = []
    for name in scoring:
        values = np.asarray(scores[f"test_{name}"], dtype=float)
        if name in {"mae", "rmse"}:
            values = -values
        finite = values[np.isfinite(values)]
        if finite.size:
            out.append(CVMetric(name, float(finite.mean()), float(finite.std(ddof=0))))
    return tuple(out)


def _overfitting(task: TaskType, train: dict[str, float], test: dict[str, float]) -> tuple[bool, str | None]:
    if task is TaskType.CLASSIFICATION:
        gap = train.get("balanced_accuracy", 0.0) - test.get("balanced_accuracy", 0.0)
        if gap > 0.10:
            return True, f"Training balanced accuracy exceeds test balanced accuracy by {gap:.3f}."
    else:
        gap = train.get("r2", 0.0) - test.get("r2", 0.0)
        if gap > 0.15:
            return True, f"Training R² exceeds test R² by {gap:.3f}."
    return False, None


def run_modeling(
    dataframe: pd.DataFrame,
    understanding: UnderstandingResult,
    preparation: PreparationResult,
    target: str | None,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> ModelingResult:
    """Fit leakage-safe supervised pipelines after splitting raw structural rows.

    The Phase 3 spec remains immutable/unfitted. Every estimator receives a fresh
    transformer built from a target-excluded copy. train_test_split happens before
    any pipeline fit; cross_validate clones that pipeline and fits preprocessing
    independently inside each fold.
    """
    assessment = assess_target(dataframe, understanding, target)
    leakage_note = (
        "Split first, fit second: every model is a single sklearn Pipeline containing a fresh Phase 3 preprocessor plus "
        "the estimator. Imputation, IQR bounds, scaling, and encoding are learned from training data only; CV clones and "
        "refits the whole pipeline inside each fold. The original Phase 3 PreprocessingSpec is never fitted or mutated."
    )
    if not assessment.usable or target is None:
        return ModelingResult(target, assessment.task_type, assessment, True, assessment.reason, None, (), (), 0, None, leakage_note)

    working = dataframe.loc[dataframe[target].notna()].copy()
    working, cap_note = deterministic_cap(working, CAPS.modeling_max_rows, random_state=random_state)
    y = working[target]
    X = working.drop(columns=[target])
    spec = _target_excluded_spec(preparation.preprocessing_spec, target)
    feature_columns = tuple(c for c in (*spec.numeric_columns, *spec.categorical_columns) if c in X.columns)
    if not feature_columns:
        reason = "No usable predictor columns remain after excluding the confirmed target and structural/non-generic fields."
        return ModelingResult(target, assessment.task_type, assessment, True, reason, None, (), (), 0, None, leakage_note)
    X = X.loc[:, list(feature_columns)]

    stratify = y if assessment.task_type is TaskType.CLASSIFICATION else None
    if assessment.task_type is TaskType.CLASSIFICATION and int(y.value_counts().min()) < 2:
        reason = "At least one target class has fewer than two rows, so a stratified holdout cannot be created safely."
        return ModelingResult(target, assessment.task_type, assessment, True, reason, None, feature_columns, (), 0, None, leakage_note)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as exc:
        return ModelingResult(target, assessment.task_type, assessment, True, f"A safe train/test split could not be created: {exc}", None, feature_columns, (), 0, None, leakage_note)

    split = SplitInfo(
        train_rows=len(X_train), test_rows=len(X_test), train_indices=tuple(X_train.index), test_indices=tuple(X_test.index),
        stratified=assessment.task_type is TaskType.CLASSIFICATION, test_size=test_size, random_state=random_state,
    )
    estimators = classification_estimators() if assessment.task_type is TaskType.CLASSIFICATION else regression_estimators()
    folds, cv, scoring = _cv_configuration(assessment.task_type, y_train)
    results: list[ModelResult] = []
    model_failures: list[str] = []

    for name, estimator in estimators.items():
        try:
            pipeline = Pipeline([("preprocess", spec.build_transformer()), ("model", clone(estimator))])
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)
            if assessment.task_type is TaskType.CLASSIFICATION:
                train_metrics = _classification_metrics(y_train, train_pred, pipeline, X_train)
                test_metrics = _classification_metrics(y_test, test_pred, pipeline, X_test)
            else:
                train_metrics = _regression_metrics(y_train, train_pred)
                test_metrics = _regression_metrics(y_test, test_pred)

            cv_metrics: tuple[CVMetric, ...] = ()
            if folds and cv is not None and scoring is not None:
                cv_pipeline = Pipeline([("preprocess", spec.build_transformer()), ("model", clone(estimator))])
                try:
                    raw_scores = cross_validate(cv_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None, error_score=np.nan)
                    cv_metrics = _summarize_cv(raw_scores, scoring)
                except Exception:
                    cv_metrics = ()
            flag, reason = _overfitting(assessment.task_type, train_metrics, test_metrics)
            results.append(ModelResult(name, name == "Dummy baseline", pipeline, train_metrics, test_metrics, cv_metrics, flag, reason))
        except Exception as exc:
            model_failures.append(f"{name} could not be fit safely ({type(exc).__name__}).")

    if not results:
        reason = "All candidate estimators failed safely; no fitted model result is available."
        return ModelingResult(target, assessment.task_type, assessment, True, reason, split, feature_columns, (), folds, None, leakage_note, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), (cap_note,) if cap_note else (), tuple(model_failures))

    primary = "balanced_accuracy" if assessment.task_type is TaskType.CLASSIFICATION else "r2"
    return ModelingResult(target, assessment.task_type, assessment, False, None, split, feature_columns, tuple(results), folds, primary, leakage_note, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), (cap_note,) if cap_note else (), tuple(model_failures))


def transformer_is_unfitted(transformer) -> bool:
    try:
        check_is_fitted(transformer)
        return False
    except (NotFittedError, AttributeError):
        return True
