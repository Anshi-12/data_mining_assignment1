"""Reasoned model selection and reusable Phase 7 evaluation structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from crispdm_studio.modeling.comparison import CVMetric, ModelResult, ModelingResult
from crispdm_studio.understanding.task_inference import TaskType
from crispdm_studio.hardening import CAPS


@dataclass(frozen=True, slots=True)
class BaselineImprovement:
    model_name: str
    metric: str
    baseline_value: float
    model_value: float
    absolute_improvement: float
    relative_improvement: float | None
    beats_baseline: bool


@dataclass(frozen=True, slots=True)
class SelectionScore:
    model_name: str
    test_improvement: float
    cv_improvement: float | None
    cv_std: float | None
    overfitting_penalty: float
    complexity_penalty: float
    composite_score: float
    reason: str


@dataclass(frozen=True, slots=True)
class ClassImbalanceDiagnostic:
    applicable: bool
    majority_share: float | None
    minority_share: float | None
    class_count: int | None
    flagged: bool
    message: str


@dataclass(frozen=True, slots=True)
class PermutationImportanceItem:
    feature: str
    importance_mean: float
    importance_std: float


_COMPLEXITY = {
    "Logistic Regression": 1,
    "Ridge": 1,
    "HistGradientBoosting": 2,
    "Random Forest": 3,
}


def _cv_metric(model: ModelResult, name: str) -> CVMetric | None:
    return next((m for m in model.cv_metrics if m.name == name), None)


def baseline_improvements(modeling: ModelingResult) -> tuple[BaselineImprovement, ...]:
    baseline = modeling.baseline
    metric = modeling.primary_metric
    if baseline is None or metric is None:
        return ()
    base = float(baseline.test_metrics.get(metric, np.nan))
    out: list[BaselineImprovement] = []
    for model in modeling.model_results:
        if model.is_baseline or metric not in model.test_metrics:
            continue
        value = float(model.test_metrics[metric])
        absolute = value - base
        relative = None if abs(base) < 1e-12 else absolute / abs(base)
        out.append(BaselineImprovement(model.name, metric, base, value, absolute, relative, absolute > 1e-9))
    return tuple(out)


def class_imbalance(modeling: ModelingResult) -> ClassImbalanceDiagnostic:
    if modeling.task_type is not TaskType.CLASSIFICATION or modeling.y_train is None or modeling.y_test is None:
        return ClassImbalanceDiagnostic(False, None, None, None, False, "Class imbalance does not apply to this regression task.")
    y = np.concatenate([modeling.y_train.to_numpy(), modeling.y_test.to_numpy()])
    _, counts = np.unique(y, return_counts=True)
    shares = counts / counts.sum()
    majority = float(shares.max())
    minority = float(shares.min())
    flagged = majority >= 0.70 or minority <= 0.15
    if flagged:
        msg = f"Target classes are imbalanced: largest class {majority:.1%}, smallest class {minority:.1%}."
    else:
        msg = f"Target balance is moderate: largest class {majority:.1%}, smallest class {minority:.1%}."
    return ClassImbalanceDiagnostic(True, majority, minority, int(len(counts)), flagged, msg)


def reasoned_selection(modeling: ModelingResult) -> tuple[ModelResult | None, tuple[SelectionScore, ...], str]:
    baseline = modeling.baseline
    metric = modeling.primary_metric
    if modeling.skipped or baseline is None or metric is None:
        return None, (), modeling.skip_reason or "No completed supervised modeling result is available."

    baseline_test = float(baseline.test_metrics.get(metric, np.nan))
    baseline_cv_obj = _cv_metric(baseline, metric)
    baseline_cv = baseline_cv_obj.mean if baseline_cv_obj else None
    scores: list[SelectionScore] = []
    candidates: list[tuple[float, ModelResult]] = []

    for model in modeling.model_results:
        if model.is_baseline or metric not in model.test_metrics:
            continue
        test_imp = float(model.test_metrics[metric]) - baseline_test
        cv_obj = _cv_metric(model, metric)
        cv_imp = None if cv_obj is None or baseline_cv is None else float(cv_obj.mean - baseline_cv)
        cv_std = None if cv_obj is None else float(cv_obj.std)
        overfit_penalty = 0.15 if model.overfitting_flag else 0.0
        complexity_penalty = 0.015 * _COMPLEXITY.get(model.name, 2)
        stability_penalty = 0.50 * (cv_std if cv_std is not None else 0.05)
        cv_term = 0.50 * (cv_imp if cv_imp is not None else 0.0)
        composite = test_imp + cv_term - stability_penalty - overfit_penalty - complexity_penalty
        reason = (
            f"test improvement {test_imp:+.3f}; "
            + (f"CV improvement {cv_imp:+.3f} with std {cv_std:.3f}; " if cv_imp is not None and cv_std is not None else "CV evidence unavailable; ")
            + ("overfitting penalty applied; " if model.overfitting_flag else "no configured overfitting penalty; ")
            + f"complexity penalty {complexity_penalty:.3f}."
        )
        item = SelectionScore(model.name, test_imp, cv_imp, cv_std, overfit_penalty, complexity_penalty, composite, reason)
        scores.append(item)
        # Must beat baseline on holdout and not clearly underperform baseline in CV when CV exists.
        defensible = test_imp > 1e-9 and (cv_imp is None or cv_imp > -0.02)
        if defensible:
            candidates.append((composite, model))

    if not candidates:
        return None, tuple(scores), f"No candidate model defensibly beat the Dummy baseline on {metric.replace('_', ' ')} while satisfying the CV guardrail."
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[0][1]
    selected_score = next(s for s in scores if s.model_name == selected.name)
    rationale = (
        f"{selected.name} was selected because it beat the Dummy baseline on held-out {metric.replace('_', ' ')} "
        f"by {selected_score.test_improvement:+.3f} and achieved the strongest risk-adjusted score after considering "
        "CV stability, train/test divergence, and model complexity."
    )
    return selected, tuple(scores), rationale


def compute_permutation_importance(modeling: ModelingResult, selected: ModelResult, *, n_repeats: int = CAPS.permutation_repeats, random_state: int = 42) -> tuple[PermutationImportanceItem, ...]:
    if modeling.X_test is None or modeling.y_test is None or modeling.primary_metric is None:
        return ()
    scoring = "balanced_accuracy" if modeling.task_type is TaskType.CLASSIFICATION else "r2"
    X_test = modeling.X_test
    y_test = modeling.y_test
    if len(X_test) > CAPS.permutation_max_rows:
        X_test = X_test.sample(n=CAPS.permutation_max_rows, random_state=random_state).sort_index()
        y_test = y_test.loc[X_test.index]
    result = permutation_importance(
        selected.fitted_pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=None,
    )
    items = [
        PermutationImportanceItem(str(feature), float(mean), float(std))
        for feature, mean, std in zip(X_test.columns, result.importances_mean, result.importances_std)
    ]
    return tuple(sorted(items, key=lambda x: x.importance_mean, reverse=True))
