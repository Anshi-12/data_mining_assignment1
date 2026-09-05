"""Target validation and task routing for supervised modeling."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from crispdm_studio.understanding.profiler import UnderstandingResult
from crispdm_studio.understanding.task_inference import TaskType
from crispdm_studio.understanding.type_inference import FeatureType


@dataclass(frozen=True, slots=True)
class TargetAssessment:
    target: str | None
    task_type: TaskType | None
    usable: bool
    reason: str
    non_missing_rows: int = 0
    unique_values: int = 0


def assess_target(df: pd.DataFrame, understanding: UnderstandingResult, target: str | None) -> TargetAssessment:
    if not target:
        return TargetAssessment(None, None, False, "No target was confirmed. Supervised modeling requires an explicit user-selected target.")
    if target not in df.columns:
        return TargetAssessment(target, None, False, f"The confirmed target '{target}' is not available after structural cleaning.")

    profile = next((item for item in understanding.columns if item.name == target), None)
    series = df[target]
    observed = series.dropna()
    unique = int(observed.nunique(dropna=True))
    if len(observed) < 8:
        return TargetAssessment(target, None, False, f"Target '{target}' has only {len(observed)} non-missing rows; at least 8 are required.", len(observed), unique)
    if unique < 2:
        return TargetAssessment(target, None, False, f"Target '{target}' has fewer than two observed values, so supervised learning is not defensible.", len(observed), unique)
    if profile and profile.feature_type in {FeatureType.IDENTIFIER, FeatureType.TEXT, FeatureType.DATETIME, FeatureType.EMPTY}:
        return TargetAssessment(target, None, False, f"Target '{target}' is inferred as {profile.feature_type.value}; the generic supervised workflow does not treat that type as a defensible outcome.", len(observed), unique)

    if pd.api.types.is_bool_dtype(observed.dtype) or (profile and profile.feature_type in {FeatureType.CATEGORICAL, FeatureType.BOOLEAN}):
        if unique > min(50, max(10, len(observed) // 5)):
            return TargetAssessment(target, None, False, f"Target '{target}' has {unique} classes, which is too high-cardinality for the generic classification workflow.", len(observed), unique)
        return TargetAssessment(target, TaskType.CLASSIFICATION, True, f"'{target}' is categorical/boolean with {unique} observed classes.", len(observed), unique)

    if pd.api.types.is_numeric_dtype(observed.dtype):
        classification_cutoff = min(20, max(2, int(0.05 * len(observed))))
        if unique <= classification_cutoff:
            return TargetAssessment(target, TaskType.CLASSIFICATION, True, f"'{target}' is numeric but has only {unique} distinct values, so it is routed to classification.", len(observed), unique)
        if unique >= max(8, int(0.10 * len(observed))):
            return TargetAssessment(target, TaskType.REGRESSION, True, f"'{target}' is numeric with {unique} distinct values, so it is routed to regression.", len(observed), unique)

    return TargetAssessment(target, None, False, f"Target '{target}' does not have a defensible generic classification or regression structure.", len(observed), unique)
