"""Candidate supervised-learning target inference and CRISP-DM objective generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

import pandas as pd

from crispdm_studio.understanding.type_inference import FeatureType, TypeInference


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass(frozen=True, slots=True)
class TargetCandidate:
    column: str
    task_type: TaskType
    score: float
    reasons: tuple[str, ...]
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalyticalObjective:
    known_facts: tuple[str, ...]
    inferred_objectives: tuple[str, ...]
    honesty_note: str


_TARGET_NAME_RE = re.compile(
    r"(^|_)(target|label|outcome|class|response|status|churn|default|fraud|survived|price|sales|revenue|score)(_|$)",
    re.IGNORECASE,
)


def infer_target_candidates(
    df: pd.DataFrame, inferences: dict[str, TypeInference], *, max_candidates: int = 8
) -> list[TargetCandidate]:
    candidates: list[TargetCandidate] = []
    row_count = len(df)

    for column in df.columns:
        series = df[column]
        inference = inferences[column]
        non_null = series.dropna()
        observed = len(non_null)
        unique = int(non_null.nunique(dropna=True))
        if observed == 0 or inference.feature_type in {
            FeatureType.IDENTIFIER,
            FeatureType.TEXT,
            FeatureType.DATETIME,
            FeatureType.EMPTY,
        }:
            continue
        if observed / max(row_count, 1) < 0.50:
            continue

        score = 0.0
        reasons: list[str] = []
        caveats: list[str] = []
        name_hint = bool(_TARGET_NAME_RE.search(column))
        if name_hint:
            score += 0.35
            reasons.append("Its name resembles a commonly used outcome/label field.")

        if inference.feature_type in {FeatureType.CATEGORICAL, FeatureType.BOOLEAN}:
            if 2 <= unique <= min(50, max(2, int(0.2 * row_count))):
                score += 0.45
                reasons.append(f"It has {unique} observed classes, a plausible classification target size.")
                task_type = TaskType.CLASSIFICATION
            else:
                continue
        elif inference.feature_type is FeatureType.NUMERIC:
            if 2 <= unique <= min(20, max(2, int(0.05 * row_count))):
                score += 0.35
                reasons.append(
                    f"It is numeric but has only {unique} distinct values, so classification may be plausible."
                )
                task_type = TaskType.CLASSIFICATION
            elif unique >= max(10, int(0.1 * observed)):
                score += 0.35
                reasons.append("It is numeric with enough variation for a regression target.")
                task_type = TaskType.REGRESSION
            else:
                continue
        else:
            continue

        missing_ratio = 1 - observed / max(row_count, 1)
        if missing_ratio == 0:
            score += 0.10
            reasons.append("It has no missing target values.")
        elif missing_ratio > 0.20:
            caveats.append(f"{missing_ratio:.1%} of rows have a missing value in this candidate target.")

        if column == df.columns[-1]:
            score += 0.05
            reasons.append("It is the final column, a weak structural signal sometimes used for labels.")

        candidates.append(
            TargetCandidate(
                column=column,
                task_type=task_type,
                score=min(score, 1.0),
                reasons=tuple(reasons),
                caveats=tuple(caveats),
            )
        )

    return sorted(candidates, key=lambda item: (-item.score, item.column))[:max_candidates]


def generate_analytical_objective(
    df: pd.DataFrame,
    inferences: dict[str, TypeInference],
    candidates: list[TargetCandidate],
) -> AnalyticalObjective:
    rows, columns = df.shape
    type_counts: dict[FeatureType, int] = {kind: 0 for kind in FeatureType}
    for inference in inferences.values():
        type_counts[inference.feature_type] += 1

    known = [
        f"The uploaded dataset contains {rows:,} rows and {columns:,} columns.",
        f"The schema contains {type_counts[FeatureType.NUMERIC]:,} numeric, "
        f"{type_counts[FeatureType.CATEGORICAL] + type_counts[FeatureType.BOOLEAN]:,} categorical/boolean, "
        f"{type_counts[FeatureType.DATETIME]:,} datetime-like, and "
        f"{type_counts[FeatureType.IDENTIFIER]:,} identifier-like column(s) based on observed values and names.",
        "No business objective, target definition, decision cost, or domain context was supplied in the CSV itself.",
    ]

    inferred = [
        "Assess data quality and prepare a defensible dataset for exploratory analysis.",
        "Explore distributions, relationships, and natural segments where the available feature types support them.",
    ]
    if candidates:
        candidate_names = ", ".join(candidate.column for candidate in candidates[:3])
        inferred.append(
            "Evaluate supervised learning only after a user confirms an appropriate target; "
            f"current plausible candidates include {candidate_names}."
        )
    else:
        inferred.append(
            "Do not assume a supervised-learning target from the available schema; focus on descriptive and unsupervised analysis unless the user supplies one."
        )

    return AnalyticalObjective(
        known_facts=tuple(known),
        inferred_objectives=tuple(inferred),
        honesty_note=(
            "The inferred objectives are analytical hypotheses generated from dataset structure, not claims about the uploader's real business intent."
        ),
    )
