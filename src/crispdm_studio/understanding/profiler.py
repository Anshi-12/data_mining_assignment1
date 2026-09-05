"""Pure Phase 2 dataset-understanding engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crispdm_studio.models import IngestedDataset
from crispdm_studio.understanding.quality import (
    LeakageFlag,
    QualityWarning,
    build_quality_warnings,
    detect_leakage_flags,
)
from crispdm_studio.understanding.task_inference import (
    AnalyticalObjective,
    TargetCandidate,
    generate_analytical_objective,
    infer_target_candidates,
)
from crispdm_studio.understanding.type_inference import FeatureType, TypeInference, infer_feature_type


@dataclass(frozen=True, slots=True)
class DistributionItem:
    label: str
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class NumericSummary:
    mean: float | None
    std: float | None
    minimum: float | None
    q25: float | None
    median: float | None
    q75: float | None
    maximum: float | None
    skewness: float | None


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    name: str
    storage_dtype: str
    feature_type: FeatureType
    type_confidence: float
    type_reason: str
    rows: int
    non_null: int
    missing: int
    missing_ratio: float
    unique: int
    uniqueness_ratio: float
    is_constant: bool
    is_near_constant: bool
    dominant_value_share: float | None
    possible_timestamp: bool
    numeric_summary: NumericSummary | None
    top_values: tuple[DistributionItem, ...]
    observed_summary: str


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    rows: int
    columns: int
    duplicate_rows: int
    duplicate_row_ratio: float
    missing_cells: int
    missing_cell_ratio: float
    feature_type_counts: dict[str, int]
    possible_timestamp_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    near_constant_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnderstandingResult:
    dataset: DatasetProfile
    columns: tuple[ColumnProfile, ...]
    identifiers: tuple[str, ...]
    leakage_flags: tuple[LeakageFlag, ...]
    target_candidates: tuple[TargetCandidate, ...]
    quality_warnings: tuple[QualityWarning, ...]
    objective: AnalyticalObjective


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _numeric_summary(series: pd.Series) -> NumericSummary:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return NumericSummary(None, None, None, None, None, None, None, None)
    return NumericSummary(
        mean=_finite_or_none(numeric.mean()),
        std=_finite_or_none(numeric.std()),
        minimum=_finite_or_none(numeric.min()),
        q25=_finite_or_none(numeric.quantile(0.25)),
        median=_finite_or_none(numeric.median()),
        q75=_finite_or_none(numeric.quantile(0.75)),
        maximum=_finite_or_none(numeric.max()),
        skewness=_finite_or_none(numeric.skew()),
    )


def _top_values(series: pd.Series, limit: int = 5) -> tuple[DistributionItem, ...]:
    non_null = series.dropna()
    if non_null.empty:
        return ()
    counts = non_null.astype("string").value_counts(dropna=False).head(limit)
    total = len(non_null)
    return tuple(
        DistributionItem(label=str(label), count=int(count), share=float(count / total))
        for label, count in counts.items()
    )


def _profile_column(
    name: str,
    series: pd.Series,
    inference: TypeInference,
    *,
    near_constant_threshold: float,
) -> ColumnProfile:
    rows = len(series)
    non_null = series.dropna()
    non_null_count = len(non_null)
    missing = int(series.isna().sum())
    unique = int(non_null.nunique(dropna=True))
    uniqueness_ratio = unique / max(non_null_count, 1)
    top_values = _top_values(series)
    dominant_share = top_values[0].share if top_values else None
    is_constant = unique <= 1
    is_near_constant = bool(
        not is_constant and dominant_share is not None and dominant_share >= near_constant_threshold
    )
    numeric_summary = (
        _numeric_summary(series) if inference.feature_type is FeatureType.NUMERIC else None
    )

    facts = [
        f"{missing / max(rows, 1):.1%} missing",
        f"{unique:,} distinct observed value{'s' if unique != 1 else ''}",
    ]
    if dominant_share is not None:
        facts.append(f"most common observed value share {dominant_share:.1%}")
    if numeric_summary and numeric_summary.median is not None:
        facts.append(f"median {numeric_summary.median:,.4g}")
        if numeric_summary.skewness is not None:
            facts.append(f"skewness {numeric_summary.skewness:,.3g}")

    return ColumnProfile(
        name=name,
        storage_dtype=str(series.dtype),
        feature_type=inference.feature_type,
        type_confidence=inference.confidence,
        type_reason=inference.reason,
        rows=rows,
        non_null=non_null_count,
        missing=missing,
        missing_ratio=missing / max(rows, 1),
        unique=unique,
        uniqueness_ratio=uniqueness_ratio,
        is_constant=is_constant,
        is_near_constant=is_near_constant,
        dominant_value_share=dominant_share,
        possible_timestamp=inference.possible_timestamp,
        numeric_summary=numeric_summary,
        top_values=top_values,
        observed_summary="; ".join(facts) + ".",
    )


def profile_dataset(
    ingested: IngestedDataset, *, near_constant_threshold: float = 0.95
) -> UnderstandingResult:
    """Return structured Business/Data Understanding results for an ingested CSV."""
    df = ingested.dataframe
    inferences = {column: infer_feature_type(column, df[column]) for column in df.columns}
    columns = tuple(
        _profile_column(
            column,
            df[column],
            inferences[column],
            near_constant_threshold=near_constant_threshold,
        )
        for column in df.columns
    )

    rows, column_count = df.shape
    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    total_cells = rows * column_count
    type_counts: dict[str, int] = {}
    for inference in inferences.values():
        type_counts[inference.feature_type.value] = type_counts.get(inference.feature_type.value, 0) + 1

    dataset_profile = DatasetProfile(
        rows=rows,
        columns=column_count,
        duplicate_rows=duplicate_rows,
        duplicate_row_ratio=duplicate_rows / max(rows, 1),
        missing_cells=missing_cells,
        missing_cell_ratio=missing_cells / max(total_cells, 1),
        feature_type_counts=type_counts,
        possible_timestamp_columns=tuple(
            column for column, inference in inferences.items() if inference.possible_timestamp
        ),
        constant_columns=tuple(profile.name for profile in columns if profile.is_constant),
        near_constant_columns=tuple(profile.name for profile in columns if profile.is_near_constant),
    )

    target_candidates = infer_target_candidates(df, inferences)
    return UnderstandingResult(
        dataset=dataset_profile,
        columns=columns,
        identifiers=tuple(
            column
            for column, inference in inferences.items()
            if inference.feature_type is FeatureType.IDENTIFIER
        ),
        leakage_flags=tuple(detect_leakage_flags(df, inferences)),
        target_candidates=tuple(target_candidates),
        quality_warnings=tuple(
            build_quality_warnings(
                df, inferences, near_constant_threshold=near_constant_threshold
            )
        ),
        objective=generate_analytical_objective(df, inferences, target_candidates),
    )
