"""Leakage-safe structural cleaning for Phase 3.

Only deterministic operations that do not learn distributional parameters are
allowed to mutate the full dataframe here. Anything that estimates statistics,
category vocabularies, or bounds belongs in the deferred preprocessing spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd

from crispdm_studio.understanding.profiler import UnderstandingResult
from crispdm_studio.understanding.type_inference import FeatureType


class DecisionStage(StrEnum):
    APPLIED_NOW = "applied_now_structural"
    DEFERRED = "deferred_to_modeling"


@dataclass(frozen=True, slots=True)
class PreparationDecision:
    action: str
    stage: DecisionStage
    reason: str
    rows_affected: int = 0
    columns_affected: int = 0
    columns: tuple[str, ...] = ()
    before: str | None = None
    after: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class StructuralCleaningResult:
    dataframe: pd.DataFrame
    decisions: tuple[PreparationDecision, ...]


def _lossless_numeric_candidates(
    df: pd.DataFrame, understanding: UnderstandingResult
) -> list[str]:
    """Return object/string columns that can be converted without creating nulls.

    This is deterministic type repair, not statistical estimation. ID, datetime,
    boolean, text and already-numeric columns are intentionally excluded.
    """
    profile_by_name = {profile.name: profile for profile in understanding.columns}
    candidates: list[str] = []
    for column in df.columns:
        profile = profile_by_name.get(column)
        if profile is None or profile.feature_type in {
            FeatureType.IDENTIFIER,
            FeatureType.DATETIME,
            FeatureType.BOOLEAN,
            FeatureType.TEXT,
            FeatureType.NUMERIC,
            FeatureType.EMPTY,
        }:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series.dtype):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        converted = pd.to_numeric(non_null, errors="coerce")
        if converted.notna().all():
            candidates.append(column)
    return candidates


def apply_structural_cleaning(
    dataframe: pd.DataFrame, understanding: UnderstandingResult
) -> StructuralCleaningResult:
    """Apply only leakage-safe structural cleaning to the full dataframe."""
    df = dataframe.copy(deep=True)
    decisions: list[PreparationDecision] = []

    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        before_rows = len(df)
        df = df.loc[~duplicate_mask].copy()
        decisions.append(
            PreparationDecision(
                action="Remove exact duplicate rows",
                stage=DecisionStage.APPLIED_NOW,
                reason="Exact duplicate records add repeated weight without adding information; removal uses no learned statistics.",
                rows_affected=duplicate_count,
                before=f"{before_rows:,} rows",
                after=f"{len(df):,} rows",
            )
        )
    else:
        decisions.append(
            PreparationDecision(
                action="Check exact duplicate rows",
                stage=DecisionStage.APPLIED_NOW,
                reason="No exact duplicate rows were present, so no rows were removed.",
                rows_affected=0,
                before=f"{len(df):,} rows",
                after=f"{len(df):,} rows",
            )
        )

    existing_columns = set(df.columns)
    constant_columns = tuple(
        column for column in understanding.dataset.constant_columns if column in existing_columns
    )
    if constant_columns:
        df = df.drop(columns=list(constant_columns))
    decisions.append(
        PreparationDecision(
            action="Remove constant columns",
            stage=DecisionStage.APPLIED_NOW,
            reason=(
                "Constant columns contain no observed variation and therefore cannot discriminate records. "
                "This structural decision does not estimate a model parameter."
            ),
            columns_affected=len(constant_columns),
            columns=constant_columns,
            before=f"{len(existing_columns):,} columns",
            after=f"{df.shape[1]:,} columns",
        )
    )

    identifier_columns = tuple(column for column in understanding.identifiers if column in df.columns)
    if identifier_columns:
        before_cols = df.shape[1]
        df = df.drop(columns=list(identifier_columns))
    else:
        before_cols = df.shape[1]
    decisions.append(
        PreparationDecision(
            action="Exclude identifier-like columns",
            stage=DecisionStage.APPLIED_NOW,
            reason=(
                "Identifier-like columns usually represent record identity rather than reusable signal and can enable memorization. "
                "They are removed before generic modeling unless a later domain-specific override is introduced."
            ),
            columns_affected=len(identifier_columns),
            columns=identifier_columns,
            before=f"{before_cols:,} columns",
            after=f"{df.shape[1]:,} columns",
        )
    )

    numeric_candidates = _lossless_numeric_candidates(df, understanding)
    converted_columns: list[str] = []
    converted_cells = 0
    for column in numeric_candidates:
        before = df[column]
        non_null_count = int(before.notna().sum())
        converted = pd.to_numeric(before, errors="coerce")
        if int(converted.notna().sum()) == non_null_count:
            df[column] = converted
            converted_columns.append(column)
            converted_cells += non_null_count
    decisions.append(
        PreparationDecision(
            action="Repair losslessly numeric string columns",
            stage=DecisionStage.APPLIED_NOW,
            reason=(
                "A string column is converted only when every observed non-missing value parses as numeric; "
                "no distributional threshold or fitted statistic is used and no new missing values are introduced."
            ),
            rows_affected=converted_cells,
            columns_affected=len(converted_columns),
            columns=tuple(converted_columns),
            details={"converted_non_missing_cells": converted_cells},
        )
    )

    return StructuralCleaningResult(dataframe=df, decisions=tuple(decisions))
