"""Data-quality and leakage heuristics for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

import pandas as pd

from crispdm_studio.understanding.type_inference import FeatureType, TypeInference


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class QualityWarning:
    code: str
    severity: WarningSeverity
    message: str
    column: str | None = None


@dataclass(frozen=True, slots=True)
class LeakageFlag:
    column: str
    risk: str
    reason: str


_LEAKAGE_NAME_RE = re.compile(
    r"(^|_)(target|label|outcome|result|actual|ground.?truth|prediction|predicted|score_after|post|final)(_|$)",
    re.IGNORECASE,
)


def detect_leakage_flags(
    df: pd.DataFrame, inferences: dict[str, TypeInference]
) -> list[LeakageFlag]:
    """Flag columns that deserve review before supervised modeling.

    These are warnings, not proof of leakage. The app deliberately avoids claiming
    semantic knowledge that is unavailable from an arbitrary CSV.
    """
    flags: list[LeakageFlag] = []
    for column in df.columns:
        series = df[column]
        inference = inferences[column]
        non_null = series.dropna()
        if inference.feature_type is FeatureType.IDENTIFIER:
            flags.append(
                LeakageFlag(
                    column=column,
                    risk="medium",
                    reason="ID-like columns can let a model memorize records and usually should not be predictors.",
                )
            )
        if _LEAKAGE_NAME_RE.search(column):
            flags.append(
                LeakageFlag(
                    column=column,
                    risk="high",
                    reason="The column name suggests a target, outcome, prediction, or post-event value; review its timing before modeling.",
                )
            )
        if not non_null.empty and non_null.nunique(dropna=True) == len(non_null) and len(non_null) >= 20:
            if inference.feature_type not in {FeatureType.IDENTIFIER, FeatureType.TEXT}:
                flags.append(
                    LeakageFlag(
                        column=column,
                        risk="low",
                        reason="Every observed value is unique; this feature may behave like a record identifier.",
                    )
                )

    # Deduplicate same-column/same-risk messages while preserving order.
    seen: set[tuple[str, str, str]] = set()
    result: list[LeakageFlag] = []
    for flag in flags:
        key = (flag.column, flag.risk, flag.reason)
        if key not in seen:
            seen.add(key)
            result.append(flag)
    return result


def build_quality_warnings(
    df: pd.DataFrame,
    inferences: dict[str, TypeInference],
    *,
    near_constant_threshold: float = 0.95,
) -> list[QualityWarning]:
    warnings: list[QualityWarning] = []
    rows, columns = df.shape

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(
            QualityWarning(
                code="duplicate_rows",
                severity=WarningSeverity.WARNING,
                message=f"{duplicate_rows:,} duplicate row(s) were detected ({duplicate_rows / rows:.1%} of rows).",
            )
        )

    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        missing_ratio = missing / rows if rows else 0.0
        non_null = series.dropna()
        unique = int(non_null.nunique(dropna=True))

        if missing_ratio >= 0.80:
            warnings.append(
                QualityWarning(
                    code="very_high_missingness",
                    severity=WarningSeverity.HIGH,
                    column=column,
                    message=f"'{column}' is {missing_ratio:.1%} missing; its analytical value may be limited.",
                )
            )
        elif missing_ratio >= 0.40:
            warnings.append(
                QualityWarning(
                    code="high_missingness",
                    severity=WarningSeverity.WARNING,
                    column=column,
                    message=f"'{column}' is {missing_ratio:.1%} missing and will need an explicit preparation decision.",
                )
            )

        if unique <= 1:
            warnings.append(
                QualityWarning(
                    code="constant_column",
                    severity=WarningSeverity.WARNING,
                    column=column,
                    message=f"'{column}' is constant (or has only one observed value) and carries no variation for modeling.",
                )
            )
        elif not non_null.empty:
            top_share = float(non_null.value_counts(normalize=True, dropna=True).iloc[0])
            if top_share >= near_constant_threshold:
                warnings.append(
                    QualityWarning(
                        code="near_constant_column",
                        severity=WarningSeverity.INFO,
                        column=column,
                        message=f"'{column}' is near-constant: its most common observed value represents {top_share:.1%} of non-missing rows.",
                    )
                )

        if (
            inferences[column].feature_type is FeatureType.CATEGORICAL
            and unique >= 50
            and unique / max(len(non_null), 1) >= 0.20
        ):
            warnings.append(
                QualityWarning(
                    code="high_cardinality",
                    severity=WarningSeverity.WARNING,
                    column=column,
                    message=f"'{column}' has high categorical cardinality ({unique:,} distinct values), so naive one-hot encoding could be expensive.",
                )
            )

    if columns == 1:
        warnings.append(
            QualityWarning(
                code="single_column",
                severity=WarningSeverity.INFO,
                message="The dataset has one column, which limits relationship analysis, clustering, and supervised modeling.",
            )
        )
    if rows < 30:
        warnings.append(
            QualityWarning(
                code="small_sample",
                severity=WarningSeverity.INFO,
                message=f"The dataset contains only {rows:,} rows; model evaluation will have high uncertainty.",
            )
        )

    return warnings
