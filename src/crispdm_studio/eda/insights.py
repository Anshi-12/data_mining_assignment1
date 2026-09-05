"""Computed, structured EDA findings. No canned dataset claims live here."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from crispdm_studio.eda.relationships import (
    CategoricalAssociation,
    CorrelationPair,
    NumericCategoricalRelationship,
)
from crispdm_studio.eda.statistics import CategoricalDistribution, NumericDistribution


class FindingKind(StrEnum):
    NUMERIC = "numeric_distribution"
    CATEGORICAL = "categorical_distribution"
    MISSINGNESS = "missingness"
    OUTLIER = "outlier_diagnostic"
    CORRELATION = "correlation"
    NUMERIC_CATEGORICAL = "numeric_categorical_relationship"
    CATEGORICAL_ASSOCIATION = "categorical_association"


@dataclass(frozen=True, slots=True)
class EDAFinding:
    finding_id: str
    kind: FindingKind
    title: str
    text: str
    columns: tuple[str, ...]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkippedAnalysis:
    analysis: str
    reason: str


def numeric_findings(items: tuple[NumericDistribution, ...]) -> tuple[EDAFinding, ...]:
    findings: list[EDAFinding] = []
    for item in items:
        if item.count == 0:
            continue
        parts = [f"{item.column} has {item.count:,} observed numeric values"]
        if item.median is not None:
            parts.append(f"median {item.median:,.4g}")
        if item.skewness is not None:
            if item.skewness >= 0.75:
                parts.append(f"right skew {item.skewness:.2f}")
            elif item.skewness <= -0.75:
                parts.append(f"left skew {item.skewness:.2f}")
            else:
                parts.append(f"skewness {item.skewness:.2f}")
        if item.q95 is not None and item.median is not None:
            distance = item.q95 - item.median
            parts.append(f"95th percentile {item.q95:,.4g} ({distance:,.4g} above the median)")
        findings.append(
            EDAFinding(
                finding_id=f"numeric:{item.column}",
                kind=FindingKind.NUMERIC,
                title=f"Distribution of {item.column}",
                text="; ".join(parts) + ".",
                columns=(item.column,),
                metrics={
                    "count": item.count,
                    "median": item.median,
                    "skewness": item.skewness,
                    "q95": item.q95,
                    "q05": item.q05,
                },
            )
        )
        if item.outlier_count > 0 and item.lower_fence is not None and item.upper_fence is not None:
            findings.append(
                EDAFinding(
                    finding_id=f"outliers:{item.column}",
                    kind=FindingKind.OUTLIER,
                    title=f"Outlier diagnostic for {item.column}",
                    text=(
                        f"{item.column} has {item.outlier_count:,} observations ({item.outlier_ratio:.1%}) "
                        f"outside the descriptive 1.5×IQR fences [{item.lower_fence:,.4g}, {item.upper_fence:,.4g}]. "
                        "These are diagnostic flags only; Phase 4 does not remove or clip them."
                    ),
                    columns=(item.column,),
                    metrics={
                        "outlier_count": item.outlier_count,
                        "outlier_ratio": item.outlier_ratio,
                        "lower_fence": item.lower_fence,
                        "upper_fence": item.upper_fence,
                    },
                )
            )
    return tuple(findings)


def categorical_findings(items: tuple[CategoricalDistribution, ...]) -> tuple[EDAFinding, ...]:
    findings: list[EDAFinding] = []
    for item in items:
        if not item.top_values:
            continue
        top = item.top_values[0]
        imbalance = "strongly concentrated" if top.share >= 0.80 else "moderately concentrated" if top.share >= 0.60 else "not dominated by one level"
        findings.append(
            EDAFinding(
                finding_id=f"categorical:{item.column}",
                kind=FindingKind.CATEGORICAL,
                title=f"Distribution of {item.column}",
                text=(
                    f"{item.column} has {item.cardinality:,} observed levels; the most common value "
                    f"'{top.label}' occurs {top.count:,} times ({top.share:.1%}), so the distribution is {imbalance}."
                ),
                columns=(item.column,),
                metrics={
                    "cardinality": item.cardinality,
                    "dominant_value": top.label,
                    "dominant_share": top.share,
                    "normalized_entropy": item.normalized_entropy,
                },
            )
        )
    return tuple(findings)


def missingness_findings(missing: tuple[tuple[str, int, float], ...]) -> tuple[EDAFinding, ...]:
    findings: list[EDAFinding] = []
    for column, count, ratio in missing:
        if count <= 0:
            continue
        findings.append(
            EDAFinding(
                finding_id=f"missing:{column}",
                kind=FindingKind.MISSINGNESS,
                title=f"Missingness in {column}",
                text=f"{column} is missing in {count:,} rows ({ratio:.1%} of the structurally cleaned dataset).",
                columns=(column,),
                metrics={"missing_count": count, "missing_ratio": ratio},
            )
        )
    return tuple(findings)


def relationship_findings(
    correlations: tuple[CorrelationPair, ...],
    numeric_categorical: tuple[NumericCategoricalRelationship, ...],
    categorical: tuple[CategoricalAssociation, ...],
) -> tuple[EDAFinding, ...]:
    findings: list[EDAFinding] = []
    for item in correlations:
        if abs(item.correlation) < 0.30:
            continue
        direction = "positive" if item.correlation > 0 else "negative"
        findings.append(
            EDAFinding(
                finding_id=f"correlation:{item.left}:{item.right}",
                kind=FindingKind.CORRELATION,
                title=f"Correlation: {item.left} vs {item.right}",
                text=(
                    f"{item.left} and {item.right} have Pearson r={item.correlation:.2f} across "
                    f"{item.observations:,} complete observations, indicating a {direction} linear association."
                ),
                columns=(item.left, item.right),
                metrics={"pearson_r": item.correlation, "observations": item.observations},
            )
        )
    for item in numeric_categorical:
        if item.eta_squared < 0.05:
            continue
        strongest = item.group_medians[0] if item.group_medians else None
        suffix = ""
        if strongest is not None:
            suffix = f" The highest group median is '{strongest[0]}' at {strongest[1]:,.4g}."
        findings.append(
            EDAFinding(
                finding_id=f"numcat:{item.numeric}:{item.categorical}",
                kind=FindingKind.NUMERIC_CATEGORICAL,
                title=f"{item.numeric} by {item.categorical}",
                text=(
                    f"Group membership in {item.categorical} explains eta-squared={item.eta_squared:.2f} "
                    f"of observed variation in {item.numeric} across {item.observations:,} complete rows.{suffix}"
                ),
                columns=(item.numeric, item.categorical),
                metrics={"eta_squared": item.eta_squared, "observations": item.observations},
            )
        )
    for item in categorical:
        if item.cramers_v < 0.10:
            continue
        findings.append(
            EDAFinding(
                finding_id=f"catassoc:{item.left}:{item.right}",
                kind=FindingKind.CATEGORICAL_ASSOCIATION,
                title=f"Association: {item.left} vs {item.right}",
                text=(
                    f"{item.left} and {item.right} have bias-corrected Cramér's V={item.cramers_v:.2f} "
                    f"across {item.observations:,} complete rows ({item.left_levels}×{item.right_levels} levels)."
                ),
                columns=(item.left, item.right),
                metrics={"cramers_v": item.cramers_v, "observations": item.observations},
            )
        )
    return tuple(findings)
