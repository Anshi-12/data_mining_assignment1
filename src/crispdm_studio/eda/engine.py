"""Pure Phase 4 EDA engine.

Input: the structurally cleaned dataframe plus the Phase 2 UnderstandingResult.
Output: one structured EDAResult containing descriptive statistics, findings,
reusable Plotly charts, relationship metrics, and explicit skip reasons.

This module performs no imputation, scaling, encoding, model fitting, or train/test
splitting. Missing observations are handled descriptively/pairwise only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crispdm_studio.eda.charts import (
    EDAChart,
    categorical_association_chart,
    categorical_distribution_chart,
    correlation_heatmap,
    missingness_chart,
    numeric_box_chart,
    numeric_categorical_box_chart,
    numeric_distribution_chart,
)
from crispdm_studio.eda.insights import (
    EDAFinding,
    SkippedAnalysis,
    categorical_findings,
    missingness_findings,
    numeric_findings,
    relationship_findings,
)
from crispdm_studio.eda.relationships import (
    CategoricalAssociation,
    CorrelationPair,
    NumericCategoricalRelationship,
    categorical_associations,
    numeric_categorical_relationships,
    numeric_correlations,
)
from crispdm_studio.eda.statistics import (
    CategoricalDistribution,
    NumericDistribution,
    summarize_categorical,
    summarize_numeric,
)
from crispdm_studio.understanding.profiler import UnderstandingResult
from crispdm_studio.understanding.type_inference import FeatureType
from crispdm_studio.hardening import CAPS, deterministic_cap


@dataclass(frozen=True, slots=True)
class EDAResult:
    rows: int
    columns: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_distributions: tuple[NumericDistribution, ...]
    categorical_distributions: tuple[CategoricalDistribution, ...]
    missingness: tuple[tuple[str, int, float], ...]
    correlations: tuple[CorrelationPair, ...]
    numeric_categorical_relationships: tuple[NumericCategoricalRelationship, ...]
    categorical_associations: tuple[CategoricalAssociation, ...]
    findings: tuple[EDAFinding, ...]
    charts: tuple[EDAChart, ...]
    skipped_analyses: tuple[SkippedAnalysis, ...]
    resource_notes: tuple[str, ...] = ()

    def charts_for_section(self, section: str) -> tuple[EDAChart, ...]:
        return tuple(chart for chart in self.charts if chart.section == section)


def _usable_columns(
    dataframe: pd.DataFrame, understanding: UnderstandingResult
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    profiles = {profile.name: profile for profile in understanding.columns}
    numeric: list[str] = []
    categorical: list[str] = []
    for column in dataframe.columns:
        profile = profiles.get(column)
        if pd.api.types.is_numeric_dtype(dataframe[column].dtype):
            numeric.append(column)
        elif profile and profile.feature_type in {FeatureType.NUMERIC}:
            converted = pd.to_numeric(dataframe[column], errors="coerce")
            if converted.notna().sum() >= 2:
                numeric.append(column)
        elif profile and profile.feature_type in {FeatureType.CATEGORICAL, FeatureType.BOOLEAN}:
            categorical.append(column)
    return tuple(numeric), tuple(categorical)


def run_eda(
    dataframe: pd.DataFrame,
    understanding: UnderstandingResult,
    *,
    max_univariate_charts_per_type: int = 12,
    max_relationship_charts: int = 8,
) -> EDAResult:
    """Run adaptive, descriptive EDA without fitting any learned transformer."""
    rows, columns = dataframe.shape
    analysis_df, cap_note = deterministic_cap(dataframe, CAPS.eda_max_rows)
    numeric_columns, categorical_columns = _usable_columns(analysis_df, understanding)
    relationship_numeric = numeric_columns[: CAPS.eda_max_numeric_relationship_columns]
    relationship_categorical = categorical_columns[: CAPS.eda_max_categorical_relationship_columns]
    skipped: list[SkippedAnalysis] = []
    charts: list[EDAChart] = []

    numeric_summaries = tuple(
        summarize_numeric(column, analysis_df[column]) for column in numeric_columns
    )
    if not numeric_columns:
        skipped.append(SkippedAnalysis("Numeric distributions", "No usable numeric columns are present."))
        skipped.append(SkippedAnalysis("Outlier diagnostics", "No usable numeric columns are present."))
    for item in numeric_summaries[:max_univariate_charts_per_type]:
        observed = pd.to_numeric(analysis_df[item.column], errors="coerce").dropna()
        if len(observed) >= 1:
            charts.append(numeric_distribution_chart(item.column, observed))
            charts.append(numeric_box_chart(item.column, observed))

    categorical_summaries = tuple(
        summarize_categorical(column, analysis_df[column]) for column in categorical_columns
    )
    if not categorical_columns:
        skipped.append(
            SkippedAnalysis("Categorical distributions", "No categorical or boolean columns are present.")
        )
    for item in categorical_summaries[:max_univariate_charts_per_type]:
        if item.top_values:
            charts.append(
                categorical_distribution_chart(
                    item.column,
                    [entry.label for entry in item.top_values],
                    [entry.count for entry in item.top_values],
                )
            )

    missing = tuple(
        sorted(
            (
                (column, int(dataframe[column].isna().sum()), float(dataframe[column].isna().mean()))
                for column in dataframe.columns
            ),
            key=lambda item: item[2],
            reverse=True,
        )
    )
    missing_nonzero = tuple(item for item in missing if item[1] > 0)
    if missing_nonzero:
        charts.append(
            missingness_chart(
                [item[0] for item in missing_nonzero],
                [item[1] for item in missing_nonzero],
                [item[2] for item in missing_nonzero],
            )
        )
    else:
        skipped.append(SkippedAnalysis("Missingness visualization", "No missing cells remain after structural cleaning."))

    correlations = numeric_correlations(analysis_df, relationship_numeric)
    if len(relationship_numeric) < 2:
        skipped.append(SkippedAnalysis("Correlation analysis", "At least two usable numeric columns are required."))
    elif not correlations:
        skipped.append(
            SkippedAnalysis(
                "Correlation analysis",
                "Numeric columns exist, but no pair has enough non-missing variation for a Pearson correlation.",
            )
        )
    else:
        corr_frame = analysis_df[list(relationship_numeric)].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
        charts.append(correlation_heatmap(relationship_numeric, corr_frame.to_numpy()))

    numcat = numeric_categorical_relationships(analysis_df, relationship_numeric, relationship_categorical)
    if not relationship_numeric or not relationship_categorical:
        skipped.append(
            SkippedAnalysis(
                "Numeric ↔ categorical relationships",
                "At least one usable numeric and one usable categorical column are required.",
            )
        )
    elif not numcat:
        skipped.append(
            SkippedAnalysis(
                "Numeric ↔ categorical relationships",
                "No numeric/categorical pair met the minimum observations and category-count guardrails.",
            )
        )
    for item in numcat[:max_relationship_charts]:
        pair = analysis_df[[item.numeric, item.categorical]].copy()
        pair[item.numeric] = pd.to_numeric(pair[item.numeric], errors="coerce")
        pair = pair.dropna()
        charts.append(numeric_categorical_box_chart(item.numeric, item.categorical, pair))

    cat_assoc = categorical_associations(analysis_df, relationship_categorical)
    if len(relationship_categorical) < 2:
        skipped.append(
            SkippedAnalysis("Categorical associations", "At least two usable categorical columns are required.")
        )
    elif not cat_assoc:
        skipped.append(
            SkippedAnalysis(
                "Categorical associations",
                "No categorical pair met the minimum observations and cardinality guardrails.",
            )
        )
    for item in cat_assoc[:max_relationship_charts]:
        table = pd.crosstab(analysis_df[item.left], analysis_df[item.right])
        charts.append(categorical_association_chart(item.left, item.right, table))

    findings = (
        numeric_findings(numeric_summaries)
        + categorical_findings(categorical_summaries)
        + missingness_findings(missing_nonzero)
        + relationship_findings(correlations, numcat, cat_assoc)
    )

    return EDAResult(
        rows=rows,
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_distributions=numeric_summaries,
        categorical_distributions=categorical_summaries,
        missingness=missing,
        correlations=correlations,
        numeric_categorical_relationships=numcat,
        categorical_associations=cat_assoc,
        findings=findings,
        charts=tuple(charts),
        skipped_analyses=tuple(skipped),
        resource_notes=tuple(note for note in [cap_note, (f"Relationship analysis capped at {len(relationship_numeric)} numeric and {len(relationship_categorical)} categorical columns." if len(numeric_columns) > len(relationship_numeric) or len(categorical_columns) > len(relationship_categorical) else None)] if note),
    )
