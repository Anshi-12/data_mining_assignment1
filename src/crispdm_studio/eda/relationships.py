"""Relationship statistics for Phase 4 EDA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


@dataclass(frozen=True, slots=True)
class CorrelationPair:
    left: str
    right: str
    correlation: float
    observations: int


@dataclass(frozen=True, slots=True)
class NumericCategoricalRelationship:
    numeric: str
    categorical: str
    eta_squared: float
    groups: int
    observations: int
    group_medians: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class CategoricalAssociation:
    left: str
    right: str
    cramers_v: float
    observations: int
    left_levels: int
    right_levels: int


def numeric_correlations(
    dataframe: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    *,
    min_observations: int = 3,
) -> tuple[CorrelationPair, ...]:
    results: list[CorrelationPair] = []
    for i, left in enumerate(numeric_columns):
        for right in numeric_columns[i + 1 :]:
            pair = dataframe[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < min_observations:
                continue
            if pair[left].nunique() <= 1 or pair[right].nunique() <= 1:
                continue
            value = pair[left].corr(pair[right], method="pearson")
            if pd.notna(value):
                results.append(CorrelationPair(left, right, float(value), len(pair)))
    return tuple(sorted(results, key=lambda item: abs(item.correlation), reverse=True))


def _eta_squared(values: pd.Series, groups: pd.Series) -> float | None:
    data = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups}).dropna()
    if len(data) < 3 or data["group"].nunique() < 2:
        return None
    grand_mean = float(data["value"].mean())
    total_ss = float(((data["value"] - grand_mean) ** 2).sum())
    if total_ss <= 0:
        return None
    between_ss = 0.0
    for _, group in data.groupby("group", observed=True):
        between_ss += len(group) * (float(group["value"].mean()) - grand_mean) ** 2
    return max(0.0, min(1.0, between_ss / total_ss))


def numeric_categorical_relationships(
    dataframe: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    *,
    max_categories: int = 12,
    min_observations: int = 5,
) -> tuple[NumericCategoricalRelationship, ...]:
    results: list[NumericCategoricalRelationship] = []
    for categorical in categorical_columns:
        cardinality = int(dataframe[categorical].dropna().nunique())
        if cardinality < 2 or cardinality > max_categories:
            continue
        for numeric in numeric_columns:
            pair = dataframe[[numeric, categorical]].copy()
            pair[numeric] = pd.to_numeric(pair[numeric], errors="coerce")
            pair = pair.dropna()
            if len(pair) < min_observations or pair[numeric].nunique() <= 1:
                continue
            eta = _eta_squared(pair[numeric], pair[categorical])
            if eta is None:
                continue
            medians = tuple(
                (str(label), float(value))
                for label, value in pair.groupby(categorical, observed=True)[numeric]
                .median()
                .sort_values(ascending=False)
                .items()
            )
            results.append(
                NumericCategoricalRelationship(
                    numeric=numeric,
                    categorical=categorical,
                    eta_squared=float(eta),
                    groups=cardinality,
                    observations=len(pair),
                    group_medians=medians,
                )
            )
    return tuple(sorted(results, key=lambda item: item.eta_squared, reverse=True))


def _cramers_v(table: pd.DataFrame) -> float | None:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    n = int(table.to_numpy().sum())
    if n <= 1:
        return None
    try:
        chi2 = float(chi2_contingency(table, correction=False)[0])
    except ValueError:
        return None
    phi2 = chi2 / n
    r, k = table.shape
    correction = ((k - 1) * (r - 1)) / max(n - 1, 1)
    phi2_corr = max(0.0, phi2 - correction)
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denominator = min(k_corr - 1, r_corr - 1)
    if denominator <= 0:
        return None
    return float(np.sqrt(phi2_corr / denominator))


def categorical_associations(
    dataframe: pd.DataFrame,
    categorical_columns: tuple[str, ...],
    *,
    max_categories: int = 20,
    min_observations: int = 5,
) -> tuple[CategoricalAssociation, ...]:
    eligible = tuple(
        column
        for column in categorical_columns
        if 2 <= dataframe[column].dropna().nunique() <= max_categories
    )
    results: list[CategoricalAssociation] = []
    for i, left in enumerate(eligible):
        for right in eligible[i + 1 :]:
            pair = dataframe[[left, right]].dropna()
            if len(pair) < min_observations:
                continue
            table = pd.crosstab(pair[left], pair[right])
            value = _cramers_v(table)
            if value is None:
                continue
            results.append(
                CategoricalAssociation(
                    left=left,
                    right=right,
                    cramers_v=value,
                    observations=len(pair),
                    left_levels=table.shape[0],
                    right_levels=table.shape[1],
                )
            )
    return tuple(sorted(results, key=lambda item: item.cramers_v, reverse=True))
