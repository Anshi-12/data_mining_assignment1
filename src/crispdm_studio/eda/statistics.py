"""Descriptive statistics for dataset-agnostic exploratory analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class NumericDistribution:
    column: str
    count: int
    missing: int
    missing_ratio: float
    mean: float | None
    std: float | None
    minimum: float | None
    q05: float | None
    q25: float | None
    median: float | None
    q75: float | None
    q95: float | None
    maximum: float | None
    iqr: float | None
    skewness: float | None
    outlier_count: int
    outlier_ratio: float
    lower_fence: float | None
    upper_fence: float | None


@dataclass(frozen=True, slots=True)
class CategoryValue:
    label: str
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class CategoricalDistribution:
    column: str
    count: int
    missing: int
    missing_ratio: float
    cardinality: int
    top_values: tuple[CategoryValue, ...]
    dominant_share: float | None
    normalized_entropy: float | None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def summarize_numeric(column: str, series: pd.Series) -> NumericDistribution:
    numeric = pd.to_numeric(series, errors="coerce")
    observed = numeric.dropna()
    rows = len(series)
    missing = int(numeric.isna().sum())
    if observed.empty:
        return NumericDistribution(
            column, 0, missing, missing / max(rows, 1),
            None, None, None, None, None, None, None, None, None, None, None,
            0, 0.0, None, None,
        )

    q05 = _finite(observed.quantile(0.05))
    q25 = _finite(observed.quantile(0.25))
    q75 = _finite(observed.quantile(0.75))
    q95 = _finite(observed.quantile(0.95))
    iqr = None if q25 is None or q75 is None else q75 - q25
    lower = None if iqr is None else q25 - 1.5 * iqr
    upper = None if iqr is None else q75 + 1.5 * iqr
    if lower is None or upper is None:
        outlier_count = 0
    else:
        outlier_count = int(((observed < lower) | (observed > upper)).sum())

    return NumericDistribution(
        column=column,
        count=int(observed.size),
        missing=missing,
        missing_ratio=missing / max(rows, 1),
        mean=_finite(observed.mean()),
        std=_finite(observed.std()),
        minimum=_finite(observed.min()),
        q05=q05,
        q25=q25,
        median=_finite(observed.median()),
        q75=q75,
        q95=q95,
        maximum=_finite(observed.max()),
        iqr=_finite(iqr),
        skewness=_finite(observed.skew()),
        outlier_count=outlier_count,
        outlier_ratio=outlier_count / max(int(observed.size), 1),
        lower_fence=_finite(lower),
        upper_fence=_finite(upper),
    )


def summarize_categorical(
    column: str, series: pd.Series, *, top_n: int = 12
) -> CategoricalDistribution:
    rows = len(series)
    observed = series.dropna().astype("string")
    missing = int(series.isna().sum())
    cardinality = int(observed.nunique())
    if observed.empty:
        return CategoricalDistribution(
            column, 0, missing, missing / max(rows, 1), cardinality, (), None, None
        )

    counts = observed.value_counts(dropna=False)
    top = tuple(
        CategoryValue(str(label), int(count), float(count / len(observed)))
        for label, count in counts.head(top_n).items()
    )
    probabilities = counts.to_numpy(dtype=float) / len(observed)
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    max_entropy = float(np.log(len(probabilities))) if len(probabilities) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return CategoricalDistribution(
        column=column,
        count=int(observed.size),
        missing=missing,
        missing_ratio=missing / max(rows, 1),
        cardinality=cardinality,
        top_values=top,
        dominant_share=top[0].share if top else None,
        normalized_entropy=normalized_entropy,
    )
