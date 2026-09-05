"""Dataset-agnostic semantic type inference for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

import numpy as np
import pandas as pd


class FeatureType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class TypeInference:
    feature_type: FeatureType
    confidence: float
    reason: str
    possible_timestamp: bool = False


_ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|identifier|uuid|guid|key$|_key$)", re.IGNORECASE)
_DATE_NAME_RE = re.compile(r"date|time|timestamp|datetime|created|updated|year|month|day", re.IGNORECASE)


def _looks_identifier_like_name(name: str) -> bool:
    return bool(_ID_NAME_RE.search(name.strip()))


def _datetime_parse_ratio(series: pd.Series, sample_size: int = 200) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.astype("string").head(sample_size)
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError, OverflowError):
        return 0.0
    return float(parsed.notna().mean())


def infer_feature_type(name: str, series: pd.Series) -> TypeInference:
    """Infer a useful analytical type without mutating the source series."""
    non_null = series.dropna()
    if non_null.empty:
        return TypeInference(FeatureType.EMPTY, 1.0, "All values are missing.")

    row_count = len(series)
    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = unique_count / max(len(non_null), 1)

    if pd.api.types.is_bool_dtype(series.dtype):
        return TypeInference(FeatureType.BOOLEAN, 1.0, "Stored as boolean values.")

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return TypeInference(
            FeatureType.DATETIME, 1.0, "Stored as datetime values.", possible_timestamp=True
        )

    if pd.api.types.is_numeric_dtype(series.dtype):
        if _looks_identifier_like_name(name) and unique_ratio >= 0.9:
            return TypeInference(
                FeatureType.IDENTIFIER,
                0.95,
                "The column name is ID-like and values are almost entirely unique.",
            )
        if unique_ratio >= 0.98 and row_count >= 20:
            values = pd.to_numeric(non_null, errors="coerce")
            if values.notna().all():
                ordered = values.sort_values().to_numpy(dtype=float)
                if len(ordered) > 1:
                    diffs = np.diff(ordered)
                    if np.allclose(diffs, diffs[0]) and abs(float(diffs[0])) in (1.0,):
                        return TypeInference(
                            FeatureType.IDENTIFIER,
                            0.75,
                            "Values are nearly unique and form a sequential integer-like key.",
                        )
        return TypeInference(FeatureType.NUMERIC, 0.99, "Stored as numeric values.")

    datetime_ratio = _datetime_parse_ratio(series)
    date_named = bool(_DATE_NAME_RE.search(name))
    if datetime_ratio >= 0.95 and (date_named or unique_count > 2):
        return TypeInference(
            FeatureType.DATETIME,
            min(0.99, datetime_ratio),
            f"{datetime_ratio:.0%} of sampled non-missing values parse as datetimes.",
            possible_timestamp=True,
        )

    if _looks_identifier_like_name(name) and unique_ratio >= 0.9:
        return TypeInference(
            FeatureType.IDENTIFIER,
            0.95,
            "The column name is ID-like and values are almost entirely unique.",
        )

    if unique_ratio >= 0.98 and row_count >= 20:
        avg_length = float(non_null.astype("string").str.len().mean())
        if avg_length <= 64:
            return TypeInference(
                FeatureType.IDENTIFIER,
                0.70,
                "Values are almost entirely unique, which is consistent with a record key.",
            )

    if unique_count <= max(20, int(0.05 * row_count)):
        return TypeInference(
            FeatureType.CATEGORICAL,
            0.90,
            f"Only {unique_count} distinct non-missing values occur across {len(non_null)} observations.",
        )

    text = non_null.astype("string")
    avg_length = float(text.str.len().mean())
    whitespace_ratio = float(text.str.contains(r"\s", regex=True).mean())
    if avg_length >= 30 or whitespace_ratio >= 0.5:
        return TypeInference(
            FeatureType.TEXT,
            0.85,
            f"Values are free-form strings (average length {avg_length:.1f} characters).",
        )

    return TypeInference(
        FeatureType.CATEGORICAL,
        0.70,
        "Values are discrete strings and are treated as categorical for analysis.",
    )
