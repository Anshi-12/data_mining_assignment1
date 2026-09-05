"""Missing-value strategy selection for deferred fold-safe preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class MissingValueStrategy:
    numeric_strategy: str = "median"
    categorical_strategy: str = "most_frequent"
    categorical_fill_value: str = "__MISSING__"


def describe_missing_strategy(dataframe: pd.DataFrame, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> str:
    numeric_missing = int(dataframe[list(numeric)].isna().sum().sum()) if numeric else 0
    categorical_missing = int(dataframe[list(categorical)].isna().sum().sum()) if categorical else 0
    return (
        f"Numeric median imputation is required for {numeric_missing:,} missing numeric cell(s); "
        f"categorical most-frequent imputation is required for {categorical_missing:,} missing categorical cell(s)."
    )
