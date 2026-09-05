"""Shared domain models used by ingestion, UI, and future analysis phases."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class UploadMetadata:
    filename: str
    mime_type: str
    size_bytes: int
    encoding: str
    delimiter: str


@dataclass(frozen=True, slots=True)
class DatasetOverview:
    rows: int
    columns: int
    memory_bytes: int
    missing_cells: int
    duplicate_rows: int
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int
    other_columns: int


@dataclass(slots=True)
class IngestedDataset:
    dataframe: pd.DataFrame
    metadata: UploadMetadata
    overview: DatasetOverview
    warnings: list[str] = field(default_factory=list)
    header_changes: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
