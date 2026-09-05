"""Pre- and post-read validation for CSV uploads."""

from pathlib import Path

import pandas as pd

from crispdm_studio.config import AppConfig
from crispdm_studio.exceptions import (
    DimensionLimitError,
    EmptyFileError,
    NoColumnsError,
    UnsupportedFileTypeError,
    ZeroRowError,
)


def validate_upload_identity(filename: str, mime_type: str, data: bytes, config: AppConfig) -> None:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in config.allowed_extensions:
        raise UnsupportedFileTypeError(
            f"Please upload a CSV file. '{suffix or 'no extension'}' files are not supported."
        )

    if mime_type and mime_type.lower() not in config.allowed_mime_types:
        raise UnsupportedFileTypeError(
            f"The uploaded file type '{mime_type}' is not supported. Please upload a CSV file."
        )

    if not data or not data.strip(b"\x00\t\n\r "):
        raise EmptyFileError()

    if len(data) > config.max_upload_bytes:
        raise DimensionLimitError(
            f"This file is {len(data) / (1024 * 1024):.1f} MB. The upload limit is {config.max_upload_mb} MB."
        )


def validate_dataframe(df: pd.DataFrame, config: AppConfig) -> None:
    if df.shape[1] == 0:
        raise NoColumnsError()
    if df.shape[0] == 0:
        raise ZeroRowError()
    if df.shape[0] > config.max_rows:
        raise DimensionLimitError(
            f"This CSV contains {df.shape[0]:,} rows; the current limit is {config.max_rows:,}."
        )
    if df.shape[1] > config.max_columns:
        raise DimensionLimitError(
            f"This CSV contains {df.shape[1]:,} columns; the current limit is {config.max_columns:,}."
        )
