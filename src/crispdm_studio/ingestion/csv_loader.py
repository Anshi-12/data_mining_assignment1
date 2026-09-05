"""Dataset-agnostic CSV ingestion service."""

from __future__ import annotations

import csv
import io
import logging

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from crispdm_studio.config import CONFIG, AppConfig
from crispdm_studio.exceptions import (
    CSVReadError,
    CrispDMError,
    DelimiterDetectionError,
    EmptyFileError,
    NoColumnsError,
)
from crispdm_studio.ingestion.encoding import detect_encoding
from crispdm_studio.ingestion.schema import make_headers_unique
from crispdm_studio.ingestion.validator import validate_dataframe, validate_upload_identity
from crispdm_studio.models import DatasetOverview, IngestedDataset, UploadMetadata

logger = logging.getLogger(__name__)

COMMON_DELIMITERS = ",;\t|"


def _detect_delimiter(text: str) -> str:
    sample = text[: CONFIG.delimiter_sample_chars]
    meaningful = [line for line in sample.splitlines() if line.strip()]
    if not meaningful:
        raise EmptyFileError()

    candidate = "\n".join(meaningful[:50])
    try:
        dialect = csv.Sniffer().sniff(candidate, delimiters=COMMON_DELIMITERS)
        return dialect.delimiter
    except csv.Error:
        counts = {delimiter: candidate.count(delimiter) for delimiter in COMMON_DELIMITERS}
        delimiter, count = max(counts.items(), key=lambda item: item[1])
        if count == 0:
            # A valid single-column CSV has no delimiter; comma is harmless for pandas.
            if len(meaningful) >= 1:
                return ","
            raise DelimiterDetectionError()
        return delimiter


def _read_header(text: str, delimiter: str) -> tuple[list[str], int]:
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        header = next(reader)
        return header, reader.line_num
    except StopIteration as exc:
        raise NoColumnsError() from exc
    except csv.Error as exc:
        raise CSVReadError(detail=str(exc)) from exc


def _read_dataframe(
    text: str, delimiter: str, headers: list[str], header_line_count: int
) -> pd.DataFrame:
    try:
        return pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            engine="python",
            header=None,
            names=headers,
            skiprows=header_line_count,
            dtype_backend="numpy_nullable",
            on_bad_lines="error",
        )
    except EmptyDataError as exc:
        raise NoColumnsError() from exc
    except (ParserError, UnicodeError, csv.Error, ValueError) as exc:
        raise CSVReadError(detail=str(exc)) from exc


def _build_overview(df: pd.DataFrame) -> DatasetOverview:
    type_names = df.dtypes.astype(str)
    numeric = sum(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes)
    datetime = sum(pd.api.types.is_datetime64_any_dtype(dtype) for dtype in df.dtypes)
    categorical = sum(
        pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(dtype)
        for dtype in df.dtypes
    )
    other = len(type_names) - numeric - datetime - categorical

    return DatasetOverview(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        memory_bytes=int(df.memory_usage(index=True, deep=True).sum()),
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=int(numeric),
        categorical_columns=int(categorical),
        datetime_columns=int(datetime),
        other_columns=int(other),
    )


def ingest_csv(
    *,
    filename: str,
    mime_type: str,
    data: bytes,
    config: AppConfig = CONFIG,
) -> IngestedDataset:
    """Validate, decode, parse, normalize, and summarize an uploaded CSV.

    Raises only CrispDMError subclasses for expected user-facing failures.
    Unexpected exceptions are logged and converted to a safe CSVReadError.
    """
    try:
        validate_upload_identity(filename, mime_type, data, config)
        encoding = detect_encoding(data, config.encoding_sample_bytes)

        try:
            text = data.decode(encoding, errors="strict")
        except (UnicodeDecodeError, LookupError) as exc:
            raise CSVReadError("We couldn't decode this CSV using the detected text encoding.") from exc

        delimiter = _detect_delimiter(text)
        raw_headers, header_line_count = _read_header(text, delimiter)
        headers, header_changes, warnings = make_headers_unique(raw_headers)
        df = _read_dataframe(text, delimiter, headers, header_line_count)
        validate_dataframe(df, config)

        metadata = UploadMetadata(
            filename=filename,
            mime_type=mime_type or "unknown",
            size_bytes=len(data),
            encoding=encoding,
            delimiter=delimiter,
        )

        return IngestedDataset(
            dataframe=df,
            metadata=metadata,
            overview=_build_overview(df),
            warnings=warnings,
            header_changes=header_changes,
        )
    except CrispDMError:
        raise
    except Exception as exc:  # final defensive boundary: never leak tracebacks to users
        logger.exception("Unexpected CSV ingestion failure for %s", filename)
        raise CSVReadError(
            "We couldn't read this CSV safely. Please check the file and try again.",
            detail=str(exc),
        ) from exc
