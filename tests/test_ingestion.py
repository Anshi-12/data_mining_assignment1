from __future__ import annotations

import pytest

from crispdm_studio.config import AppConfig
from crispdm_studio.exceptions import (
    DimensionLimitError,
    EmptyFileError,
    UnsupportedFileTypeError,
    ZeroRowError,
)
from crispdm_studio.ingestion import ingest_csv


def ingest(data: bytes, filename: str = "sample.csv", mime: str = "text/csv", **limits):
    config = AppConfig(**limits) if limits else AppConfig()
    return ingest_csv(filename=filename, mime_type=mime, data=data, config=config)


def test_valid_csv_returns_dataset_overview():
    dataset = ingest(b"name,score\nAda,10\nLinus,12\n")
    assert dataset.overview.rows == 2
    assert dataset.overview.columns == 2
    assert list(dataset.dataframe.columns) == ["name", "score"]


def test_empty_file_is_friendly_error():
    with pytest.raises(EmptyFileError, match="empty"):
        ingest(b"")


def test_header_only_csv_is_rejected():
    with pytest.raises(ZeroRowError, match="no data rows"):
        ingest(b"a,b,c\n")


def test_wrong_extension_is_rejected():
    with pytest.raises(UnsupportedFileTypeError, match="CSV"):
        ingest(b"a,b\n1,2\n", filename="sample.txt", mime="text/plain")


def test_duplicate_and_blank_headers_are_normalized():
    dataset = ingest(b"name,name,\nAda,10,x\n")
    assert len(set(dataset.dataframe.columns)) == 3
    assert dataset.dataframe.columns[2].startswith("column_")
    assert dataset.warnings


def test_semicolon_delimiter_is_detected():
    dataset = ingest(b"a;b;c\n1;2;3\n")
    assert dataset.metadata.delimiter == ";"
    assert dataset.overview.columns == 3


def test_single_column_csv_is_valid():
    dataset = ingest(b"name\nAda\nGrace\n")
    assert dataset.overview.rows == 2
    assert dataset.overview.columns == 1


def test_cp1252_encoding_is_supported():
    dataset = ingest("name,city\nRené,Zürich\n".encode("cp1252"))
    assert dataset.dataframe.iloc[0, 0] == "René"


def test_row_limit_is_enforced():
    config = AppConfig(max_rows=1)
    with pytest.raises(DimensionLimitError, match="rows"):
        ingest_csv(filename="x.csv", mime_type="text/csv", data=b"a\n1\n2\n", config=config)


def test_column_limit_is_enforced():
    config = AppConfig(max_columns=1)
    with pytest.raises(DimensionLimitError, match="columns"):
        ingest_csv(filename="x.csv", mime_type="text/csv", data=b"a,b\n1,2\n", config=config)
