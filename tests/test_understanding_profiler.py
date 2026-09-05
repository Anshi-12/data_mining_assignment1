from __future__ import annotations

import pandas as pd

from crispdm_studio.models import DatasetOverview, IngestedDataset, UploadMetadata
from crispdm_studio.understanding import profile_dataset


def _ingested(frame: pd.DataFrame) -> IngestedDataset:
    return IngestedDataset(
        dataframe=frame,
        metadata=UploadMetadata("fixture.csv", "text/csv", 100, "utf-8", ","),
        overview=DatasetOverview(
            rows=len(frame),
            columns=len(frame.columns),
            memory_bytes=0,
            missing_cells=int(frame.isna().sum().sum()),
            duplicate_rows=int(frame.duplicated().sum()),
            numeric_columns=0,
            categorical_columns=0,
            datetime_columns=0,
            other_columns=0,
        ),
    )


def test_profiler_computes_dimensions_missingness_constants_and_distributions():
    frame = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, None],
            "status": ["yes", "yes", "no", "yes"],
            "constant": [7, 7, 7, 7],
        }
    )
    result = profile_dataset(_ingested(frame))

    assert result.dataset.rows == 4
    assert result.dataset.columns == 3
    assert result.dataset.missing_cells == 1
    assert "constant" in result.dataset.constant_columns

    score = next(item for item in result.columns if item.name == "score")
    assert score.numeric_summary is not None
    assert score.numeric_summary.median == 2.0
    assert score.missing_ratio == 0.25

    status = next(item for item in result.columns if item.name == "status")
    assert status.top_values[0].label == "yes"
    assert status.top_values[0].count == 3
    assert "3 distinct" not in status.observed_summary


def test_profiler_marks_near_constant_and_duplicate_rows():
    frame = pd.DataFrame({"flag": ["x"] * 19 + ["y"], "value": list(range(20))})
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = profile_dataset(_ingested(frame), near_constant_threshold=0.90)

    assert "flag" in result.dataset.near_constant_columns
    assert result.dataset.duplicate_rows == 1
    assert any(warning.code == "duplicate_rows" for warning in result.quality_warnings)
