from __future__ import annotations

import numpy as np
import pandas as pd

from crispdm_studio.models import DatasetOverview, IngestedDataset, UploadMetadata
from crispdm_studio.preparation import DecisionStage, prepare_dataset
from crispdm_studio.preparation.outliers import IQRClipper
from crispdm_studio.understanding import profile_dataset


def _understanding(frame: pd.DataFrame):
    ingested = IngestedDataset(
        dataframe=frame,
        metadata=UploadMetadata("fixture.csv", "text/csv", 100, "utf-8", ","),
        overview=DatasetOverview(len(frame), len(frame.columns), 0, 0, 0, 0, 0, 0, 0),
    )
    return profile_dataset(ingested)


def test_structural_cleaning_deduplicates_and_drops_constant_and_identifier():
    rows = 30
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(rows)],
            "amount": list(range(rows)),
            "constant": ["x"] * rows,
            "segment": ["a", "b"] * 15,
        }
    )
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = prepare_dataset(frame, _understanding(frame))

    assert len(result.structural_dataframe) == rows
    assert "customer_id" not in result.structural_dataframe.columns
    assert "constant" not in result.structural_dataframe.columns
    assert result.summary.rows_removed == 1
    assert result.summary.columns_removed == 2


def test_missing_values_are_not_imputed_during_phase3():
    frame = pd.DataFrame({"amount": [1.0, None, 3.0, 4.0], "group": ["a", None, "b", "a"]})
    result = prepare_dataset(frame, _understanding(frame))

    assert result.structural_dataframe["amount"].isna().sum() == 1
    assert result.structural_dataframe["group"].isna().sum() == 1
    deferred = [d for d in result.decisions if d.stage is DecisionStage.DEFERRED]
    assert any("imputation" in d.action.lower() for d in deferred)


def test_lossless_numeric_string_coercion_is_structural():
    # Enough rows make the source strings a categorical profile, while every value is numeric parseable.
    frame = pd.DataFrame({"amount_text": [str((i % 12) + 1) for i in range(30)], "label": ["a", "b"] * 15})
    result = prepare_dataset(frame, _understanding(frame))

    assert pd.api.types.is_numeric_dtype(result.structural_dataframe["amount_text"])
    decision = next(d for d in result.decisions if "numeric string" in d.action.lower())
    assert decision.stage is DecisionStage.APPLIED_NOW
    assert "amount_text" in decision.columns


def test_non_lossless_numeric_like_strings_are_preserved():
    frame = pd.DataFrame({"mixed": ["1", "2", "oops", "3"] * 8, "label": ["a", "b"] * 16})
    result = prepare_dataset(frame, _understanding(frame))
    assert not pd.api.types.is_numeric_dtype(result.structural_dataframe["mixed"])
    assert "oops" in set(result.structural_dataframe["mixed"])


def test_high_cardinality_encoding_is_deferred_and_capped():
    rows = 100
    frame = pd.DataFrame(
        {
            "category": [f"g{i % 30}" for i in range(rows)],
            "amount": np.arange(rows, dtype=float),
            "label": ["yes", "no"] * 50,
        }
    )
    result = prepare_dataset(frame, _understanding(frame))

    assert "category" in result.preprocessing_spec.high_cardinality_columns
    decision = next(d for d in result.decisions if "high-cardinality" in d.action.lower())
    assert decision.stage is DecisionStage.DEFERRED


def test_preprocessing_spec_builds_unfitted_column_transformer_and_fits_on_train_only():
    frame = pd.DataFrame(
        {
            "amount": [1.0, 2.0, None, 4.0, 100.0, 6.0, 7.0, 8.0],
            "segment": ["a", "a", "b", None, "b", "a", "c", "c"],
        }
    )
    result = prepare_dataset(frame, _understanding(frame))
    transformer = result.preprocessing_spec.build_transformer()

    assert not hasattr(transformer, "transformers_")
    train = result.structural_dataframe.iloc[:6]
    transformed = transformer.fit_transform(train)
    assert transformed.shape[0] == 6
    assert hasattr(transformer, "transformers_")


def test_iqr_clipper_learns_bounds_in_fit_and_clips_without_dropping_rows():
    clipper = IQRClipper(multiplier=1.5)
    train = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    transformed = clipper.fit_transform(train)
    assert transformed.shape == train.shape
    assert transformed[-1, 0] < 100.0


def test_scaling_encoding_and_outliers_are_all_deferred():
    frame = pd.DataFrame({"x": range(40), "group": ["a", "b"] * 20})
    result = prepare_dataset(frame, _understanding(frame))
    deferred_actions = {d.action for d in result.deferred_decisions}

    assert "Defer categorical encoding" in deferred_actions
    assert "Defer numeric outlier clipping" in deferred_actions
    assert "Defer numeric scaling" in deferred_actions
