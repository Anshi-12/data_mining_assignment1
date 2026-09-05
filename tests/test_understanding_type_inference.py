import pandas as pd

from crispdm_studio.understanding.type_inference import FeatureType, infer_feature_type


def test_infers_numeric_boolean_categorical_and_datetime():
    frame = pd.DataFrame(
        {
            "amount": [10.5, 20.0, 30.25, 40.0],
            "active": [True, False, True, True],
            "segment": ["A", "B", "A", "B"],
            "event_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )

    assert infer_feature_type("amount", frame["amount"]).feature_type is FeatureType.NUMERIC
    assert infer_feature_type("active", frame["active"]).feature_type is FeatureType.BOOLEAN
    assert infer_feature_type("segment", frame["segment"]).feature_type is FeatureType.CATEGORICAL
    inferred_date = infer_feature_type("event_date", frame["event_date"])
    assert inferred_date.feature_type is FeatureType.DATETIME
    assert inferred_date.possible_timestamp


def test_detects_id_like_unique_column():
    series = pd.Series([f"CUST-{i:03d}" for i in range(30)], dtype="string")
    inferred = infer_feature_type("customer_id", series)
    assert inferred.feature_type is FeatureType.IDENTIFIER
