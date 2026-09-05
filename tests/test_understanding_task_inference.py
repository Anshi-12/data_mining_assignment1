from __future__ import annotations

import pandas as pd

from crispdm_studio.models import DatasetOverview, IngestedDataset, UploadMetadata
from crispdm_studio.understanding import profile_dataset


def _ingested(frame: pd.DataFrame) -> IngestedDataset:
    return IngestedDataset(
        dataframe=frame,
        metadata=UploadMetadata("fixture.csv", "text/csv", 100, "utf-8", ","),
        overview=DatasetOverview(len(frame), len(frame.columns), 0, 0, 0, 0, 0, 0, 0),
    )


def test_target_candidates_and_leakage_are_cautious_and_structural():
    rows = 60
    frame = pd.DataFrame(
        {
            "customer_id": [f"C-{i:04d}" for i in range(rows)],
            "age": [20 + (i % 35) for i in range(rows)],
            "region": ["north", "south", "east"] * 20,
            "churn": [0, 1] * 30,
            "post_result": ["won", "lost"] * 30,
        }
    )
    result = profile_dataset(_ingested(frame))

    assert "customer_id" in result.identifiers
    candidates = {candidate.column: candidate for candidate in result.target_candidates}
    assert "churn" in candidates
    assert candidates["churn"].task_type.value == "classification"
    assert any(flag.column == "post_result" and flag.risk == "high" for flag in result.leakage_flags)
    assert any(flag.column == "customer_id" for flag in result.leakage_flags)


def test_objective_separates_known_facts_from_inferred_objectives():
    frame = pd.DataFrame({"x": range(40), "label": ["a", "b"] * 20})
    result = profile_dataset(_ingested(frame))

    assert result.objective.known_facts
    assert result.objective.inferred_objectives
    assert any("No business objective" in fact for fact in result.objective.known_facts)
    assert "not claims" in result.objective.honesty_note
