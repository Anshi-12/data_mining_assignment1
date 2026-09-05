"""Categorical encoding policy for deferred preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EncodingStrategy:
    method: str = "one_hot"
    handle_unknown: str = "infrequent_if_exist"
    min_frequency: float = 0.01
    max_categories: int = 50

    @property
    def rationale(self) -> str:
        return (
            "One-hot encoding preserves nominal categories. Rare/high-cardinality levels are grouped by "
            "OneHotEncoder using training-fold frequencies only, with unknown levels handled safely at transform time."
        )
