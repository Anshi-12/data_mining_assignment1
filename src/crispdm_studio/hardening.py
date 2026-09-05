"""Application-wide hardening helpers and deterministic resource limits."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, MutableMapping, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResourceCaps:
    profile_max_cells: int = 25_000_000
    eda_max_rows: int = 100_000
    eda_max_numeric_relationship_columns: int = 30
    eda_max_categorical_relationship_columns: int = 20
    clustering_max_rows: int = 20_000
    modeling_max_rows: int = 50_000
    permutation_max_rows: int = 5_000
    permutation_repeats: int = 5


CAPS = ResourceCaps()


def deterministic_cap(df: pd.DataFrame, max_rows: int, *, random_state: int = 42) -> tuple[pd.DataFrame, str | None]:
    """Return a deterministic row sample when a phase would otherwise exceed its cap."""
    if len(df) <= max_rows:
        return df, None
    sampled = df.sample(n=max_rows, random_state=random_state).sort_index()
    return sampled, f"Resource cap applied: analyzed {max_rows:,} of {len(df):,} rows using a deterministic sample (random_state={random_state})."


def friendly_exception_message(stage: str) -> str:
    return (
        f"We couldn't complete {stage} safely for this dataset. "
        "No partial result from that stage is being presented as valid; earlier completed results are unchanged."
    )


def guarded_call(stage: str, func: Callable[[], T]) -> tuple[T | None, str | None]:
    """Non-UI safety boundary useful for integration tests and orchestration."""
    try:
        return func(), None
    except Exception:
        logger.exception("Unexpected %s failure", stage)
        return None, friendly_exception_message(stage)
