"""Session-state helpers and consistent downstream invalidation."""
from typing import Any, MutableMapping
from crispdm_studio.models import IngestedDataset

DATASET_KEY = "ingested_dataset"
ERROR_KEY = "upload_error"
MODEL_TARGET_KEY = "confirmed_model_target"
MODELING_RESULT_KEY = "modeling_result"
UNDERSTANDING_RESULT_KEY = "understanding_result"
PREPARATION_RESULT_KEY = "preparation_result"
EDA_RESULT_KEY = "eda_result"
CLUSTERING_RESULT_KEY = "clustering_result"
EVALUATION_RESULT_KEY = "evaluation_result"
REPORT_RESULT_KEY = "report_result"

_ORDER = (
    UNDERSTANDING_RESULT_KEY,
    PREPARATION_RESULT_KEY,
    EDA_RESULT_KEY,
    CLUSTERING_RESULT_KEY,
    MODELING_RESULT_KEY,
    EVALUATION_RESULT_KEY,
    REPORT_RESULT_KEY,
)


def initialize_state(state: MutableMapping[str, Any]) -> None:
    state.setdefault(DATASET_KEY, None)
    state.setdefault(ERROR_KEY, None)
    state.setdefault(MODEL_TARGET_KEY, None)
    for key in _ORDER:
        state.setdefault(key, None)


def invalidate_from(state: MutableMapping[str, Any], key: str) -> None:
    """Clear a stage and every result that can depend on it."""
    if key not in _ORDER:
        return
    start = _ORDER.index(key)
    for downstream in _ORDER[start:]:
        state[downstream] = None
    if key in {UNDERSTANDING_RESULT_KEY, PREPARATION_RESULT_KEY, MODELING_RESULT_KEY}:
        state[MODEL_TARGET_KEY] = None if key != MODELING_RESULT_KEY else state.get(MODEL_TARGET_KEY)


def set_dataset(state: MutableMapping[str, Any], dataset: IngestedDataset) -> None:
    state[DATASET_KEY] = dataset
    state[ERROR_KEY] = None
    state[MODEL_TARGET_KEY] = None
    for key in _ORDER:
        state[key] = None


def clear_dataset(state: MutableMapping[str, Any]) -> None:
    state[DATASET_KEY] = None
    state[MODEL_TARGET_KEY] = None
    for key in _ORDER:
        state[key] = None


def set_error(state: MutableMapping[str, Any], message: str) -> None:
    """Upload failure invalidates all previous pipeline state to prevent stale results."""
    clear_dataset(state)
    state[ERROR_KEY] = message
