from __future__ import annotations

from functools import lru_cache

from server.config import settings
from server.runtime import ModelRuntime


@lru_cache(maxsize=1)
def get_runtime() -> ModelRuntime:
    return ModelRuntime(settings.artifact_dir, settings.artifact_schema_version)


def reset_runtime() -> ModelRuntime:
    get_runtime.cache_clear()
    return get_runtime()
