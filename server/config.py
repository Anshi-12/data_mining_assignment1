from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central runtime configuration.

    Phase 1 keeps environment-dependent values here so later ML/training code does
    not scatter ports, random seeds, artifact paths, or thresholds across modules.
    """

    app_name: str = "Anomaly Detection Studio"
    api_prefix: str = "/api"
    host: str = "127.0.0.1"
    port: int = 8006
    frontend_origin: str = "http://localhost:5179"
    random_seed: int = 42
    artifact_schema_version: str = "1.0"
    artifact_dir: str = "server/artifacts"

    model_config = SettingsConfigDict(env_prefix="ANOMALY_", env_file=".env", extra="ignore")


settings = Settings()
