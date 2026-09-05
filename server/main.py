from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.config import settings
from server.contracts import HealthResponse, RetrainRequest, RetrainResponse, ScoreResult, TelemetryVector
from server.dependencies import get_runtime
from server.logging_config import configure_logging
from server.runtime import ArtifactValidationError
from server.services.retraining import RetrainAlreadyRunning, run_real_retrain

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5179"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ArtifactValidationError)
async def artifact_error_handler(request: Request, exc: ArtifactValidationError) -> JSONResponse:
    logger.warning("Artifact error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "The service hit an unexpected error. Check the server log for details."},
    )


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    runtime = get_runtime()
    manifest = runtime.manifest or {}
    return HealthResponse(
        service=settings.app_name,
        ml_ready=runtime.ready,
        ml_reason=runtime.reason,
        scorer=runtime.scorer.name if runtime.ready else "unavailable",
        model_type=manifest.get("production_model"),
        selected_hyperparameters=manifest.get("selected_params"),
        artifact_schema_version=str(manifest.get("schema_version", settings.artifact_schema_version)),
    )




@app.get(f"{settings.api_prefix}/metadata")
def metadata():
    runtime = get_runtime()
    manifest = runtime.manifest
    if not runtime.ready or manifest is None:
        raise ArtifactValidationError(runtime.reason or "ML artifacts are unavailable.")
    return {
        "schema_version": manifest["schema_version"],
        "source": manifest["source"],
        "dataset_rows": manifest["dataset_rows"],
        "training_rows": manifest["training_rows"],
        "validation_rows": manifest["validation_rows"],
        "feature_order": manifest["feature_order"],
        "feature_metadata": manifest["feature_metadata"],
        "ground_truth_usage": manifest["ground_truth_usage"],
        "production_model": manifest["production_model"],
        "selected_params": manifest["selected_params"],
        "seed": manifest["seed"],
        "retrain": {
            "default_rows": min(int(manifest["dataset_rows"]), 2_000),
            "min_rows": 500,
            "max_rows": 5_000,
            "rows_step": 100,
            "default_contamination": float(manifest["selected_params"]["contamination"]),
            "min_contamination": 0.001,
            "max_contamination": 0.199,
            "contamination_step": 0.001,
            "default_seed": int(manifest["seed"]),
            "default_validation_fraction": float(manifest["validation_rows"]) / float(manifest["dataset_rows"]),
            "min_validation_fraction": 0.1,
            "max_validation_fraction": 0.4,
            "validation_step": 0.01,
        },
    }


@app.post(f"{settings.api_prefix}/anomaly/score", response_model=ScoreResult)
def score_anomaly(observation: TelemetryVector) -> ScoreResult:
    return get_runtime().scorer.score(observation)


@app.get(f"{settings.api_prefix}/benchmarks")
def benchmarks():
    return get_runtime().read_json_artifact("benchmarks.json")


@app.get(f"{settings.api_prefix}/manifold")
def manifold():
    return get_runtime().read_json_artifact("manifold.json")


@app.get(f"{settings.api_prefix}/anomalies/top")
def top_anomalies():
    return get_runtime().read_json_artifact("top_anomalies.json")


@app.get(f"{settings.api_prefix}/autoresearch/history")
def search_history():
    return get_runtime().read_json_artifact("search_history.json")


@app.get(f"{settings.api_prefix}/retrain/options")
def retrain_options():
    return {
        "default_rows": 2_000,
        "min_rows": 500,
        "max_rows": 5_000,
        "rows_step": 100,
        "default_contamination": 0.035,
        "min_contamination": 0.001,
        "max_contamination": 0.199,
        "contamination_step": 0.001,
        "default_seed": settings.random_seed,
        "default_validation_fraction": 0.25,
        "min_validation_fraction": 0.1,
        "max_validation_fraction": 0.4,
        "validation_step": 0.01,
    }


@app.post(f"{settings.api_prefix}/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest) -> RetrainResponse:
    try:
        return run_real_retrain(get_runtime(), request)
    except RetrainAlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=settings.host, port=settings.port, reload=True)
