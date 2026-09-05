from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ml.training import SearchRun, train_phase3
from server.contracts import RetrainRequest, RetrainResponse
from server.runtime import ModelRuntime


class RetrainAlreadyRunning(RuntimeError):
    pass


def _selected_run(history: tuple[SearchRun, ...], params: dict[str, Any]) -> SearchRun:
    for run in history:
        if run.params == params:
            return run
    raise RuntimeError("Selected hyperparameters are missing from measured search history.")


def run_real_retrain(runtime: ModelRuntime, request: RetrainRequest) -> RetrainResponse:
    """Run the real bounded search and atomically activate the resulting artifacts."""
    if not runtime.retrain_lock.acquire(blocking=False):
        raise RetrainAlreadyRunning("A retraining run is already in progress.")

    temp_dir: Path | None = None
    try:
        parent = runtime.artifact_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=".retrain-artifacts-", dir=parent))
        result = train_phase3(
            rows=request.rows,
            contamination=request.contamination,
            seed=request.seed,
            validation_fraction=request.validation_fraction,
            artifact_dir=temp_dir,
            top_n=min(100, request.rows),
        )
        selected = _selected_run(result.search_history, result.selected_params)

        # This validates the complete temp artifact set before the live directory is touched.
        runtime.replace_artifact_directory(temp_dir)
        temp_dir = None  # directory was renamed into the live path

        manifest = runtime.manifest or {}
        return RetrainResponse(
            status="ok",
            rows_used=result.dataset_rows,
            validation_rows=result.split.validation_rows,
            search_runs=len(result.search_history),
            selected_params=result.selected_params,
            selection_reason=result.selection_reason,
            selected_metrics={
                "roc_auc": selected.roc_auc,
                "pr_auc": selected.pr_auc,
                "precision": selected.precision,
                "recall": selected.recall,
                "f1": selected.f1,
                "fit_seconds": selected.fit_seconds,
                "inference_ms_per_row": selected.inference_ms_per_row,
            },
            artifact_schema_version=str(manifest.get("schema_version", "unknown")),
            message=(
                "Retraining completed using a real bounded 12-run Isolation Forest search; "
                "the validated artifact set was atomically activated."
            ),
        )
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        runtime.retrain_lock.release()
