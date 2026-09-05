from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import joblib

from ml.data_loader import FEATURE_NAMES
from server.artifact_schema import ArtifactLayout
from server.services.scoring import ProductionScorer

logger = logging.getLogger(__name__)


class ArtifactValidationError(RuntimeError):
    pass


class ModelRuntime:
    """Owns the validated, in-memory production artifact set."""

    def __init__(self, artifact_dir: str | Path, expected_schema_version: str = "1.0") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.expected_schema_version = expected_schema_version
        self._state_lock = threading.RLock()
        self._retrain_lock = threading.Lock()
        self._scorer: ProductionScorer | None = None
        self._manifest: dict[str, Any] | None = None
        self._reason: str | None = None
        self.reload()

    @property
    def retrain_lock(self) -> threading.Lock:
        return self._retrain_lock

    @property
    def ready(self) -> bool:
        return self._scorer is not None

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def manifest(self) -> dict[str, Any] | None:
        return self._manifest

    @property
    def scorer(self) -> ProductionScorer:
        if self._scorer is None:
            raise ArtifactValidationError(self._reason or "ML artifacts are unavailable.")
        return self._scorer

    def _validate_manifest(self, manifest: dict[str, Any], layout: ArtifactLayout) -> None:
        version = str(manifest.get("schema_version", ""))
        if version != self.expected_schema_version:
            raise ArtifactValidationError(
                f"Artifact schema mismatch: expected {self.expected_schema_version}, found {version or 'missing'}."
            )
        feature_order = manifest.get("feature_order")
        if feature_order != list(FEATURE_NAMES):
            raise ArtifactValidationError(
                "Artifact feature-order mismatch; retrain artifacts with the current canonical telemetry schema."
            )
        feature_metadata = manifest.get("feature_metadata")
        if not isinstance(feature_metadata, list) or [item.get("feature") for item in feature_metadata] != list(FEATURE_NAMES):
            raise ArtifactValidationError(
                "Artifact feature metadata is missing or incompatible with the canonical telemetry schema."
            )
        required_metadata = {"feature", "api_field", "unit", "normal_range", "description", "min", "max", "step", "default"}
        if any(not required_metadata.issubset(item) for item in feature_metadata):
            raise ArtifactValidationError("Artifact feature metadata is incomplete; retrain artifacts.")
        if manifest.get("production_model") != "IsolationForest":
            raise ArtifactValidationError("Unsupported production model in manifest.")
        required = [
            layout.scaler,
            layout.detector,
            layout.calibration,
            layout.search_history,
            layout.benchmarks,
            layout.manifold,
            layout.top_anomalies,
        ]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            raise ArtifactValidationError("Missing ML artifacts: " + ", ".join(missing))

    def reload(self) -> None:
        layout = ArtifactLayout(self.artifact_dir)
        try:
            if not layout.manifest.exists():
                raise ArtifactValidationError("manifest.json is missing; run Phase 3 training first.")
            manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
            self._validate_manifest(manifest, layout)
            scaler = joblib.load(layout.scaler)
            model = joblib.load(layout.detector)
            calibration = json.loads(layout.calibration.read_text(encoding="utf-8"))
            scorer = ProductionScorer(scaler, model, calibration, manifest)
        except Exception as exc:
            logger.warning("ML artifacts unavailable: %s", exc)
            with self._state_lock:
                self._scorer = None
                self._manifest = None
                self._reason = str(exc)
            return
        with self._state_lock:
            self._scorer = scorer
            self._manifest = manifest
            self._reason = None

    def read_json_artifact(self, filename: str) -> Any:
        with self._state_lock:
            if not self.ready:
                raise ArtifactValidationError(self.reason or "ML artifacts unavailable.")
            path = self.artifact_dir / filename
            if not path.exists():
                raise ArtifactValidationError(f"Stored artifact '{filename}' is missing.")
            return json.loads(path.read_text(encoding="utf-8"))

    def replace_artifact_directory(self, prepared_dir: str | Path) -> None:
        """Swap a fully validated artifact directory into service with rollback.

        The caller must prepare and validate ``prepared_dir`` first. Reads are blocked
        during the short directory swap. The old directory is restored if activation
        fails, so a failed retrain cannot corrupt the live artifact set.
        """
        prepared = Path(prepared_dir)
        probe = ModelRuntime(prepared, self.expected_schema_version)
        if not probe.ready:
            raise ArtifactValidationError(probe.reason or "Prepared artifact set is invalid.")

        backup = self.artifact_dir.parent / f".{self.artifact_dir.name}.backup-{uuid.uuid4().hex}"
        with self._state_lock:
            had_live = self.artifact_dir.exists()
            try:
                if had_live:
                    os.replace(self.artifact_dir, backup)
                os.replace(prepared, self.artifact_dir)
                self.reload()
                if not self.ready:
                    raise ArtifactValidationError(self.reason or "New artifacts failed activation.")
            except Exception:
                if self.artifact_dir.exists():
                    shutil.rmtree(self.artifact_dir, ignore_errors=True)
                if had_live and backup.exists():
                    os.replace(backup, self.artifact_dir)
                self.reload()
                raise
            else:
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
