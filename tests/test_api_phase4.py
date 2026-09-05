from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ml.data_loader import FEATURE_NAMES, generate_anomaly_dataset
from ml.training import load_production_artifacts, score_fixed_vector
from server.contracts import RetrainRequest
from server.main import app
from server.runtime import ModelRuntime
from server.services.retraining import run_real_retrain
from server.services.scoring import FEATURE_TO_API

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "server" / "artifacts"


def _api_payload_from_row(row) -> dict[str, float]:
    return {FEATURE_TO_API[name]: float(row[name]) for name in FEATURE_NAMES}


def test_api_live_score_matches_offline_round_trip() -> None:
    scaler, model, calibration, _ = load_production_artifacts(ARTIFACT_DIR)
    dataset = generate_anomaly_dataset(n_samples=2_000, random_state=42)
    row = dataset.features.iloc[11]
    offline_raw, offline_threat = score_fixed_vector(
        row.to_numpy(dtype=float),
        scaler=scaler,
        model=model,
        calibration_payload=calibration,
    )

    response = TestClient(app).post("/api/anomaly/score", json=_api_payload_from_row(row))
    assert response.status_code == 200
    body = response.json()
    assert np.isclose(body["raw_anomaly_score"], offline_raw, atol=1e-12)
    assert np.isclose(body["threat_index"], offline_threat, atol=1e-12)
    assert body["model_type"] == "IsolationForest"
    assert body["attribution_method"] == "separate_robust_iqr_deviation"
    assert len(body["attribution"]) == 10


def test_schema_mismatch_is_cleanly_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(ARTIFACT_DIR, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_order"] = list(reversed(manifest["feature_order"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    runtime = ModelRuntime(copied, "1.0")
    assert runtime.ready is False
    assert "feature-order mismatch" in (runtime.reason or "")


def test_artifact_endpoints_return_exact_stored_json() -> None:
    client = TestClient(app)
    mapping = {
        "/api/benchmarks": "benchmarks.json",
        "/api/manifold": "manifold.json",
        "/api/anomalies/top": "top_anomalies.json",
        "/api/autoresearch/history": "search_history.json",
    }
    for endpoint, filename in mapping.items():
        expected = json.loads((ARTIFACT_DIR / filename).read_text(encoding="utf-8"))
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json() == expected


def test_real_retrain_activates_new_artifacts_and_runtime_health_state(tmp_path: Path) -> None:
    live = tmp_path / "artifacts"
    shutil.copytree(ARTIFACT_DIR, live)
    old_manifest = json.loads((live / "manifest.json").read_text(encoding="utf-8"))
    runtime = ModelRuntime(live, "1.0")
    assert runtime.ready

    result = run_real_retrain(runtime, RetrainRequest(rows=500, seed=43))
    assert result.status == "ok"
    assert result.rows_used == 500
    assert result.search_runs == 12
    assert all(key in result.selected_metrics for key in ("roc_auc", "pr_auc", "f1"))
    assert runtime.ready
    assert runtime.manifest is not None
    assert runtime.manifest["dataset_rows"] == 500
    assert runtime.manifest["seed"] == 43
    assert runtime.manifest["generated_at_utc"] != old_manifest["generated_at_utc"]
    assert (live / "robust_scaler.joblib").exists()
    assert (live / "isolation_forest.joblib").exists()


def test_invalid_prepared_artifact_set_does_not_replace_live(tmp_path: Path) -> None:
    live = tmp_path / "artifacts"
    shutil.copytree(ARTIFACT_DIR, live)
    before = (live / "manifest.json").read_bytes()
    runtime = ModelRuntime(live, "1.0")

    invalid = tmp_path / "invalid"
    shutil.copytree(ARTIFACT_DIR, invalid)
    (invalid / "isolation_forest.joblib").unlink()
    try:
        runtime.replace_artifact_directory(invalid)
    except Exception:
        pass
    else:
        raise AssertionError("Invalid artifacts should not activate")

    assert (live / "manifest.json").read_bytes() == before
    assert runtime.ready


def test_retrain_endpoint_runs_real_search_and_health_reflects_new_artifacts(tmp_path: Path, monkeypatch) -> None:
    live = tmp_path / "api-artifacts"
    shutil.copytree(ARTIFACT_DIR, live)
    runtime = ModelRuntime(live, "1.0")
    monkeypatch.setattr("server.main.get_runtime", lambda: runtime)
    client = TestClient(app)

    response = client.post("/api/retrain", json={"rows": 500, "seed": 44})
    assert response.status_code == 200
    body = response.json()
    assert body["rows_used"] == 500
    assert body["search_runs"] == 12
    assert body["selected_metrics"]["pr_auc"] >= 0.0

    health = client.get("/api/health").json()
    assert health["ml_ready"] is True
    assert health["selected_hyperparameters"] == runtime.manifest["selected_params"]
    assert runtime.manifest["seed"] == 44


def test_concurrent_retrain_is_rejected_with_409(tmp_path: Path, monkeypatch) -> None:
    live = tmp_path / "locked-artifacts"
    shutil.copytree(ARTIFACT_DIR, live)
    runtime = ModelRuntime(live, "1.0")
    monkeypatch.setattr("server.main.get_runtime", lambda: runtime)
    runtime.retrain_lock.acquire()
    try:
        response = TestClient(app).post("/api/retrain", json={"rows": 500})
        assert response.status_code == 409
        assert "already in progress" in response.json()["detail"]
    finally:
        runtime.retrain_lock.release()


def test_health_reports_missing_artifacts_without_crashing(tmp_path: Path, monkeypatch) -> None:
    runtime = ModelRuntime(tmp_path / "missing", "1.0")
    monkeypatch.setattr("server.main.get_runtime", lambda: runtime)
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ml_ready"] is False
    assert body["scorer"] == "unavailable"
    assert "manifest.json is missing" in body["ml_reason"]


def test_metadata_endpoint_is_manifest_driven() -> None:
    response = TestClient(app).get("/api/metadata")
    assert response.status_code == 200
    body = response.json()
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert body["feature_order"] == manifest["feature_order"]
    assert body["feature_metadata"] == manifest["feature_metadata"]
    assert body["selected_params"] == manifest["selected_params"]
    assert body["source"] == "locally synthesized deterministic server telemetry"


def test_feature_metadata_mismatch_is_cleanly_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(ARTIFACT_DIR, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_metadata"][0]["feature"] = "WrongFeature"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = ModelRuntime(copied, "1.0")
    assert runtime.ready is False
    assert "feature metadata" in (runtime.reason or "")


def test_retrain_options_are_available_without_ml_artifacts(tmp_path: Path, monkeypatch) -> None:
    runtime = ModelRuntime(tmp_path / "missing", "1.0")
    monkeypatch.setattr("server.main.get_runtime", lambda: runtime)
    client = TestClient(app)
    assert client.get("/api/health").json()["ml_ready"] is False
    response = client.get("/api/retrain/options")
    assert response.status_code == 200
    body = response.json()
    assert body["min_rows"] <= body["default_rows"] <= body["max_rows"]
    assert body["default_seed"] == 42
