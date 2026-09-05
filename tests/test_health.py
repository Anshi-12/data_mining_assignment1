from fastapi.testclient import TestClient

from server.main import app


def test_health_contract() -> None:
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["phase"] == 6
    assert payload["ml_ready"] is True
    assert payload["scorer"] == "persisted-robustscaler-isolationforest"
    assert payload["model_type"] == "IsolationForest"
    assert payload["artifact_schema_version"] == "1.0"
    assert isinstance(payload["selected_hyperparameters"], dict)
