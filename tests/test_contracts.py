from pathlib import Path

from server.artifact_schema import ArtifactLayout
from server.config import settings


def test_fixed_ports_and_seed() -> None:
    assert settings.port == 8006
    assert settings.frontend_origin.endswith(":5179")
    assert settings.random_seed == 42


def test_future_artifact_layout_reserves_real_model_components() -> None:
    layout = ArtifactLayout(Path("server/artifacts"))
    assert layout.scaler.name == "robust_scaler.joblib"
    assert layout.detector.name == "isolation_forest.joblib"
    assert layout.calibration.name == "score_calibration.json"
    assert layout.search_history.name == "search_history.json"
