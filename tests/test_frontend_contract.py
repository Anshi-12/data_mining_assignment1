from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vite_proxy_points_to_fastapi() -> None:
    config = (ROOT / "client" / "vite.config.js").read_text(encoding="utf-8")
    assert "port: 5179" in config
    assert "'/api'" in config
    assert "http://127.0.0.1:8006" in config


def test_browser_health_client_uses_relative_api_path() -> None:
    api = (ROOT / "client" / "src" / "services" / "api.js").read_text(encoding="utf-8")
    assert "getHealth = () => request('/api/health')" in api


def test_phase5_dashboard_uses_all_real_api_endpoints() -> None:
    api = (ROOT / "client" / "src" / "services" / "api.js").read_text(encoding="utf-8")
    for path in (
        "/api/metadata", "/api/anomaly/score", "/api/manifold", "/api/benchmarks",
        "/api/autoresearch/history", "/api/anomalies/top", "/api/retrain/options", "/api/retrain",
    ):
        assert path in api


def test_phase5_dashboard_separates_model_score_from_iqr_explanation() -> None:
    app = (ROOT / "client" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "Why it looks unusual · separate explanation" in app
    assert "do not produce or override the model verdict" in app
    assert "Real measured search runs" in app
    assert "locally synthesized rows" in app
    assert "separable synthetic anomaly archetypes" in app


def test_scorer_controls_are_driven_by_feature_metadata() -> None:
    app = (ROOT / "client" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "metadata?.feature_metadata" in app
    assert "min={f.min}" in app
    assert "max={f.max}" in app
    assert "step={f.step}" in app
