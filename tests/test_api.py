from fastapi.testclient import TestClient

from api.main import app


def test_health_reports_status():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_predict_rejects_when_no_model_loaded():
    with TestClient(app) as client:
        resp = client.post("/predict", files={"file": ("x.jpg", b"abc", "image/jpeg")})
        assert resp.status_code == 503
