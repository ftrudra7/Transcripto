import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_summarize_empty_text():
    response = client.post("/api/v1/summary/", json={"text": ""})
    # Pydantic validation should fail on min_length=1
    assert response.status_code == 422

def test_summarize_heuristic_fallback():
    # If API keys are not set, it falls back to heuristic which is 2 sentences max.
    text = "This is a test. It should just return the first two sentences. And ignore this one."
    response = client.post("/api/v1/summary/", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data

def test_predict_sign_empty():
    response = client.post("/api/v1/sign/", json={"landmarks": []})
    assert response.status_code == 422

def test_predict_sign_valid():
    landmarks = [{"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(21)]
    response = client.post("/api/v1/sign/", json={"landmarks": landmarks})
    assert response.status_code == 200
    assert "text" in response.json()
    assert "score" in response.json()
