import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Avoid importing heavy transformer and torch libraries during tests
    import sys
    import types
    from pathlib import Path

    dummy_transformers = types.SimpleNamespace(
        pipeline=lambda *args, **kwargs: (lambda text: [])
    )
    monkeypatch.setitem(sys.modules, "transformers", dummy_transformers)

    # Ensure project root on path for "import main"
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))

    from main import app

    return TestClient(app)


def test_predict_success(client, monkeypatch):
    def mock_analyze(text):
        return {"scores": {"POSITIVE": 0.9, "NEGATIVE": 0.1}}

    def mock_extract(text):
        return "AAPL"

    def mock_get_price_indicators(ticker):
        return {"direction": "up", "prob_up": 0.8}

    def mock_make_recommendation(sentiment, indicators):
        return {"action": "BUY", "confidence": 0.95, "reason": "Test"}

    monkeypatch.setattr("api.routes.analyze_news_article", mock_analyze)
    monkeypatch.setattr("api.routes.extract_ticker", mock_extract)
    monkeypatch.setattr("api.routes.get_price_indicators", mock_get_price_indicators)
    monkeypatch.setattr("api.routes.make_recommendation", mock_make_recommendation)

    response = client.post(
        "/predict/",
        json={"title": "Some title", "content": "Some informative content"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ticker": "AAPL",
        "action": "BUY",
        "confidence": 0.95,
        "reason": "Test",
    }


def test_predict_missing_content(client):
    response = client.post("/predict/", json={"title": "Missing content"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"].endswith(
        "Article content is required."
    )


def test_predict_empty_content(client):
    response = client.post(
        "/predict/", json={"title": "Empty content", "content": ""}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"].endswith(
        "Article content cannot be empty."
    )
