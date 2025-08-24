import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Avoid importing heavy transformer and torch libraries during tests
    import sys
    import types
    from pathlib import Path

    dummy_transformers = types.SimpleNamespace(
        pipeline=lambda *args, **kwargs: (lambda text: [])
    )
    monkeypatch.setitem(sys.modules, "transformers", dummy_transformers)

    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))

    from main import app

    monkeypatch.setenv("RSS_FEED_URL", "http://example.com/rss")
    monkeypatch.chdir(tmp_path)

    return TestClient(app)


def test_start_endpoint(client, monkeypatch):
    def mock_fetch(url):
        return [{"title": "Test article", "content": "Company AAPL is great"}]

    def mock_analyze(text):
        return {"scores": {"POSITIVE": 0.9, "NEGATIVE": 0.1}}

    def mock_extract(text):
        return "AAPL"

    def mock_get_indicators(ticker):
        return {"direction": "up", "prob_up": 0.8}

    def mock_recommend(sentiment, indicators):
        return {
            "action": "BUY",
            "confidence": 0.95,
            "reason": "Good outlook",
        }

    monkeypatch.setattr("main.fetch_articles", mock_fetch)
    monkeypatch.setattr("main.analyze_news_article", mock_analyze)
    monkeypatch.setattr("main.extract_ticker", mock_extract)
    monkeypatch.setattr("main.get_price_indicators", mock_get_indicators)
    monkeypatch.setattr("main.make_recommendation", mock_recommend)

    response = client.get("/start")
    expected = [
        {
            "title": "Test article",
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 0.95,
            "reason": "Good outlook",
        }
    ]

    assert response.status_code == 200
    assert response.json() == expected

    with open("predictions_log.json") as f:
        data = json.load(f)
    assert data == expected
