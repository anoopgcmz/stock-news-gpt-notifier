
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from api.routes import router as prediction_router

# Scheduler imports
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import json
import os

from context.news_scraper import fetch_articles
from model.hf_predict import analyze_news_article
from model.stock_predict import (
    extract_ticker,
    get_price_indicators,
    make_recommendation,
)


app = FastAPI()
app.include_router(prediction_router, prefix="/predict")


@app.get("/", response_class=HTMLResponse)
def read_predictions():
    """Display stored analysis predictions as an HTML table."""
    log_file = "predictions_log.json"
    if not os.path.exists(log_file):
        return "<h1>No predictions available</h1>"

    with open(log_file, "r") as f:
        predictions = json.load(f)

    rows = "".join(
        f"<tr><td>{p['title']}</td>"
        f"<td>{p.get('ticker','')}</td>"
        f"<td>{p.get('action','')}</td>"
        f"<td>{p.get('confidence','')}</td>"
        f"<td>{p.get('reason','')}</td></tr>"
        for p in predictions
    )

    html = f"""
    <html>
        <head><title>Model Predictions</title></head>
        <body>
            <h1>Model Predictions</h1>
            <table border='1'>
                <tr><th>Article</th><th>Ticker</th><th>Action</th><th>Confidence</th><th>Reason</th></tr>
                {rows}
            </table>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/start")
def start_process():
    """Run article processing and return structured predictions."""

    rss_url = os.getenv("RSS_FEED_URL")
    articles = fetch_articles(rss_url)

    predictions = []
    for article in articles:
        sentiment = analyze_news_article(article["content"])
        ticker = extract_ticker(article["title"] + " " + article["content"])
        indicators = get_price_indicators(ticker) if ticker else {}
        recommendation = make_recommendation(sentiment, indicators)
        prediction = {
            "title": article["title"],
            "ticker": ticker,
            "action": recommendation["action"],
            "confidence": recommendation["confidence"],
            "reason": recommendation["reason"],
        }
        predictions.append(prediction)

    log_file = "predictions_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(predictions)
    with open(log_file, "w") as f:
        json.dump(existing, f, indent=2)

    return {"predictions": predictions}


def process_articles():
    """Fetch recent articles and log model predictions."""
    rss_url = os.getenv("RSS_FEED_URL")
    articles = fetch_articles(rss_url)
    predictions = []
    for article in articles:
        sentiment = analyze_news_article(article["content"])
        ticker = extract_ticker(article["title"] + " " + article["content"])
        indicators = get_price_indicators(ticker) if ticker else {}
        recommendation = make_recommendation(sentiment, indicators)
        predictions.append(
            {
                "title": article["title"],
                "ticker": ticker,
                "action": recommendation["action"],
                "confidence": recommendation["confidence"],
                "reason": recommendation["reason"],
            }
        )

    if not predictions:
        return []

    log_file = "predictions_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(predictions)
    with open(log_file, "w") as f:
        json.dump(existing, f, indent=2)

    return predictions


# Run the job every hour in the background (disabled by default).
# To enable, set the environment variable ENABLE_SCHEDULER to a truthy value.
if os.getenv("ENABLE_SCHEDULER"):
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_articles, "interval", hours=1)
    scheduler.start()

    # Shutdown scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
