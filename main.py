
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
        f"<td>{p.get('sentiment',{}).get('label','')}</td>"
        f"<td>{p.get('indicators',{}).get('ma5','')}</td>"
        f"<td>{p.get('indicators',{}).get('ma20','')}</td>"
        f"<td>{p.get('indicators',{}).get('rsi','')}</td>"
        f"<td>{p.get('indicators',{}).get('direction','')}</td>"
        f"<td>{p.get('recommendation','')}</td></tr>"
        for p in predictions
    )

    html = f"""
    <html>
        <head><title>Hugging Face Analysis</title></head>
        <body>
            <h1>Hugging Face Analysis Predictions</h1>
            <table border='1'>
                <tr><th>Article</th><th>Ticker</th><th>Sentiment</th><th>MA5</th><th>MA20</th><th>RSI</th><th>Trend</th><th>Recommendation</th></tr>
                {rows}
            </table>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/start", response_class=HTMLResponse)
def start_process():
    """Run article processing and display step-by-step status as HTML."""

    # Step 1: Fetch articles
    rss_url = os.getenv("RSS_FEED_URL")
    articles = fetch_articles(rss_url)
    titles_html = "".join(f"<li>{a['title']}</li>" for a in articles) or "<li>No articles found</li>"

    # Step 2: Analyze articles with sentiment and price data
    predictions = []
    analysis_items = []
    for article in articles:
        sentiment = analyze_news_article(article["content"])
        ticker = extract_ticker(article["title"] + " " + article["content"])
        indicators = get_price_indicators(ticker) if ticker else {}
        recommendation = make_recommendation(
            sentiment.get("label"), indicators.get("direction")
        )
        prediction = {
            "title": article["title"],
            "ticker": ticker,
            "sentiment": sentiment,
            "indicators": indicators,
            "recommendation": recommendation,
        }
        predictions.append(prediction)
        analysis_items.append(
            f"<li><strong>{article['title']}</strong>: {recommendation}</li>"
        )
    analysis_html = "".join(analysis_items) or "<li>No analyses performed</li>"

    # Step 3: Append predictions to the log file
    log_file = "predictions_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(predictions)
    with open(log_file, "w") as f:
        json.dump(existing, f, indent=2)
    save_message = f"Saved {len(predictions)} prediction(s) to {log_file}."

    html = f"""
    <html>
        <head><title>Article Processing</title></head>
        <body>
            <h1>Article Processing Steps</h1>
            <h2>Step 1: Fetch Articles</h2>
            <ul>{titles_html}</ul>
            <h2>Step 2: Analyze Articles</h2>
            <ul>{analysis_html}</ul>
            <h2>Step 3: Save Predictions</h2>
            <p>{save_message}</p>
        </body>
    </html>
    """

    return HTMLResponse(content=html)


def process_articles():
    """Fetch recent articles and log model predictions."""
    rss_url = os.getenv("RSS_FEED_URL")
    articles = fetch_articles(rss_url)
    predictions = []
    for article in articles:
        sentiment = analyze_news_article(article["content"])
        ticker = extract_ticker(article["title"] + " " + article["content"])
        indicators = get_price_indicators(ticker) if ticker else {}
        recommendation = make_recommendation(
            sentiment.get("label"), indicators.get("direction")
        )
        predictions.append(
            {
                "title": article["title"],
                "ticker": ticker,
                "sentiment": sentiment,
                "indicators": indicators,
                "recommendation": recommendation,
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
