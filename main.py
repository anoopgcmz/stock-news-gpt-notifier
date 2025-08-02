
from fastapi import FastAPI
from api.routes import router as prediction_router

# Scheduler imports
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import json
import os

from context.news_scraper import fetch_articles
from model.gpt_predict import analyze_news_article


app = FastAPI()
app.include_router(prediction_router, prefix="/predict")


def process_articles():
    """Fetch recent articles and log model predictions."""
    articles = fetch_articles()
    predictions = []
    for article in articles:
        result = analyze_news_article(article["content"])
        predictions.append({"title": article["title"], "prediction": result})

    if not predictions:
        return

    log_file = "predictions_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(predictions)
    with open(log_file, "w") as f:
        json.dump(existing, f, indent=2)


# Run the job every hour in the background
scheduler = BackgroundScheduler()
scheduler.add_job(process_articles, "interval", hours=1)
scheduler.start()

# Shutdown scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())
