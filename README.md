# Stock News GPT Notifier

Fetches the latest financial news and analyzes each article with Google's Gemini model.

## Setup

1. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:

   - Linux/macOS:

     ```bash
     source venv/bin/activate
     ```

   - Windows:

     ```bash
     venv\Scripts\activate
     ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This project depends on:

   - APScheduler
   - fastapi
   - feedparser
   - google-generativeai
   - lxml[html_clean]
   - newspaper3k
   - python-dotenv

4. Configure a `.env` file with your `GOOGLE_API_KEY`.

5. Start the FastAPI server with Uvicorn:

   ```bash
   uvicorn main:app --reload
   ```

   The background scheduler fetches articles every hour and logs Gemini's
   predictions to `predictions_log.json`.

6. Open [http://localhost:8000/](http://localhost:8000/) to see the stored
   predictions in a simple HTML table.  Use the interactive API docs at
   [http://localhost:8000/docs](http://localhost:8000/docs) to manually submit
   articles to `/predict`.

## Choosing the News Source

By default the app pulls articles from the Economic Times stocks RSS feed. To
use a different source, set the `RSS_FEED_URL` environment variable before
starting the server or pass a feed URL to `fetch_articles` in your own code.

```bash
export RSS_FEED_URL="https://in.finance.yahoo.com/rss/topstories"  # Yahoo Finance
uvicorn main:app --reload
```

Constants such as `ECONOMIC_TIMES_RSS` and `YAHOO_FINANCE_INDIA_RSS` are
available in `context/news_scraper.py` for convenience.


## `/start` Endpoint

Visiting `/start` triggers the article processing pipeline and displays three steps:

1. Fetch articles.
2. Run Gemini analysis on each article.
3. Save predictions to `predictions_log.json`.

The page returns an HTML report showing the results of each step.
