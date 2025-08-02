# Stock News GPT Notifier

Fetches the latest financial news and analyzes each article with Google's Gemini model.

## Setup

1. Install the dependencies:

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

2. Configure a `.env` file with your `GOOGLE_API_KEY`.

3. Start the FastAPI server with Uvicorn:

   ```bash
   uvicorn main:app --reload
   ```

   The background scheduler fetches articles every hour and logs Gemini's
   predictions to `predictions_log.json`.

4. Open [http://localhost:8000/](http://localhost:8000/) to see the stored
   predictions in a simple HTML table.  Use the interactive API docs at
   [http://localhost:8000/docs](http://localhost:8000/docs) to manually submit
   articles to `/predict`.

