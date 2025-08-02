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

3. Run `main.py` to start the notifier.

