import os
import time
import json
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub

load_dotenv()

# --- Configuration ---
LOG_FILE = "request_log.json"
MAX_REQUESTS_PER_MINUTE = 5
MAX_REQUESTS_PER_DAY = 100
ALLOW_FALLBACK = True

# --- Setup Hugging Face ---
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PRO_MODEL_NAME = os.getenv("HF_PRO_MODEL", "google/flan-t5-large")
FALLBACK_MODEL_NAME = os.getenv("HF_FALLBACK_MODEL", "google/flan-t5-base")

"""Prediction helper using a Hugging Face hosted model via LangChain."""

# --- Load or Init Request Log ---
def load_request_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_request_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f)

# --- Cleanup + Count ---
def update_and_check_limits():
    log = load_request_log()
    now = time.time()
    one_minute_ago = now - 60
    one_day_ago = now - 86400

    # Remove old entries
    log = [ts for ts in log if ts > one_day_ago]

    requests_last_minute = len([ts for ts in log if ts > one_minute_ago])
    requests_today = len(log)

    return log, requests_last_minute, requests_today

# --- Rate-limit aware request function ---
def analyze_news_article(text: str) -> str:
    log, rpm, rpd = update_and_check_limits()

    # Handle over-limit
    if rpd >= MAX_REQUESTS_PER_DAY:
        return "❌ Daily limit reached: Please try again tomorrow."

    if rpm >= MAX_REQUESTS_PER_MINUTE:
        return "⏳ Too many requests this minute. Please wait a moment."

    # Choose model name
    if rpd > MAX_REQUESTS_PER_DAY - 10 or rpm >= MAX_REQUESTS_PER_MINUTE - 1:
        if ALLOW_FALLBACK:
            model_name = FALLBACK_MODEL_NAME
        else:
            return "⚠️ Request rate near limit. Try again later."
    else:
        model_name = PRO_MODEL_NAME

    # Compose prompt
    prompt = f"""
    Given the following financial news article: "{text}"

    1. Identify which public stock/company it is about (include name + ticker if possible).
    2. Predict whether the stock will go UP, DOWN, or stay NEUTRAL tomorrow.
    3. Give a brief reason.

    Format:
    Ticker: XYZ
    Prediction: UP/DOWN/NEUTRAL
    Reason: ...
    """

    try:
        llm = HuggingFaceHub(
            repo_id=model_name,
            huggingfacehub_api_token=HF_API_TOKEN,
        )
        result = llm.invoke(prompt).strip()
    except Exception as e:
        return f"❌ Hugging Face API error: {str(e)}"

    # Log success
    log.append(time.time())
    save_request_log(log)

    return result
