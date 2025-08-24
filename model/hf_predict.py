import os
import time
import threading
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

# --- Configuration ---
MAX_REQUESTS_PER_MINUTE = 5
MAX_REQUESTS_PER_DAY = 100

# --- Model Initialization ---
MODEL_DIR = os.getenv("FINBERT_MODEL_DIR", "models/finbert-tone")

try:
    # ``return_all_scores`` will let us expose a probability for each label
    _classifier = pipeline("text-classification", model=MODEL_DIR, return_all_scores=True)
    _classifier_error = None
except Exception as e:  # pragma: no cover - model path may be missing in tests
    _classifier = None
    _classifier_error = str(e)

"""Prediction helper using a locally cached Hugging Face model."""

# --- In-memory Request Log ---
_REQUEST_LOG = []
_LOG_LOCK = threading.Lock()


def load_request_log():
    """Return a copy of the current request log."""
    with _LOG_LOCK:
        return list(_REQUEST_LOG)


def save_request_log(timestamp: float) -> None:
    """Append a timestamp to the request log in a thread-safe manner."""
    with _LOG_LOCK:
        _REQUEST_LOG.append(timestamp)


def update_and_check_limits():
    """Prune old requests and return current usage counts atomically."""
    with _LOG_LOCK:
        now = time.time()
        one_minute_ago = now - 60
        one_day_ago = now - 86400

        # Remove requests older than one day
        _REQUEST_LOG[:] = [ts for ts in _REQUEST_LOG if ts > one_day_ago]

        requests_last_minute = len([ts for ts in _REQUEST_LOG if ts > one_minute_ago])
        requests_today = len(_REQUEST_LOG)

    return requests_last_minute, requests_today

# --- Rate-limit aware request function ---
def analyze_news_article(text: str) -> dict:
    """Classify the sentiment of a financial news article.

    Returns a dictionary with the most likely ``label`` as well as
    probabilities for all labels under ``scores``. If the model is unavailable
    or a rate limit is exceeded, an ``error`` key is returned instead of
    raising an exception.
    """

    if _classifier is None:
        return {"error": f"Model not loaded: {_classifier_error}"}

    rpm, rpd = update_and_check_limits()

    # Handle over-limit
    if rpd >= MAX_REQUESTS_PER_DAY:
        return {"error": "Daily limit reached: Please try again tomorrow."}

    if rpm >= MAX_REQUESTS_PER_MINUTE:
        return {"error": "Too many requests this minute. Please wait a moment."}

    try:
        output = _classifier(text)[0]
        scores = {item["label"].upper(): round(item["score"], 2) for item in output}
        label = max(scores, key=scores.get)
        prediction = {"label": label, "scores": scores}
    except Exception as e:
        return {"error": f"Hugging Face model error: {str(e)}"}

    # Log success
    save_request_log(time.time())

    return prediction
