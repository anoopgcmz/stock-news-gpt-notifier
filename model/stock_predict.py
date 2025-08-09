import re
from typing import Optional

try:  # pragma: no cover - optional dependencies may be missing
    import pandas as pd
    import yfinance as yf
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    pd = None
    yf = None
    Pipeline = StandardScaler = LogisticRegression = None


def extract_ticker(text: str) -> Optional[str]:
    """Very naive ticker extractor.

    Looks for the first occurrence of a capitalised word of 1-5 letters. This
    is simplistic but sufficient for demo purposes.
    """
    if not text:
        return None
    matches = re.findall(r"\b[A-Z]{1,5}\b", text)
    return matches[0] if matches else None


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_price_indicators(ticker: str) -> dict:
    """Fetch recent price data and compute indicators and trend direction."""
    if not ticker:
        return {}
    if yf is None or pd is None or Pipeline is None:
        return {}
    try:
        df = yf.download(
            ticker, period="3mo", interval="1d", auto_adjust=True, progress=False
        )
    except Exception:
        return {}
    if df.empty:
        return {}

    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["RSI"] = _compute_rsi(df["Close"])

    feature_cols = ["MA5", "MA20", "RSI"]
    dataset = df.dropna(subset=feature_cols)
    if len(dataset) < 25:
        return {}

    dataset["Target"] = (dataset["Close"].shift(-1) > dataset["Close"]).astype(int)
    dataset = dataset.dropna(subset=["Target"])

    X = dataset[feature_cols]
    y = dataset["Target"]

    if y.nunique() < 2:
        return {}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression()),
    ])
    try:
        model.fit(X.iloc[:-1], y.iloc[:-1])
        prob_up = model.predict_proba(X.iloc[[-1]])[0][1]
    except Exception:
        return {}

    indicators = {
        "ma5": round(dataset.iloc[-1]["MA5"], 2),
        "ma20": round(dataset.iloc[-1]["MA20"], 2),
        "rsi": round(dataset.iloc[-1]["RSI"], 2),
        "direction": "up" if prob_up >= 0.5 else "down",
        "prob_up": round(float(prob_up), 2),
    }
    return indicators


def make_recommendation(sentiment: dict, indicators: dict) -> dict:
    """Combine sentiment probabilities and price trend into a trading signal."""

    scores = {k.lower(): v for k, v in sentiment.get("scores", {}).items()}
    direction = indicators.get("direction")
    prob_up = indicators.get("prob_up")
    if not scores or direction is None or prob_up is None:
        return {"action": "HOLD", "confidence": 0.0, "reason": "Insufficient data"}

    sentiment_label = max(scores, key=scores.get)
    if sentiment_label == "positive" and direction == "up":
        action = "BUY"
        confidence = (scores["positive"] + prob_up) / 2
        reason = "Positive sentiment and upward trend"
    elif sentiment_label == "negative" and direction == "down":
        action = "SELL"
        confidence = (scores["negative"] + (1 - prob_up)) / 2
        reason = "Negative sentiment and downward trend"
    else:
        action = "HOLD"
        confidence = max(scores.get("neutral", 0), 1 - abs(0.5 - prob_up) * 2)
        reason = "Mixed signals"

    return {"action": action, "confidence": round(float(confidence), 2), "reason": reason}
