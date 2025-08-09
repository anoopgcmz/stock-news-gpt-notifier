from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator

from model.hf_predict import analyze_news_article
from model.stock_predict import extract_ticker, get_price_indicators, make_recommendation


class NewsArticle(BaseModel):
    """Schema for incoming news articles."""

    title: str
    content: str | None = None

    @validator("content", pre=True, always=True)
    def validate_content(cls, value: str | None) -> str:  # pragma: no cover - simple validation
        """Ensure that article content is present and non-empty."""
        if value is None:
            raise ValueError("Article content is required.")
        if not str(value).strip():
            raise ValueError("Article content cannot be empty.")
        return value


router = APIRouter()


@router.post("/")
async def predict_from_news(article: NewsArticle):
    try:
        sentiment = analyze_news_article(article.content)
        if "error" in sentiment:
            raise HTTPException(status_code=400, detail=sentiment["error"])
        text = f"{article.title} {article.content}"
        ticker = extract_ticker(text)
        indicators = get_price_indicators(ticker) if ticker else {}
        recommendation = make_recommendation(sentiment, indicators)
        return {
            "ticker": ticker,
            "action": recommendation["action"],
            "confidence": recommendation["confidence"],
            "reason": recommendation["reason"],
        }
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive programming
        raise HTTPException(status_code=500, detail=str(e))
