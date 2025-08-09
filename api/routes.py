from fastapi import APIRouter, HTTPException
from model.hf_predict import analyze_news_article
from model.stock_predict import extract_ticker, get_price_indicators, make_recommendation

router = APIRouter()

@router.post("/")
async def predict_from_news(article: dict):
    try:
        sentiment = analyze_news_article(article["content"])
        if "error" in sentiment:
            raise HTTPException(status_code=400, detail=sentiment["error"])
        text = article.get("title", "") + " " + article["content"]
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
