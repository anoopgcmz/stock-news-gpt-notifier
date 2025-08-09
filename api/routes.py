from fastapi import APIRouter, HTTPException
from model.hf_predict import analyze_news_article

router = APIRouter()

@router.post("/")
async def predict_from_news(article: dict):
    try:
        result = analyze_news_article(article["content"])
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
