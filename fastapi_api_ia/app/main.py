from fastapi import FastAPI

from app.schemas import PredictRequest, PredictResponse
from app.model_service import model_service


app = FastAPI(
    title="API IA - Classification multi-label des avis clients",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "API IA multi-label opérationnelle"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "labels": model_service.label_cols
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = model_service.predict_one(
        commentaire=request.commentaire,
        concurrent=request.concurrent
    )
    return result