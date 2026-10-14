from fastapi import FastAPI

from app.schemas import PredictRequest, PredictResponse, SentimentResponse
# from app.model_service import model_service
from app.model_service import multilabel_service, sentiment_service


app = FastAPI(
    title="API IA - Classification multi-label et analyse de sentiment des avis clients",
    version="1.0.1"
)


@app.get("/")
def root():
    return {
        "message": "API IA multi-label et analyse de sentiment opérationnelle"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "labels": multilabel_service.label_cols
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = multilabel_service.predict_one(
        commentaire=request.commentaire,
        concurrent=request.concurrent
    )
    return result

@app.post("/predict_sentiment", response_model=SentimentResponse) # ajout de la route pour le modèle CamemBERT
def predict_sentiment(request: PredictRequest):
    result = sentiment_service.predict_sentiment(request.commentaire)
    return result
