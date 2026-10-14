from pathlib import Path
from typing import Dict, Optional
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent.parent

class ModelService:
    def __init__(self, model_filename: str, is_camembert: bool = False):
        self.is_camembert = is_camembert
        self.model_path = BASE_DIR / "models" / model_filename
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        if self.is_camembert:
            print(f"Chargement du modèle CamemBERT depuis {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.eval()
            self.id2label = {0: "négatif", 1: "mitigé", 2: "positif"}
        else:
            print(f"Chargement du modèle multi-label depuis {self.model_path}...")
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self.model = bundle["model"]
                self.label_cols = bundle.get("label_cols", [])
            else:
                self.model = bundle
                self.label_cols = []
            self.default_threshold = 0.65

    def predict_sentiment(self, text: str) -> dict:
        """Méthode dédiée au modèle CamemBERT (Sentiment)"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()

        return {
            "commentaire": text,
            "sentiment": self.id2label[pred_id],
            "probabilities": {
                "négatif": round(float(probs[0]), 4),
                "mitigé": round(float(probs[1]), 4),
                "positif": round(float(probs[2]), 4)
            }
        }

    def predict_one(self, commentaire: str, concurrent: Optional[str] = None) -> Dict:
        """Méthode dédiée au modèle multi-label (Scikit-Learn)"""
        probas = self.model.predict_proba([commentaire])[0]
        predictions = {}
        probabilities = {}
        thresholds = {}
        problemes_predits = []

        for label, proba in zip(self.label_cols, probas):
            proba = float(proba)
            threshold = self.default_threshold
            pred = int(proba >= threshold)

            predictions[label] = pred
            probabilities[label] = round(proba, 4)
            thresholds[label] = threshold

            if pred == 1:
                problemes_predits.append(label)

        return {
            "commentaire": commentaire,
            "concurrent": concurrent,
            "problemes_predits": problemes_predits,
            "predictions": predictions,
            "probabilities": probabilities,
            "thresholds": thresholds,
        }

# Instanciation des deux services distincts
multilabel_service = ModelService("modele_multilabel_avis_local.joblib", is_camembert=False)
sentiment_service = ModelService("modele_sentiment_camembert", is_camembert=True)