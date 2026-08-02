from pathlib import Path
from typing import Dict, List, Optional

import joblib


# Chemin vers ton modèle
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modele_multilabel_avis_local.joblib"


class ModelService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

        bundle = joblib.load(MODEL_PATH)

        # Ton modèle local est normalement sauvegardé sous forme :
        # {"model": model, "label_cols": label_cols}
        self.model = bundle["model"]
        self.label_cols = bundle["label_cols"]

        # Seuil de décision.
        # Si proba >= 0.65, on active le label.
        self.default_threshold = 0.65

    def predict_one(self, commentaire: str, concurrent: Optional[str] = None) -> Dict:
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


model_service = ModelService()