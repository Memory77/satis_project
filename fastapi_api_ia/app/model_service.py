from pathlib import Path
from typing import Dict, Optional
import json

import torch
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =========================
# Configuration MLflow Model Registry (Multi-label)
# =========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "avis_multilabel_model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


# =========================
# Fallback local joblib
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modele_multilabel_avis_local.joblib"

# =========================
# CamemBERT
# =========================
CAMEMBERT_PATH = BASE_DIR / "models" / "modele_sentiment_camembert"
SENTIMENT_NAME = "avis_sentiment_model"
SENTIMENT_ALIAS = "champion1"

class ModelService:
    def __init__(self):
        self.client = MlflowClient()

        # -------------------------
        # 1. Récupérer la version exacte du modèle champion
        # -------------------------

        self.model_version = self.client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS
        )

        # -------------------------
        # 2. Charger le modèle depuis MLflow Registry
        # -------------------------

        self.model = mlflow.sklearn.load_model(MODEL_URI)

        # -------------------------
        # 3. Infos du modèle pour traçabilité API + database
        # -------------------------

        self.model_info = {
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "model_version": self.model_version.version,
            "mlflow_run_id": self.model_version.run_id,
            "model_uri": MODEL_URI,
            "source": self.model_version.source,
        }

        # -------------------------
        # 4. Charger les labels
        # -------------------------
        # Méthode propre : depuis MLflow si label_cols.json existe
        # Fallback : depuis le .joblib local si pas encore présent dans MLflow

        self.label_cols = self._load_label_cols()

        # -------------------------
        # 5. Seuil de décision
        # -------------------------
        # 0.50 = cohérent avec l'évaluation actuelle faite avec model.predict()
        # Plus tard, tu pourras passer à des seuils optimisés par label.

        self.default_threshold = 0.50

    def _load_label_cols(self):
        """
        Charge les labels associés au modèle.

        Priorité :
        1. label_cols.json depuis les artefacts MLflow du run
        2. fallback depuis le fichier joblib local
        """

        try:
            label_cols_path = self.client.download_artifacts(
                run_id=self.model_version.run_id,
                path="label_cols.json"
            )

            with open(label_cols_path, "r", encoding="utf-8") as f:
                label_cols = json.load(f)

            return label_cols

        except Exception:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "Impossible de charger label_cols.json depuis MLflow "
                    f"et modèle local introuvable : {MODEL_PATH}"
                )

            bundle = joblib.load(MODEL_PATH)

            if "label_cols" not in bundle:
                raise KeyError(
                    "Le fichier joblib local ne contient pas la clé 'label_cols'."
                )

            return bundle["label_cols"]

    def predict_one(self, commentaire: str, concurrent: Optional[str] = None) -> Dict:
        # Le modèle renvoie une probabilité par label
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
            "model_info": self.model_info,
        }

class SentimentModelService:
    """Service dédié au modèle d'analyse de sentiment (CamemBERT)"""
    def __init__(self):
        self.client = MlflowClient()

        if not CAMEMBERT_PATH.exists():
            raise FileNotFoundError(f"Modèle CamemBERT introuvable : {CAMEMBERT_PATH}")

        print(f"Chargement du modèle CamemBERT depuis {CAMEMBERT_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(CAMEMBERT_PATH))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(CAMEMBERT_PATH))
        self.model.eval()
        self.id2label = {0: "négatif", 1: "mitigé", 2: "positif"}
        self.default_threshold = 0.65

        try:
            # Tentative de chargement via MLflow avec l'alias champion1
            sentiment_uri = f"models:/{SENTIMENT_NAME}@{SENTIMENT_ALIAS}"
            self.model_version = self.client.get_model_version_by_alias(
                name=SENTIMENT_NAME,
                alias=SENTIMENT_ALIAS
            )

            import mlflow.transformers
            self.pipeline = mlflow.transformers.load_model(sentiment_uri)

            # Ajout des infos pour le modèle local
            self.model_info = {
                "model_name": SENTIMENT_NAME,
                "model_alias": SENTIMENT_ALIAS,
                "model_version": self.model_version.version,
                "mlflow_run_id": run_id_ou_fallback,
                "source": "mlflow_registry"
            }
            print(">>> Modèle CamemBERT chargé avec succès depuis MLflow Registry.")

        except Exception as e:
            # Fallback sur le dossier local si MLflow échoue ou si l'alias n'existe pas encore
            print(f">>> Avertissement MLflow Sentiment ({e}). Bascule sur le dossier local : {CAMEMBERT_PATH}")
            if not CAMEMBERT_PATH.exists():
                raise FileNotFoundError(f"Modèle CamemBERT introuvable localement : {CAMEMBERT_PATH}")

            self.tokenizer = AutoTokenizer.from_pretrained(str(CAMEMBERT_PATH))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(CAMEMBERT_PATH))
            self.model.eval()

            self.model_info = {
                "model_name": SENTIMENT_NAME,
                "source": str(CAMEMBERT_PATH),
                "mode": "local_fallback"
            }
    def predict_sentiment(self, text: str) -> dict:
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
            },
            "thresholds": {
                "négatif": self.default_threshold,
                "mitigé": self.default_threshold,
                "positif": self.default_threshold
            },
            "model_info": self.model_info
        }
    
model_service = ModelService()
sentiment_model_service = SentimentModelService()