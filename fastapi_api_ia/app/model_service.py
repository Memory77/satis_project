from pathlib import Path
from typing import Dict, Optional
import json

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# =========================
# Configuration MLflow Model Registry
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


model_service = ModelService()