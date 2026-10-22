import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
import mlflow.transformers
import torch
from mlflow.tracking import MlflowClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import dagshub, os

# =========================
# Configuration MLflow Registry
# =========================
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

#Dagshub
REPO_OWNER = "maudb18"
REPO_NAME = "satis_project"
DAGSHUB_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

token = os.environ.get("DAGSHUB_USER_TOKEN")
if token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

mlflow.set_tracking_uri(DAGSHUB_URI)

# Modèle Multi-label
MODEL_NAME = "avis_multilabel_model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# Modèle Sentiment (CamemBERT)
SENTIMENT_NAME = "avis_sentiment_model"
SENTIMENT_ALIAS = "champion1"
SENTIMENT_URI = f"models:/{SENTIMENT_NAME}@{SENTIMENT_ALIAS}"

# Identifiant Hugging Face Hub (remplace le dossier en local)
CAMEMBERT_HF_ID = "solen16/modele_sentiment_camembert"

# Fallback local pour le multilabel
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modele_multilabel_avis_local.joblib"


class ModelService:

  def __init__(self):
    self.client = MlflowClient()

  def __init__(self):
    self.client = MlflowClient(tracking_uri=DAGSHUB_URI)

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
  """Classe dédiée au modèle d'analyse de sentiment (CamemBERT)"""

  def __init__(self):
    self.client = MlflowClient(tracking_uri=DAGSHUB_URI)
    self.id2label = {0: "négatif", 1: "mitigé", 2: "positif"}
    self.default_threshold = 0.65

    try:
      # Tentative de récupération des infos depuis le registre MLflow pour la traçabilité
      print(
          "Récupération des métadonnées CamemBERT depuis MLflow Registry"
          f" ({SENTIMENT_URI})..."
      )
      self.model_version = self.client.get_model_version_by_alias(
          name=SENTIMENT_NAME, alias=SENTIMENT_ALIAS
      )

      self.model_info = {
          "model_name": SENTIMENT_NAME,
          "model_alias": SENTIMENT_ALIAS,
          "model_version": self.model_version.version,
          "mlflow_run_id": self.model_version.run_id,
          "source": "mlflow_registry",
      }
    except Exception as e:
      print(
          f">>> Avertissement métadonnées MLflow Sentiment ({e}). Utilisation"
          " des infos par défaut."
      )
      self.model_info = {
          "model_name": SENTIMENT_NAME,
          "source": "fallback",
      }

    # Chargement direct et robuste du modèle PyTorch et du tokenizer
    print(
        "Chargement du modèle et du tokenizer CamemBERT depuis"
        f" {CAMEMBERT_HF_ID}..."
    )
    self.tokenizer = AutoTokenizer.from_pretrained(CAMEMBERT_HF_ID)
    self.model = AutoModelForSequenceClassification.from_pretrained(
        CAMEMBERT_HF_ID
    )
    self.model.eval()
    print(">>> Modèle CamemBERT chargé avec succès.")

  def predict_sentiment(self, text: str) -> dict:
    # Tokenisation du texte
    inputs = self.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=False
    )


    # Cas où le modèle est un modèle PyTorch brut
    if hasattr(self.model, "eval"):
      self.model.eval()
      with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
    
    else:

        clean_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        res = self.model.predict([clean_text])
        probs = torch.tensor([res[0].get("négatif", 0), res[0].get("mitigé", 0), res[0].get("positif", 0)])


    # Extraction des scores pour chaque classe
    p_negative = float(probs[0])
    p_mixed = float(probs[1])
    p_positive = float(probs[2])

    pred_id = torch.argmax(probs).item()

    return {
        "commentaire": text,
        "sentiment": self.id2label[pred_id],
        "probabilities": {
            "négatif": round(p_negative, 4),
            "mitigé": round(p_mixed, 4),
            "positif": round(p_positive, 4),
        },
        "thresholds": {
            "négatif": self.default_threshold,
            "mitigé": self.default_threshold,
            "positif": self.default_threshold,
        },
        "model_info": self.model_info,
    }


model_service = ModelService()
sentiment_model_service = SentimentModelService()