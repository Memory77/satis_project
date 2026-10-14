from pathlib import Path
import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 0. Configuration MLflow
# =========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("analyse_sentiment_camembert")

# Chemins locaux
BASE_DIR = Path(__file__).resolve().parent.parent
CAMEMBERT_MODEL_PATH = BASE_DIR / "fastapi_api_ia" / "models" / "modele_sentiment_camembert"

def save_and_register_sentiment_model():
    if not CAMEMBERT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle CamemBERT introuvable localement : {CAMEMBERT_MODEL_PATH}")

    print(f"Chargement de CamemBERT depuis {CAMEMBERT_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(CAMEMBERT_MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(CAMEMBERT_MODEL_PATH))

    # =========================
    # Démarrage du run MLflow
    # =========================
    with mlflow.start_run(run_name="camembert_sentiment_registration"):
        
        # 1. Logger les paramètres de base
        mlflow.log_param("model_type", "CamemBERT")
        mlflow.log_param("task", "sequence-classification")
        mlflow.log_param("num_labels", 3)
        mlflow.log_param("labels", "négatif, mitigé, positif")

        print("Enregistrement du modèle dans le registre MLflow (avis_sentiment_model)...")

        # 2. Logger et enregistrer le modèle via mlflow.transformers
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="sentiment_camembert_artifact",
            task="text-classification",
            registered_model_name="avis_sentiment_model"
        )

        # 3. Associer l'alias "champion" à la version nouvellement créée
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id

        model_versions = client.search_model_versions(
            f"name='avis_sentiment_model' and run_id='{run_id}'"
        )

        if model_versions:
            latest_version = max(model_versions, key=lambda mv: int(mv.version))

            client.set_registered_model_alias(
                name="avis_sentiment_model",
                alias="champion1",
                version=latest_version.version
            )

            mlflow.log_param("registered_model_name", "avis_sentiment_model")
            mlflow.log_param("registered_model_version", latest_version.version)
            mlflow.log_param("model_alias", "champion")
            
            print(f"\nSuccès ! Modèle CamemBERT enregistré sous 'avis_sentiment_model' (Version {latest_version.version}) avec l'alias '@champion'.")
        else:
            print("\nAucune version retrouvée dans le Model Registry pour CamemBERT.")

if __name__ == "__main__":
    save_and_register_sentiment_model()