import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import dagshub, os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

#Dagshub
REPO_OWNER = "maudb18"
REPO_NAME = "satis_project"
DAGSHUB_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(DAGSHUB_URI)

# =========================
# 0. Configuration MLflow
# =========================
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("analyse_sentiment_camembert")



# Identifiant sur le Hugging Face Hub (plus de chemin local)
MODEL_ID = "solen16/modele_sentiment_camembert"


def save_and_register_sentiment_model():
  print(f"Chargement de CamemBERT depuis le Hub Hugging Face ({MODEL_ID})...")

  # Téléchargement direct depuis Hugging Face Hub (géré en cache)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

  # =========================
  # Démarrage du run MLflow
  # =========================
  with mlflow.start_run(run_name="camembert_sentiment_registration"):

    # 1. Logger les paramètres de base
    mlflow.log_param("model_type", "CamemBERT")
    mlflow.log_param("task", "sequence-classification")
    mlflow.log_param("num_labels", 3)
    mlflow.log_param("labels", "négatif, mitigé, positif")
    mlflow.log_param("model_source", MODEL_ID)

    print(
        "Enregistrement du modèle dans le registre MLflow"
        " (avis_sentiment_model)..."
    )

    # 2. Logger et enregistrer le modèle via mlflow.transformers
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="sentiment_camembert_artifact",
        task="text-classification",
        registered_model_name="avis_sentiment_model",
        save_pretrained=False,  # <-- Empêche la création/copie de model.safetensors dans MLflow
    )

    # 3. Associer l'alias "champion1" à la version nouvellement créée
    client = MlflowClient(tracking_uri=DAGSHUB_URI)
    run_id = mlflow.active_run().info.run_id

    model_versions = client.search_model_versions(
        f"name='avis_sentiment_model' and run_id='{run_id}'"
    )

    if model_versions:
      latest_version = max(model_versions, key=lambda mv: int(mv.version))

      client.set_registered_model_alias(
          name="avis_sentiment_model",
          alias="champion1",
          version=latest_version.version,
      )

      mlflow.log_param("registered_model_name", "avis_sentiment_model")
      mlflow.log_param("registered_model_version", latest_version.version)
      mlflow.log_param("model_alias", "champion1")

      print(
          "\nSuccès ! Modèle CamemBERT enregistré sous 'avis_sentiment_model'"
          f" (Version {latest_version.version}) avec l'alias '@champion1'."
      )
    else:
      print(
          "\nAucune version retrouvée dans le Model Registry pour CamemBERT."
      )


if __name__ == "__main__":
  save_and_register_sentiment_model()