from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# chemin local vers le modèle
BASE_DIR = Path(__file__).resolve().parent
CAMEMBERT_MODEL_PATH = (
    BASE_DIR
    / "fastapi_api_ia"
    / "models"
    / "modele_sentiment_camembert"
)

# identifiant du dépôt sur Hugging Face Hub
REPO_ID = "solen16/modele_sentiment_camembert"

print(f"Chargement du modèle local depuis {CAMEMBERT_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(str(CAMEMBERT_MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(
    str(CAMEMBERT_MODEL_PATH)
)

print(f"Envoi du modèle vers Hugging Face Hub : {REPO_ID}...")

# pousser le modèle et le tokenizer sur Hugging Face
# private=False crée un dépôt public sur Hugging Face (pour y accéder sans token)
tokenizer.push_to_hub(REPO_ID, private=False)
model.push_to_hub(REPO_ID, private=False)

print(" Succès ! Le modèle est désormais en ligne sur le hub Hugging Face.")