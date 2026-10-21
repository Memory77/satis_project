from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Définir le chemin local vers votre modèle CamemBERT
BASE_DIR = Path(__file__).resolve().parent
CAMEMBERT_MODEL_PATH = (
    BASE_DIR
    / "fastapi_api_ia"
    / "models"
    / "modele_sentiment_camembert"
)

# 2. Remplacer "solen16" par votre vrai pseudo exact sur Hugging Face + le slash '/'
REPO_ID = "solen16/modele_sentiment_camembert"

print(f"Chargement du modèle local depuis {CAMEMBERT_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(str(CAMEMBERT_MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(
    str(CAMEMBERT_MODEL_PATH)
)

print(f"Envoi du modèle vers Hugging Face Hub : {REPO_ID}...")

# 3. Pousser le modèle et le tokenizer sur Hugging Face
# private=False crée un dépôt public sur Hugging Face (plus simple pour que l'équipe y accède sans token)
tokenizer.push_to_hub(REPO_ID, private=False)
model.push_to_hub(REPO_ID, private=False)

print(" Succès ! Votre modèle est désormais en ligne sur Hugging Face.")