import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# Charge le .env situé dans le même dossier que le script
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL absente du fichier .env")

if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY absente du fichier .env")

# Nettoyage des espaces et guillemets accidentels
SUPABASE_URL = (
    SUPABASE_URL
    .strip()
    .strip('"')
    .strip("'")
    .rstrip("/")
)

SUPABASE_KEY = (
    SUPABASE_KEY
    .strip()
    .strip('"')
    .strip("'")
)

print("URL chargée :", repr(SUPABASE_URL))
print(
    "URL REST qui sera utilisée :",
    f"{SUPABASE_URL}/rest/v1/trustpilot_scraping"
)

# Vérifie qu'aucun chemin n'a été ajouté à la Project URL
if "/rest/v1" in SUPABASE_URL:
    raise ValueError(
        "SUPABASE_URL ne doit pas contenir /rest/v1. "
        "Garde seulement https://xxxxx.supabase.co"
    )

if not SUPABASE_URL.startswith("https://"):
    raise ValueError(
        "SUPABASE_URL doit commencer par https:// "
        "et ne doit pas être une URI postgresql://"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

response = (
    supabase
    .table("trustpilot_scraping_v1")
    .select("*")
    .limit(10)
    .execute()
)

df = pd.DataFrame(response.data)

print(df.head())
print("Dimensions :", df.shape)