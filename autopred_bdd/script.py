#import requests_oauthlib
from select import select
from supabase import Client, create_client
import pandas as pd
from fastapi import FastAPI, Request
import requests, os, sys

api_url = os.environ.get("SUPABASE_URL")
secret_key = os.environ.get("SUPABASE_KEY")

#récupérer les données de supabase (id + commentaires)
supabase_client: Client = create_client(api_url, secret_key)

response = (
    supabase_client.from_("trustpilot_scraping")
    .select("id, content, company")
    .lt("rating", 3)
    .limit(5)
    .execute())

resp = response.data

#passer les données à l'API du modèle
preds = []
for y in resp:
    r = requests.post(
        url='http://127.0.0.1:8000/predict',
        json={
    "commentaire": y['content'],
    "concurrent": y['company']
        },
        headers={"Content-Type": "application/json"}
        )
    dict = r.json()
    dict['review'] = y['id']
    preds.append(dict)

#response_dict = r.json()
#response_header = r.headers
#status_code = r.status_code

#print(preds)

#enregistrer les résultats dans supabase
try:
    print(f"📤 Connexion Supabase... ({len(preds)} prédictions)")
    supabase_client: Client = create_client(api_url, secret_key)
    
    if preds:

        #correspondance données/colonnes
        clean_preds = []
        for pred in preds:
              pred.pop("commentaire")
              pred.pop("concurrent")
              clean_preds.append(pred)

        response = (
            supabase_client.from_("multilabel_predict")
            .upsert(clean_preds, on_conflict="problemes_predits, predictions, probabilities, thresholds")
            .execute()
        )
        print("Succès !")
except Exception as e:
    print(f"❌ Erreur Supabase : {e}")
    sys.exit(1)