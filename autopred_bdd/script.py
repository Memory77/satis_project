#import requests_oauthlib
from select import select
from supabase import Client, create_client
import pandas as pd
from fastapi import FastAPI, Request
import requests, os, sys

api_url = os.environ.get("SUPABASE_URL")
secret_key = os.environ.get("SUPABASE_KEY")


supabase_client: Client = create_client(api_url, secret_key)

try:
    
    #nombre d'avis déjà traités
    response = (
        supabase_client.from_("multilabel_predict")
        .select("*", count="exact", head=True)
        .execute()
    )

    avant = response.count

    #récupérer les données de supabase (id + commentaires)
    response = (
        supabase_client.from_("reviews_a_predire")
        .select("id, content, company", count="exact")
        .execute()
    )

    resp = response.data
    nb = response.count

    if nb == 0:
        print("✅ Aucun nouvel avis à prédire. Fin normale de l'exécution.")
        sys.exit(0)
    else:
        print(f"{nb} avis à traiter")
    

except Exception as e:
    print(f"❌ Erreur Supabase : {e}")
    print()
    sys.exit(1)

try:
    #passer les données à l'API du modèle
    preds = []
    unique_resp = list({item["id"]: item for item in resp}.values())
    for y in unique_resp[:1000]:
        if y['content'] is None:
            continue

        r = requests.post(
            url='http://127.0.0.1:8000/predict',
            json={
        "commentaire": y['content'],
        "concurrent": y['company']
            },
            headers={"Content-Type": "application/json"}
            )

        r.raise_for_status()
        
        dict = r.json()
        dict['review'] = y['id']
        preds.append(dict)

except Exception as e:
    print(f"💥 {e}")
    sys.exit(1)

try:
    #enregistrer les résultats dans supabase
    print(f"📤 Connexion Supabase... ({len(preds)} prédictions)")
    supabase_client: Client = create_client(api_url, secret_key)
    
    if preds:

        #correspondance données/colonnes
        clean_preds = []
        for pred in preds:
              pred.pop("commentaire")
              pred.pop("concurrent")
              clean_preds.append(pred)

        clean_preds_uniques = list({item["review"]: item for item in clean_preds}.values())
        response = (
            supabase_client.from_("multilabel_predict")
            .upsert(clean_preds_uniques, on_conflict="problemes_predits, predictions, probabilities, thresholds")
            .execute()
        )

        #nombre d'avis déjà traités
        response = (
            supabase_client.from_("multilabel_predict")
            .select("*", count="exact", head=True)
            .execute()
        )

        après = response.count

        print(f"Succès ! {après - avant} sur {nb} avis ont été traités")

except Exception as e:
    print(f"❌ Erreur Supabase : {e}")
    sys.exit(1)