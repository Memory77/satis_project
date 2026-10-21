from script import get_supabase_client
from select import select
from supabase import Client, create_client
import pandas as pd
from fastapi import FastAPI, Request
import requests, os, sys

api_url = os.environ.get("SUPABASE_URL")
secret_key = os.environ.get("SUPABASE_KEY")

supabase_client = get_supabase_client()

try:
    
    #nombre d'avis déjà traités
    response = (
        supabase_client.from_("sentiment_predict")
        .select("*", count="exact", head=True)
        .execute()
    )

    avant = response.count

    #récupérer les données de supabase (id + commentaires)
    response = (
        supabase_client.from_("reviews_a_predire_sentiment")
        .select("id, content, company, title", count="exact")
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
    for y in unique_resp[:500]:
        content = (y.get("content") or "").strip()
        title = (y.get("title") or "").strip()
        if not content and title:
            y["content"] = title

        r = requests.post(
            url='http://127.0.0.1:8000/predict_sentiment',
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
    supabase_client = get_supabase_client()
    
    if preds:

        #correspondance données/colonnes
        clean_preds = []
        for pred in preds:
              pred.pop("concurrent")
              clean_preds.append(pred)

        clean_preds_uniques = list({item["review"]: item for item in clean_preds}.values())
        
        response = (
            supabase_client.from_("multilabel_predict")
            .upsert(clean_preds_uniques, on_conflict="review, sentiment, probabilities, thresholds")
            .execute()
        )

        #nombre d'avis traités
        response = (
            supabase_client.from_("sentiment_predict")
            .select("*", count="exact", head=True)
            .execute()
        )

        après = response.count

        print(f"Succès ! {après - avant} sur {nb} avis ont été traités par le modèle sentiment")

except Exception as e:
    print(f"❌ Erreur Supabase : {e}")
    sys.exit(1)