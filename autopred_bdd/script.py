#import requests_oauthlib
from select import select
from supabase import Client, create_client
import pandas as pd
from fastapi import FastAPI, Request
import requests

api_url = os.environ.get("SUPABASE_URL")
secret_key = os.environ.get("SUPABASE_KEY")

#récupérer les données de supabase (id + commentaires)
supabase_client: Client = create_client(api_url, secret_key)

response = (
    supabase_client.from_("trustpilot_scraping")
    .select("id, content, company")
    .limit(1)
    .execute())

resp = response.data

#passer les données à l'API du modèle
for y in resp:
    r = requests.post(
        url='http://127.0.0.1:8000/predict',
        json={
    "commentaire": y['content'],
    "concurrent": y['company']
        },
        headers={"Content-Type": "application/json"}
        )

response_dict = r.json()
response_header = r.headers
status_code = r.status_code

print(response_dict)

#enregistrer les résultats dans supabase
