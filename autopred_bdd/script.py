from supabase import Client, create_client
import os

def get_supabase_client():
    api_url = os.environ.get("SUPABASE_URL")
    secret_key = os.environ.get("SUPABASE_KEY")
    if not api_url or not secret_key:
        raise ValueError("Variables SUPABASE_URL ou SUPABASE_KEY manquantes.")
    return create_client(api_url, secret_key)