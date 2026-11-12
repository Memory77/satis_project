from supabase import Client, create_client

def get_supabase_client(url, key):
    if not url or not key:
        raise ValueError("Variables SUPABASE_URL ou SUPABASE_KEY manquantes.")
    return create_client(url, key)