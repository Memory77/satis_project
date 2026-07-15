from supabase import create_client
import pandas as pd

SUPABASE_URL = ""
SUPABASE_KEY = "sb_publishable_TA_CLE"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

response = (
    supabase
    .table("trustpilot_scraping_v1")
    .select("*")
    .execute()
)

df = pd.DataFrame(response.data)

print(df.head())
print(df.shape)