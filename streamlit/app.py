import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client

st.set_page_config(
    page_title="Dashboard - Satisfaction Client",
    page_icon="📊",
    layout="wide"
)

url: str = st.secrets["SUPABASE_URL"]
key: str = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

@st.cache_data(ttl=3600)
def load_data():
    # total des avis
    res_total = supabase.table("trustpilot_scraping").select("id", count="exact").execute()
    total_avis = res_total.count if res_total.count is not None else 0

    # répartition par source d'avis
    res_sources = supabase.table("trustpilot_scraping").select("source").execute()
    df_sources = pd.DataFrame(res_sources.data)
    if not df_sources.empty:
        df_sources = df_sources.groupby("source").size().reset_index(name="count")
    else:
        df_sources = pd.DataFrame(columns=["source", "count"])

    # nombre d'enseignes 
    res_enseignes = supabase.table("trustpilot_scraping").select("company").execute()
    df_all_companies = pd.DataFrame(res_enseignes.data)
    nb_enseignes = df_all_companies["company"].nunique() if not df_all_companies.empty else 0

    # récupération globale pour les transformations pandas
    res_full = supabase.table("trustpilot_scraping").select("company, rating, date_pub").execute()
    df = pd.DataFrame(res_full.data)

    if not df.empty:
        # classement des enseignes (Note moyenne)
        df_classement = df.groupby("company")["rating"].mean().reset_index()
        df_classement.columns = ["Enseigne", "Note_Moyenne"]
        df_classement["Note_Moyenne"] = df_classement["Note_Moyenne"].round(2)
        df_classement = df_classement.sort_values(by="Note_Moyenne", ascending=False)

        # volume d'avis par enseigne (positifs vs négatifs basés sur la note >= 4)
        df["Avis_Positifs"] = df["rating"].apply(lambda x: 1 if x is not None and x >= 4 else 0)
        df["Avis_Negatifs"] = df["rating"].apply(lambda x: 1 if x is not None and x < 4 else 0)
        df_stack = df.groupby("company")[["Avis_Positifs", "Avis_Negatifs"]].sum().reset_index()
        df_stack.rename(columns={"company": "Enseigne"}, inplace=True)
        # tri par volume total décroissant
        df_stack["Total"] = df_stack["Avis_Positifs"] + df_stack["Avis_Negatifs"]
        df_stack = df_stack.sort_values(by="Total", ascending=False).drop(columns=["Total"])

        # evolution temporelle (par année)
        df["date_pub"] = pd.to_datetime(df["date_pub"], errors="coerce")
        df["Date"] = df["date_pub"].dt.strftime("%Y")
        df_temps = df.groupby("Date").size().reset_index(name="Volume")
        df_temps = df_temps.dropna().sort_values("Date")
    else:
        df_classement = pd.DataFrame(columns=["Enseigne", "Note_Moyenne"])
        df_stack = pd.DataFrame(columns=["Enseigne", "Avis_Positifs", "Avis_Negatifs"])
        df_temps = pd.DataFrame(columns=["Date", "Volume"])

    return total_avis, df_sources, nb_enseignes, df_classement, df_stack, df_temps

# chargement des données
total_avis, df_sources, nb_enseignes, df_classement, df_stack, df_temps = load_data()

st.write("Bienvenue sur notre dashboard d'analyse de la satisfaction client")
st.title("Analyse de la Satisfaction Client des enseignes d'électroménager et de high-tech")
st.markdown(f"""Ce dashboard présente l'analyse croisée de **{total_avis:,} avis clients** collectés sur le web
et traités via un modèle de Machine Learning (**Multi-label**) et de Deep Learning (**Fine-tuning CamemBERT**).""")

st.info("⬅️**Utilisez le menu de navigation dans la barre latérale à gauche** pour explorer les différentes vues :")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Vue Globale")
    st.write("Retrouvez les KPIs globaux, le classement général de toutes les enseignes et l'évolution temporelle globale.")

with col2:
    st.subheader("Vue par Enseigne")
    st.write("Filtrez les données enseigne par enseigne pour analyser en détail leurs notes et leurs derniers avis.")

st.markdown("---")
st.caption("Projet propulsé par Deborah, Maud et Solen")