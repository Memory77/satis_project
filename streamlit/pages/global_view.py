import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client

st.set_page_config(
    page_title="Dashboard - Vue Globale",
    page_icon="📊",
    layout="wide"
)

# initialisation du client Supabase via les secrets (.streamlit/secrets.toml)
url: str = st.secrets["SUPABASE_URL"]
key: str = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

# fonction mise en cache pour 1 heure (refresh automatique toutes les heures)
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

# calcul des pourcentages des sources pour l'affichage
source_dict = dict(zip(df_sources['source'], df_sources['count'])) if not df_sources.empty else {}
total_sources_sum = sum(source_dict.values()) if source_dict else 1
poulpeo_pct = (source_dict.get('Poulpeo', 0) / total_sources_sum) * 100
trustpilot_pct = (source_dict.get('Trustpilot', 0) / total_sources_sum) * 100


# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avis collectés", f"{total_avis:,}", "Mis à jour en temps réel")
col2.metric("Sources principales", f"Poulpeo ({poulpeo_pct:.1f}%)", f"Trustpilot ({trustpilot_pct:.1f}%)")
col3.metric("Enseignes analysées", f"{nb_enseignes} enseignes", "Données Supabase")
col4.metric("Modèle NLP", "CamemBERT", "Multilabel & Sentiment")

st.info("Utilisez le menu de navigation à gauche pour basculer entre la vue globale et la vue par enseigne.")
st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Classement des enseignes (Note moyenne sur 5)")
    fig_bar = px.bar(
        df_classement, 
        x="Note_Moyenne", 
        y="Enseigne", 
        orientation="h",
        title="Note moyenne par enseigne / 5",
        color="Note_Moyenne",
        color_continuous_scale="Plotly3"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("Volume d'avis par enseigne et par note")
    st.info("Ce graphique empilé montre la ventilation des avis positifs contre celle des avis négatifs par enseigne.")
    
    fig_stack = px.bar(
        df_stack, 
        x="Enseigne", 
        y=["Avis_Positifs", "Avis_Negatifs"], 
        title="Répartition simplifiée Positif vs Négatif", 
        color_discrete_map={"Avis_Positifs": "green", "Avis_Negatifs": "red"}
    )
    st.plotly_chart(fig_stack, use_container_width=True)

st.subheader("Évolution temporelle du volume d'avis")
st.markdown("""
*Note :* Visualisation de l'évolution du volume d'avis stockés en base de données.
""")

fig_line = px.line(df_temps, x="Date", y="Volume", markers=True, title="Tendance globale du volume d'avis")
st.plotly_chart(fig_line, use_container_width=True)