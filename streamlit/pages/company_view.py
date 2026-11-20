import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client

st.set_page_config(
    page_title="Dashboard - Vue par Enseigne",
    page_icon="🔍",
    layout="wide"
)

url: str = st.secrets["SUPABASE_URL"]
key: str = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

st.title("Analyse détaillée par Enseigne")
st.markdown("Sélectionnez une enseigne dans le menu latéral pour afficher ses statistiques approfondies.")

@st.cache_data(ttl=3600)
def load_companies():
    res = supabase.table("trustpilot_scraping").select("company").execute()
    df = pd.DataFrame(res.data)
    if not df.empty and "company" in df.columns:
        return sorted(df["company"].dropna().unique().tolist())
    return []

companies = load_companies()

if not companies:
    st.warning("Aucune enseigne trouvée dans la base de données Supabase.")
else:
    selected_company = st.sidebar.selectbox("Choisissez une enseigne :", companies)

    @st.cache_data(ttl=600)
    def load_company_data(company_name):
        res = supabase.table("trustpilot_scraping").select("*").eq("company", company_name).execute()
        return pd.DataFrame(res.data)

    df_company = load_company_data(selected_company)

    if df_company.empty:
        st.info(f"Aucune donnée disponible pour l'enseigne : {selected_company}")
    else:
        total_avis_enseigne = len(df_company)
        note_moyenne = df_company["rating"].mean() if "rating" in df_company.columns else 0
        sources_count = df_company["source"].nunique() if "source" in df_company.columns else 1

        col1, col2, col3 = st.columns(3)
        col1.metric("Avis pour cette enseigne", f"{total_avis_enseigne:,}")
        col2.metric("Note moyenne", f"{note_moyenne:.2f} / 5")
        col3.metric("Sources de collecte", f"{sources_count}")

        st.markdown("---")

        col_left, col_right = st.columns(2)
        rating_palette = {
            1: "#d62728",  # rouge
            2: "#ff7f0e",  # orange
            3: "#f1c40f",  # jaune
            4: "#2ecc71",  # vert clair
            5: "#1e8449",  # vert foncé
        }

        with col_left:
            st.subheader("Distribution des notes (1 à 5 étoiles)")
            if "rating" in df_company.columns:
                df_rating_counts = df_company["rating"].value_counts().reset_index()
                df_rating_counts.columns = ["Note", "Nombre"]
                df_rating_counts = df_rating_counts.sort_values("Note")
                
                fig_ratings = px.bar(
                    df_rating_counts, 
                    x="Note", 
                    y="Nombre", 
                    text="Nombre",
                    title=f"Répartition des notes - {selected_company}",
                    color=df_rating_counts["Note"].astype(str),
                    color_discrete_map={str(k): v for k, v in rating_palette.items()},
                )
                st.plotly_chart(fig_ratings, width='stretch')

        with col_right:
            st.subheader("Évolution temporelle des avis")
            if "date_pub" in df_company.columns:
                df_company["date_pub"] = pd.to_datetime(df_company["date_pub"], errors="coerce")
                df_temp_enseigne = df_company.dropna(subset=["date_pub"]).copy()
                df_temp_enseigne["Annee"] = df_temp_enseigne["date_pub"].dt.strftime("%Y")
                
                df_timeline = df_temp_enseigne.groupby("Annee").size().reset_index(name="Volume")
                
                if not df_timeline.empty:
                    fig_time = px.line(
                        df_timeline, 
                        x="Annee", 
                        y="Volume", 
                        markers=True, 
                        title=f"Volume annuel des avis - {selected_company}"
                    )
                    st.plotly_chart(fig_time, width='stretch')
                else:
                    st.info("Données de date insuffisantes pour afficher l'évolution.")


        text_cols = [c for c in ["comment", "content", "text", "review"] if c in df_company.columns]
        if text_cols:
            st.markdown("---")
            st.subheader(f"10 Derniers avis clients pour {selected_company}")
            col_text = text_cols[0]
            display_cols = [c for c in ["date_pub", "source", "rating", col_text] if c in df_company.columns]
            
            st.dataframe(
                df_company[display_cols].sort_values(by="date_pub", ascending=False).head(10),
                width='stretch'
            )