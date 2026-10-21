import streamlit as st
import pandas as pd

st.write("Bienvenue sur Streamlit!")

# Configuration de la page
st.set_page_config(
    page_title="Dashboard - Satisfaction Client",
    page_icon="📊",
    layout="wide"
)

st.title("Analyse de la Satisfaction Client")
st.markdown("""Ce dashboard présente l'analyse croisée de **35 867 avis clients** collectés sur le web
et traités via un modèle de Deep Learning (**Fine-tuning CamemBERT**) et de Machine Learning (**Multi-label**).""")

# Exemple de métriques globales (KPIs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avis collectés", "35 867", "+?% ce mois-ci")
col2.metric("Sources principales", "Poulpeo (69.8%)", "Trustpilot (30.2%)")
col3.metric("Enseignes analysées", "9 enseignes", "Darty, Fnac, Boulanger...")
col4.metric("Modèle NLP", "CamemBERT", "Multilabel & Sentiment")

st.info("👈 Utilisez le menu de navigation à gauche pour basculer entre la Vue Globale et la Vue par Enseigne.")