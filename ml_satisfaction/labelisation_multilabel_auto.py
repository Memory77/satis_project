import pandas as pd
import unicodedata


# ============================================================
# 1. Charger le dataset de base
# ============================================================

INPUT_FILE = "all_reviews.csv"
OUTPUT_FILE = "all_reviews_multilabel.csv"

df = pd.read_csv(INPUT_FILE, encoding="utf-8")

print("Dataset chargé :", df.shape)
print("Colonnes disponibles :", df.columns.tolist())


# ============================================================
# 2. Extraire le concurrent depuis page_url
# ============================================================

# Exemple :
# https://www.poulpeo.com/avis/cdiscount.htm
# devient :
# cdiscount

df["concurrent"] = df["page_url"].str.extract(r"/avis/([^/.]+)")[0]

print("\nNombre d'avis par concurrent :")
print(df["concurrent"].value_counts())


# ============================================================
# 3. Nettoyer le texte des commentaires
# ============================================================

def normalize_text(text):
    """
    Nettoyage du texte :
    - passage en minuscules
    - suppression des accents
    - harmonisation des apostrophes
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Suppression des accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Apostrophes typographiques vers apostrophe simple
    text = text.replace("’", "'")

    return text


df["content_norm"] = df["content"].apply(normalize_text)


# ============================================================
# 4. Définir les labels et leurs mots-clés
# ============================================================

labels_patterns = {
    "pb_remboursement_avoir": (
        r"\b(rembours\w*|avoir|bon d[' ]?achat|bon achat)\b"
    ),

    "pb_livraison_colis_retard": (
        r"\b(livraison|livre\w*|livreur|colis|transporteur|"
        r"retard\w*|delai\w*|expedi\w*|expedition|envoi|"
        r"relais|point relais)\b"
    ),

    "pb_service_client_sav": (
        r"\b(service client|sav|support|conseiller\w*|hotline|"
        r"appel\w*|telephone|mail\w*|email\w*|reclamation\w*|"
        r"reponse\w*|relance\w*|contact\w*|chat)\b"
    ),

    "pb_produit_defectueux_non_conforme": (
        r"\b(defectu\w*|panne\w*|casse\w*|endommage\w*|"
        r"abime\w*|non conforme|qualite|mauvais produit|"
        r"erreur de produit|article\w*|produit\w*)\b"
    ),

    "pb_paiement_prelevement_facturation": (
        r"\b(prelev\w*|debit\w*|carte bancaire|cb|banque|floa|"
        r"fraud\w*|autorisation|factur\w*|paiement\w*|payer|"
        r"encaisse\w*)\b"
    ),

    "pb_commande_annulation_stock": (
        r"\b(commande\w*|annul\w*|stock\w*|disponible\w*|"
        r"indisponible\w*|rupture|compte suspendu|suspendu\w*)\b"
    ),

    "pb_prix_promo_conditions": (
        r"\b(prix|promotion\w*|promo\w*|vente flash|cher\w*|"
        r"conditions|frais|reduction\w*|remise\w*|cashback|"
        r"poulpeo|bon plan)\b"
    ),

    "pb_site_appli_compte": (
        r"\b(site|application|appli|interface|compte|connexion|"
        r"connecter|bug\w*|navigation|plateforme|mot de passe|"
        r"login)\b"
    ),
}


# ============================================================
# 5. Appliquer la labellisation automatique
# ============================================================

# On labellise uniquement les avis négatifs.
# Ici, un avis négatif = note <= 2.
# Cela évite de classer "livraison rapide" comme problème de livraison.

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

negative_mask = df["rating"] <= 2

for label, pattern in labels_patterns.items():
    df[label] = (
        negative_mask
        & df["content_norm"].str.contains(pattern, regex=True, na=False)
    ).astype(int)


# ============================================================
# 6. Calculer le nombre de problèmes détectés
# ============================================================

label_cols = list(labels_patterns.keys())

df["nb_problemes_detectes"] = df[label_cols].sum(axis=1)


# ============================================================
# 7. Créer une colonne lisible avec la liste des problèmes
# ============================================================

df["problemes_detectes"] = df[label_cols].apply(
    lambda row: "; ".join(
        [label for label, value in row.items() if value == 1]
    ),
    axis=1
)

df.loc[
    df["problemes_detectes"] == "",
    "problemes_detectes"
] = "aucun_label_detecte"


# ============================================================
# 8. Exporter le dataset final labellisé
# ============================================================

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nFichier créé :", OUTPUT_FILE)
print("Dimensions du dataset final :", df.shape)


# ============================================================
# 9. Exporter un fichier de vérification visuelle
# ============================================================

cols_verif = [
    "concurrent",
    "rating",
]

if "title" in df.columns:
    cols_verif.append("title")

cols_verif += [
    "content",
    "problemes_detectes",
    "nb_problemes_detectes",
] + label_cols

df_verif = df[df["rating"] <= 2][cols_verif].copy()

df_verif.to_csv(
    "avis_negatifs_a_verifier.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Fichier créé : avis_negatifs_a_verifier.csv")
print("Nombre d'avis négatifs à vérifier :", len(df_verif))


# ============================================================
# 10. Créer un résumé par concurrent
# ============================================================

df_neg = df[df["rating"] <= 2].copy()

resume_pourcentage = (
    df_neg
    .groupby("concurrent")[label_cols]
    .mean()
    .mul(100)
    .round(1)
)

resume_nombre = (
    df_neg
    .groupby("concurrent")[label_cols]
    .sum()
    .astype(int)
)

nb_avis_negatifs = (
    df_neg
    .groupby("concurrent")
    .size()
    .rename("nb_avis_negatifs")
)

resume_final = resume_nombre.copy()

for col in label_cols:
    resume_final[col + "_pct"] = resume_pourcentage[col]

resume_final = resume_final.join(nb_avis_negatifs)

resume_final.to_csv(
    "resume_multilabel_concurrents.csv",
    encoding="utf-8-sig"
)

print("Fichier créé : resume_multilabel_concurrents.csv")


# ============================================================
# 11. Afficher un aperçu
# ============================================================

print("\nAperçu des avis négatifs labellisés :")
print(
    df_verif[
        [
            "concurrent",
            "rating",
            "content",
            "problemes_detectes",
            "nb_problemes_detectes",
        ]
    ].head(10)
)

print("\nRésumé en pourcentage par concurrent :")
print(resume_pourcentage)