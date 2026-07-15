import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, hamming_loss, accuracy_score


# =========================
# 1. Charger le CSV
# =========================

df = pd.read_csv("all_reviews_multilabel.csv", encoding="utf-8-sig")

label_cols = [col for col in df.columns if col.startswith("pb_")]

print("Labels détectés :")
print(label_cols)

# On entraîne sur les avis négatifs uniquement
df_model = df[df["rating"] <= 2].copy()

X = df_model["content"].fillna("")
y = df_model[label_cols].astype(int)

print(f"\nNombre d'avis utilisés pour le modèle : {len(df_model)}")


# =========================
# 2. Séparer train / test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 3. Créer le modèle
# =========================

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000
    )),
    ("clf", OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        )
    ))
])


# =========================
# 4. Entraîner
# =========================

model.fit(X_train, y_train)


# =========================
# 5. Évaluer
# =========================

y_pred = model.predict(X_test)

print("\nRapport de classification :")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_cols,
    zero_division=0
))

print("Micro F1 :", round(f1_score(y_test, y_pred, average="micro", zero_division=0), 3))
print("Macro F1 :", round(f1_score(y_test, y_pred, average="macro", zero_division=0), 3))
print("Hamming loss :", round(hamming_loss(y_test, y_pred), 3))
print("Subset accuracy :", round(accuracy_score(y_test, y_pred), 3))


# =========================
# 6. Sauvegarder le modèle
# =========================

bundle = {
    "model": model,
    "label_cols": label_cols
}

joblib.dump(bundle, "modele_multilabel_avis_local.joblib")

print("\nModèle sauvegardé : modele_multilabel_avis_local.joblib")