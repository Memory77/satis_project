import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, hamming_loss, accuracy_score


# =========================
# 0. Configuration MLflow
# =========================

mlflow.set_experiment("classification_multilabel_avis_clients")


# =========================
# 1. Charger le CSV
# =========================

df = pd.read_csv("all_reviews_multilabel.csv", encoding="utf-8-sig")

label_cols = [col for col in df.columns if col.startswith("pb_")]

print("Labels détectés :")
print(label_cols)

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# On entraîne sur les avis négatifs uniquement
df_model = df[df["rating"] <= 2].copy()

X = df_model["content"].fillna("")
y = df_model[label_cols].astype(int)

print(f"\nNombre d'avis utilisés pour le modèle : {len(df_model)}")


# =========================
# 2. Paramètres du modèle
# =========================

test_size = 0.2
random_state = 42

tfidf_params = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 30000,
}

logreg_params = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "liblinear",
}


# =========================
# 3. Séparer train / test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


# =========================
# 4. Créer le modèle
# =========================

model = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_params)),
    ("clf", OneVsRestClassifier(
        LogisticRegression(**logreg_params)
    ))
])


# =========================
# 5. Entraîner + logger dans MLflow
# =========================

with mlflow.start_run(run_name="tfidf_ovr_logistic_regression"):

    # -------------------------
    # Logger les paramètres
    # -------------------------

    mlflow.log_param("dataset", "all_reviews_multilabel.csv")
    mlflow.log_param("nb_lignes_total", len(df))
    mlflow.log_param("nb_lignes_modele_rating_inf_ou_egal_2", len(df_model))
    mlflow.log_param("nb_labels", len(label_cols))
    mlflow.log_param("labels", ", ".join(label_cols))

    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    for param, value in tfidf_params.items():
        mlflow.log_param(f"tfidf_{param}", value)

    for param, value in logreg_params.items():
        mlflow.log_param(f"logreg_{param}", value)

    mlflow.log_param("strategie_multilabel", "OneVsRestClassifier")
    mlflow.log_param("modele_base", "LogisticRegression")
    mlflow.log_param("vectorisation", "TF-IDF")

    # -------------------------
    # Entraîner
    # -------------------------

    model.fit(X_train, y_train)

    # -------------------------
    # Évaluer
    # -------------------------

    y_pred = model.predict(X_test)

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    hamming = hamming_loss(y_test, y_pred)
    subset_acc = accuracy_score(y_test, y_pred)

    print("\nRapport de classification :")
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_cols,
        zero_division=0
    )
    print(report_text)

    print("Micro F1 :", round(micro_f1, 3))
    print("Macro F1 :", round(macro_f1, 3))
    print("Weighted F1 :", round(weighted_f1, 3))
    print("Hamming loss :", round(hamming, 3))
    print("Subset accuracy :", round(subset_acc, 3))

    # -------------------------
    # Logger les métriques globales
    # -------------------------

    mlflow.log_metric("micro_f1", micro_f1)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("weighted_f1", weighted_f1)
    mlflow.log_metric("hamming_loss", hamming)
    mlflow.log_metric("subset_accuracy", subset_acc)

    # -------------------------
    # Logger les métriques par label
    # -------------------------

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_cols,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("classification_report_multilabel.csv", encoding="utf-8-sig")

    for label in label_cols:
        mlflow.log_metric(f"{label}_precision", report_dict[label]["precision"])
        mlflow.log_metric(f"{label}_recall", report_dict[label]["recall"])
        mlflow.log_metric(f"{label}_f1", report_dict[label]["f1-score"])
        mlflow.log_metric(f"{label}_support", report_dict[label]["support"])

    # -------------------------
    # Sauvegarder en joblib pour ton API FastAPI
    # -------------------------

    bundle = {
        "model": model,
        "label_cols": label_cols
    }

    joblib.dump(bundle, "modele_multilabel_avis_local.joblib")

    print("\nModèle sauvegardé : modele_multilabel_avis_local.joblib")

    # -------------------------
    # Logger les artefacts dans MLflow
    # -------------------------

    mlflow.log_artifact("modele_multilabel_avis_local.joblib")
    mlflow.log_artifact("classification_report_multilabel.csv")

    # Optionnel : logger le modèle sklearn au format MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="modele_tfidf_ovr_logreg"
    )

    print("\nRun MLflow enregistré.")