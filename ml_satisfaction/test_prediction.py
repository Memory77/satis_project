import joblib
import pandas as pd

bundle = joblib.load("modele_multilabel_avis_local.joblib")

model = bundle["model"]
label_cols = bundle["label_cols"]

commentaires = [
    "Commande annulée, service client impossible à joindre et toujours pas remboursé.",
    "Colis reçu en retard avec un produit cassé.",
    "Le site bug, impossible de me connecter à mon compte."
]

predictions = model.predict(commentaires)

resultats = pd.DataFrame(predictions, columns=label_cols)
resultats.insert(0, "commentaire", commentaires)

resultats["problemes_predits"] = resultats[label_cols].apply(
    lambda row: "; ".join([col for col, value in row.items() if value == 1]),
    axis=1
)

resultats.loc[resultats["problemes_predits"] == "", "problemes_predits"] = "aucun_label_predit"

resultats.to_csv("resultats_predictions.csv", index=False, encoding="utf-8-sig")