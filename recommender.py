import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


df_clean = pd.read_csv("Cleaned_Travel_Data.csv")

user_item_matrix = df_clean.pivot_table(
    index="UserID",
    columns="DestinationID",
    values="ExperienceRatingNorm",
    fill_value=0
)

svd = TruncatedSVD(n_components=20, random_state=42)
item_embeddings = svd.fit_transform(user_item_matrix.T)
destination_similarity = cosine_similarity(item_embeddings)

# Índices de destinos
destination_indices = {
    dest: idx for idx, dest in enumerate(user_item_matrix.columns)
}

# Base para contenido
destination_content = df_clean[[
    "DestinationID",
    "Name_y",
    "PopularityNorm",
    "PreferencesList",
    "Type",
    "BestTimeToVisit"
]].drop_duplicates(subset=["DestinationID"])

# MultiLabelBinarizer para PreferencesList
mlb = MultiLabelBinarizer()
prefs_encoded = pd.DataFrame(
    mlb.fit_transform(destination_content["PreferencesList"]),
    columns=[f"Pref_{c}" for c in mlb.classes_]
)
prefs_encoded.index = destination_content.index

# One-hot encode Type y BestTimeToVisit
dummies = pd.get_dummies(destination_content[["Type", "BestTimeToVisit"]])

# Concatenar
destination_content_encoded = pd.concat([
    destination_content[["DestinationID", "Name_y", "PopularityNorm"]],
    dummies,
    prefs_encoded
], axis=1)

# Asegurarse que todo sea float
for col in destination_content_encoded.columns:
    if destination_content_encoded[col].dtype in [bool, int]:
        destination_content_encoded[col] = destination_content_encoded[col].astype(float)

content_similarity = cosine_similarity(
    destination_content_encoded.drop(columns=["DestinationID", "Name_y"])
)


destination_names_df = df_clean[["DestinationID", "Name_y"]].drop_duplicates()

def recommend_destinations(user_id, top_n=5, alpha=0.5):
    if user_id not in user_item_matrix.index:
        return None

    visited = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
    collab_scores = np.zeros(len(user_item_matrix.columns))
    content_scores = np.zeros(len(user_item_matrix.columns))

    for dest in visited:
        idx = destination_indices[dest]
        collab_scores += destination_similarity[idx]
        content_scores += content_similarity[idx]
    
    if len(visited) > 0:
        collab_scores /= len(visited)
        content_scores /= len(visited)
    
    hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores

    recommendations = pd.DataFrame({
        "DestinationID": user_item_matrix.columns,
        "HybridScore": hybrid_scores
    })

    recommendations = recommendations[~recommendations["DestinationID"].isin(visited)]
    recommendations = recommendations.merge(destination_names_df, on="DestinationID", how="left")
    recommendations = recommendations.groupby("Name_y").agg({"HybridScore": "mean"}).reset_index()
    recommendations = recommendations.sort_values("HybridScore", ascending=False).head(top_n)
    recommendations = recommendations.rename(columns={"Name_y": "DestinationName"})
    return recommendations


def recommend_by_profile(user_profile, top_n=5):
    # One-hot de Type (puedes cambiar la lógica si quieres)
    type_cols = [c for c in destination_content_encoded.columns if c.startswith("Type_")]
    type_vec = pd.Series(0.0, index=type_cols)

    # One-hot de BestTimeToVisit (puedes ajustar la lógica)
    time_cols = [c for c in destination_content_encoded.columns if c.startswith("BestTimeToVisit_")]
    time_vec = pd.Series(0.0, index=time_cols)

    # MultiLabel de Preferences
    pref_cols = [c for c in destination_content_encoded.columns if c.startswith("Pref_")]
    pref_vec = pd.Series(0.0, index=pref_cols)
    for pref in user_profile["Preferences"].split(", "):
        col = f"Pref_{pref.strip()}"
        if col in pref_vec.index:
            pref_vec[col] = 1.0

    # Popularidad media
    popularity_mean = destination_content_encoded["PopularityNorm"].mean()

    # Concatenar
    user_vector = pd.concat([
        pd.Series({"PopularityNorm": popularity_mean}),
        type_vec,
        time_vec,
        pref_vec
    ]).to_frame().T

    # Ordenar columnas igual que en destino
    user_vector = user_vector[destination_content_encoded.drop(columns=["DestinationID", "Name_y"]).columns]

    # Convertir a float
    user_vector = user_vector.astype(float)

    # Similitud
    sim = cosine_similarity(
        destination_content_encoded.drop(columns=["DestinationID", "Name_y"]),
        user_vector
    ).flatten()

    recommendations = destination_content_encoded[["DestinationID", "Name_y"]].copy()
    recommendations["Similarity"] = sim
    recommendations = recommendations.sort_values("Similarity", ascending=False).head(top_n)
    recommendations = recommendations.rename(columns={"Name_y": "DestinationName"})
    return recommendations
