import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("title_features.csv")

print("Dataset Loaded:", df.shape)

# ==========================
# Load Sentence Transformer
# ==========================

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully")

# ==========================
# Generate Embeddings
# ==========================

embeddings = model.encode(
    df["title"].fillna("").tolist(),
    show_progress_bar=True
)

print("Embeddings Shape:", embeddings.shape)

# ==========================
# Convert Embeddings to DataFrame
# ==========================

embedding_df = pd.DataFrame(
    embeddings,
    columns=[f"embedding_{i}" for i in range(embeddings.shape[1])]
)

# ==========================
# Merge with Original Features
# ==========================

final_df = pd.concat(
    [df, embedding_df],
    axis=1
)

print(final_df.shape)

# ==========================
# Save File
# ==========================

final_df.to_csv(
    "title_features_with_embeddings.csv",
    index=False
)

print("Saved Successfully!")