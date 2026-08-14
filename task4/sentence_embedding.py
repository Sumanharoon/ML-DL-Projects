import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================
# Load Processed Dataset
# ==========================

df = pd.read_csv("processed_articles.csv")

# ==========================
# Load MiniLM Model
# ==========================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# Handle Missing Values
# ==========================

df["clean_content"] = df["clean_content"].fillna("")

# ==========================
# Generate Sentence Embeddings
# ==========================

embeddings = model.encode(
    df["clean_content"].tolist(),
    show_progress_bar=True
)

# ==========================
# Add Embeddings to DataFrame
# ==========================

embedding_df = pd.DataFrame(embeddings)

# Rename embedding columns
embedding_df.columns = [f"embedding_{i}" for i in range(embedding_df.shape[1])]

# Merge with original processed data
final_df = pd.concat([df, embedding_df], axis=1)

# ==========================
# Save New File
# ==========================

final_df.to_csv("processed_articles_with_embeddings.csv", index=False)

print("Sentence Embeddings Created Successfully!")
print("Output File: processed_articles_with_embeddings.csv")