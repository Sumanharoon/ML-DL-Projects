import pandas as pd
from sklearn.decomposition import PCA

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("scaled_features.csv")

print("Original Shape:", df.shape)

# ==========================
# Non-Numeric Columns
# ==========================

non_numeric = [
    "title",
    "predicted_category",
    "readability_level",
    "title_length_category"
]

# ==========================
# Separate Target
# ==========================

y = df["clicks"]

# ==========================
# Select Features
# ==========================

X = df.drop(columns=non_numeric + ["clicks"])

print("Features Before PCA:", X.shape)

# ==========================
# Apply PCA
# ==========================

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X)

print("Features After PCA:", X_pca.shape)

# ==========================
# Convert to DataFrame
# ==========================

pca_columns = [
    f"PC{i+1}" for i in range(X_pca.shape[1])
]

X_pca = pd.DataFrame(
    X_pca,
    columns=pca_columns
)

# ==========================
# Combine Data
# ==========================

final_df = pd.concat(
    [
        df[non_numeric].reset_index(drop=True),
        X_pca.reset_index(drop=True),
        y.reset_index(drop=True)
    ],
    axis=1
)

print("Final Shape:", final_df.shape)

# ==========================
# Save Dataset
# ==========================

final_df.to_csv(
    "pca_features.csv",
    index=False
)

print("Explained Variance:",
      pca.explained_variance_ratio_.sum())

print("PCA Completed Successfully!")