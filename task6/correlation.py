import pandas as pd
import numpy as np

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("title_features_with_embeddings.csv")

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
# Separate Numeric Features
# ==========================

numeric_df = df.drop(columns=non_numeric)

# Keep target
clicks = numeric_df["clicks"]

# Remove target for correlation
X = numeric_df.drop(columns=["clicks"])

# ==========================
# Correlation Matrix
# ==========================

corr_matrix = X.corr()

# Upper Triangle
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# ==========================
# Find Strong Correlated Features
# ==========================

threshold = 0.95

to_drop = [
    column
    for column in upper.columns
    if any((upper[column] > threshold) | (upper[column] < -threshold))
]

print("Strong Correlated Features:", len(to_drop))

# Remove Features
X = X.drop(columns=to_drop)

print("Remaining Features:", X.shape)

# ==========================
# Add Back Target
# ==========================

X["clicks"] = clicks

# ==========================
# Add Back Non-Numeric Columns
# ==========================

final_df = pd.concat(
    [
        df[non_numeric],
        X
    ],
    axis=1
)

print("Final Shape:", final_df.shape)

# ==========================
# Save Dataset
# ==========================

final_df.to_csv(
    "features_after_correlation.csv",
    index=False
)

print("Correlation Analysis Completed Successfully!")