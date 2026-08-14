import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("features_after_correlation.csv")

print("Original Shape:", df.shape)

# ==========================
# Separate Non-Numeric Columns
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
# Select Features (X)
# ==========================

X = df.drop(columns=non_numeric + ["clicks"])

print("Features Shape:", X.shape)

# ==========================
# Standard Scaling
# ==========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

# ==========================
# Add Back Columns
# ==========================

final_df = pd.concat(
    [
        df[non_numeric].reset_index(drop=True),
        X_scaled.reset_index(drop=True),
        y.reset_index(drop=True)
    ],
    axis=1
)

print("Scaled Shape:", final_df.shape)

# ==========================
# Save Dataset
# ==========================

final_df.to_csv(
    "scaled_features.csv",
    index=False
)

print("Feature Scaling Completed Successfully!")