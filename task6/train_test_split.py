import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("pca_features.csv")

print("Dataset Shape:", df.shape)

# ==========================
# Select Features and Target
# ==========================

X = df.filter(regex="^PC")

y = df["clicks"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Features:", X_train.shape)
print("Test Features :", X_test.shape)

print("\nTrain Target:", y_train.shape)
print("Test Target :", y_test.shape)

print("\nTrain-Test Split Completed Successfully!")