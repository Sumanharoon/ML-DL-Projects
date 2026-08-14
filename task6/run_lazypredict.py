import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor

df = pd.read_csv("pca_features.csv")

X = df.filter(regex="^PC")
y = df["clicks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = LazyRegressor(verbose=0, ignore_warnings=True)

models, predictions = reg.fit(
    X_train, X_test, y_train, y_test
)

print(models)

print("LazyPredict completed.")

print(models.head())

models.to_csv("lazypredict_results.csv", index=True)

print("CSV saved successfully.")