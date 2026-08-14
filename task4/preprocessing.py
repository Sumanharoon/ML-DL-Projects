import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ==========================
# Download NLTK Resources
# ==========================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("merge.csv")

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:")
print(df.shape)


# ==========================
# Check Missing Values
# ==========================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================
# Data Cleaning
# ==========================
print("Duplicate rows:", df.duplicated().sum())

print("Missing title:", df["title"].isnull().sum())

print("Missing paragraph:", df["first_paragraph"].isnull().sum())
# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where title or paragraph is missing
df = df.dropna(subset=["title", "first_paragraph"])


print("\nShape After Removing Duplicates & Missing Values:")
print(df.shape)


# ==========================
# Text Preprocessing
# ==========================

stop_words = set(stopwords.words("english"))


def clean_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Join words back
    cleaned_text = " ".join(words)

    return cleaned_text


# Apply preprocessing on article text

df["clean_content"] = df["first_paragraph"].apply(clean_text)


# Also clean titles

df["clean_title"] = df["title"].apply(clean_text)


# ==========================
# Save Processed Data
# ==========================

df.to_csv("processed_articles.csv", index=False)


print("\nPreprocessing Completed Successfully!")
print("Processed file created: processed_articles.csv")


# ==========================
# Preview Result
# ==========================

print("\nProcessed Data Preview:")
print(df[["title", "clean_title", "clean_content"]].head())