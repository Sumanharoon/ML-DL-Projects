import pandas as pd
import numpy as np
import re
import string
import nltk
import spacy
from textblob import TextBlob

from nltk.corpus import stopwords

# ==========================
# Download NLTK Resources
# ==========================

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("processed_articles.csv")

# Keep only title
df = df[["title"]].copy()

print("Dataset Loaded Successfully")
print(df.head())

# ==========================
# Generate Random Clicks
# ==========================

np.random.seed(42)

df["clicks"] = np.random.randint(
    5000,
    100001,
    len(df)
)

print("\nRandom Clicks Generated")
# ==========================
# SEO Keywords
# ==========================

seo_keywords = [

    "ai",
    "chatgpt",
    "google",
    "microsoft",
    "apple",
    "tesla",
    "bitcoin",
    "crypto",
    "pakistan",
    "india",
    "china",
    "usa",
    "covid",
    "election",
    "war",
    "breaking",
    "latest",
    "update"

]

# ==========================
# Clickbait Words
# ==========================

clickbait_words = [

    "breaking",
    "shocking",
    "exclusive",
    "secret",
    "revealed",
    "must",
    "watch",
    "finally",
    "amazing",
    "alert"

]

# ==========================
# Curiosity Words
# ==========================

curiosity_words = [

    "why",
    "how",
    "what",
    "who",
    "when",
    "where"

]

# ==========================
# Urgency Words
# ==========================

urgency_words = [

    "today",
    "now",
    "latest",
    "urgent",
    "live",
    "alert"

]

# ==========================
# Brand Names
# ==========================

brands = [

    "google",
    "apple",
    "microsoft",
    "tesla",
    "amazon",
    "meta",
    "openai",
    "intel",
    "nvidia",
    "samsung"

]

# ==========================
# Countries
# ==========================

countries = [

    "pakistan",
    "india",
    "china",
    "usa",
    "uk",
    "russia",
    "france",
    "germany",
    "canada",
    "japan"

]
# ==========================
# Word Count
# ==========================

df["word_count"] = df["title"].apply(
    lambda x: len(str(x).split())
)

# ==========================
# Character Count
# ==========================

df["char_count"] = df["title"].str.len()

# ==========================
# Average Word Length
# ==========================

df["avg_word_length"] = df["title"].apply(

    lambda x:

    sum(len(word) for word in str(x).split())

    /

    max(len(str(x).split()),1)

)

# ==========================
# Stopword Count
# ==========================

df["stopword_count"] = df["title"].apply(

    lambda x:

    sum(

        word in stop_words

        for word in str(x).lower().split()

    )

)

# ==========================
# Stopword Ratio
# ==========================

df["stopword_ratio"] = (

    df["stopword_count"]

    /

    df["word_count"].replace(0,1)

)

# ==========================
# Unique Word Count
# ==========================

df["unique_word_count"] = df["title"].apply(

    lambda x:

    len(

        set(

            str(x).lower().split()

        )

    )

)

# ==========================
# Lexical Diversity
# ==========================

df["lexical_diversity"] = (

    df["unique_word_count"]

    /

    df["word_count"].replace(0,1)

)
# ==========================
# Digit Count
# ==========================

df["digit_count"] = df["title"].apply(
    lambda x: sum(c.isdigit() for c in str(x))
)

# ==========================
# Contains Number
# ==========================

df["contains_number"] = (
    df["digit_count"] > 0
).astype(int)

# ==========================
# Uppercase Word Count
# ==========================

df["uppercase_word_count"] = df["title"].apply(
    lambda x: sum(
        word.isupper()
        for word in str(x).split()
    )
)

# ==========================
# Punctuation Count
# ==========================

df["punctuation_count"] = df["title"].apply(
    lambda x: sum(
        c in string.punctuation
        for c in str(x)
    )
)

# ==========================
# Question Title
# ==========================

df["is_question"] = df["title"].apply(
    lambda x: int("?" in str(x))
)

# ==========================
# Exclamation Title
# ==========================

df["has_exclamation"] = df["title"].apply(
    lambda x: int("!" in str(x))
)

# ==========================
# Starts With Number
# ==========================

df["starts_with_number"] = df["title"].apply(
    lambda x: int(bool(re.match(r"^\d", str(x))))
)

# ==========================
# Long Word Count (>6 letters)
# ==========================

df["long_word_count"] = df["title"].apply(
    lambda x: sum(
        len(word) > 6
        for word in str(x).split()
    )
)

# ==========================
# Short Word Count (<=3 letters)
# ==========================

df["short_word_count"] = df["title"].apply(
    lambda x: sum(
        len(word) <= 3
        for word in str(x).split()
    )
)
# ==========================
# SEO Keyword Count
# ==========================

df["seo_keyword_count"] = df["title"].apply(

    lambda x:

    sum(

        keyword in str(x).lower()

        for keyword in seo_keywords

    )

)

# ==========================
# SEO Keyword Density
# ==========================

df["seo_keyword_density"] = (

    df["seo_keyword_count"]

    /

    df["word_count"].replace(0,1)

)

# ==========================
# SEO Length Score
# Ideal Length = 50-60 Characters
# ==========================

df["seo_length_score"] = df["char_count"].apply(

    lambda x:

    1 if 50 <= x <= 60 else 0

)

# ==========================
# Primary Keyword Position
# ==========================

def keyword_first(title):

    words = str(title).lower().split()

    if len(words) == 0:
        return 0

    return int(words[0] in seo_keywords)

df["primary_keyword_first"] = df["title"].apply(keyword_first)
# ==========================
# Clickbait Score
# ==========================

df["clickbait_score"] = df["title"].apply(

    lambda x:

    sum(

        word in str(x).lower()

        for word in clickbait_words

    )

)

# ==========================
# Curiosity Score
# ==========================

df["curiosity_score"] = df["title"].apply(

    lambda x:

    sum(

        word in str(x).lower()

        for word in curiosity_words

    )

)

# ==========================
# Urgency Score
# ==========================

df["urgency_score"] = df["title"].apply(

    lambda x:

    sum(

        word in str(x).lower()

        for word in urgency_words

    )

)
# ==========================
# Brand Mentions
# ==========================

df["brand_mentions"] = df["title"].apply(

    lambda x:

    sum(

        brand in str(x).lower()

        for brand in brands

    )

)

# ==========================
# Country Mentions
# ==========================

df["country_mentions"] = df["title"].apply(

    lambda x:

    sum(

        country in str(x).lower()

        for country in countries

    )

)
print(df.head())

print(df.columns)
# ==========================
# Load spaCy Model
# ==========================

nlp = spacy.load("en_core_web_sm")
# ==========================
# Sentiment Analysis
# ==========================

df["sentiment_polarity"] = df["title"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)

df["sentiment_subjectivity"] = df["title"].apply(
    lambda x: TextBlob(str(x)).sentiment.subjectivity
)

# ==========================
# Person Count
# ==========================

def person_count(text):

    doc = nlp(str(text))

    return sum(ent.label_ == "PERSON" for ent in doc.ents)

df["person_count"] = df["title"].apply(person_count)

# ==========================
# Organization Count
# ==========================

def org_count(text):

    doc = nlp(str(text))

    return sum(ent.label_ == "ORG" for ent in doc.ents)

df["organization_count"] = df["title"].apply(org_count)

# ==========================
# Location Count
# ==========================

def location_count(text):

    doc = nlp(str(text))

    return sum(
        ent.label_ in ["GPE", "LOC"]
        for ent in doc.ents
    )

df["location_count"] = df["title"].apply(location_count)

# ==========================
# Noun Count
# ==========================

def noun_count(text):

    doc = nlp(str(text))

    return sum(token.pos_ == "NOUN" for token in doc)

df["noun_count"] = df["title"].apply(noun_count)

# ==========================
# Verb Count
# ==========================

def verb_count(text):

    doc = nlp(str(text))

    return sum(token.pos_ == "VERB" for token in doc)

df["verb_count"] = df["title"].apply(verb_count)

# ==========================
# Adjective Count
# ==========================

def adjective_count(text):

    doc = nlp(str(text))

    return sum(token.pos_ == "ADJ" for token in doc)

df["adjective_count"] = df["title"].apply(adjective_count)

# ==========================
# Proper Noun Count
# ==========================

def proper_noun_count(text):

    doc = nlp(str(text))

    return sum(token.pos_ == "PROPN" for token in doc)

df["proper_noun_count"] = df["title"].apply(proper_noun_count)

# ==========================
# Improved Category Prediction
# ==========================

def predict_category(title):

    title = str(title).lower()

    sports = [
        "cricket", "football", "soccer", "match", "player",
        "league", "cup", "goal", "t20", "odi", "test",
        "series", "tournament", "ipl", "fifa", "nba",
        "tennis", "hockey", "olympics", "championship"
    ]

    business = [
        "stock", "market", "business", "economy",
        "bank", "bitcoin", "crypto", "finance",
        "trade", "inflation", "investment"
    ]

    technology = [
        "ai", "artificial intelligence", "chatgpt",
        "google", "apple", "microsoft",
        "tesla", "openai", "technology",
        "software", "robot", "chip", "iphone"
    ]

    health = [
        "health", "covid", "virus", "hospital",
        "medicine", "doctor", "vaccine",
        "disease", "cancer", "flu"
    ]

    entertainment = [
        "movie", "film", "actor", "actress",
        "music", "celebrity", "hollywood",
        "bollywood", "netflix", "show"
    ]

    politics = [
        "election", "government", "president",
        "minister", "politics", "congress",
        "parliament", "vote", "senate",
        "prime minister"
    ]

    if any(word in title for word in sports):
        return "Sports"

    elif any(word in title for word in business):
        return "Business"

    elif any(word in title for word in technology):
        return "Technology"

    elif any(word in title for word in health):
        return "Health"

    elif any(word in title for word in entertainment):
        return "Entertainment"

    elif any(word in title for word in politics):
        return "Politics"

    else:
        return "General"


df["predicted_category"] = df["title"].apply(predict_category)
# ==========================
# Readability Level
# ==========================

def readability_level(words):

    if words <= 6:
        return "Easy"

    elif words <= 12:
        return "Medium"

    else:
        return "Hard"

df["readability_level"] = df["word_count"].apply(readability_level)

# ==========================
# Title Quality Score
# ==========================

def quality_score(row):

    score = 0

    # Ideal SEO Length
    if row["seo_length_score"] == 1:
        score += 2

    # SEO Keywords
    score += row["seo_keyword_count"]

    # Curiosity
    score += row["curiosity_score"]

    # Clickbait
    score += row["clickbait_score"]

    # Urgency
    score += row["urgency_score"]

    # Brand Mention
    if row["brand_mentions"] > 0:
        score += 2

    # Question Titles usually get higher CTR
    if row["is_question"] == 1:
        score += 1

    # Exclamation Titles
    if row["has_exclamation"] == 1:
        score += 1

    return score


df["title_quality_score"] = df.apply(
    quality_score,
    axis=1
)

# ==========================
# Title Length Category
# ==========================

def title_length(length):

    if length < 40:
        return "Short"

    elif length <= 60:
        return "Ideal"

    else:
        return "Long"

df["title_length_category"] = df["char_count"].apply(title_length)
df["predicted_category"] = df["title"].apply(predict_category)

df["title_quality_score"] = df.apply(
    quality_score,
    axis=1
)

df.to_csv("title_features.csv", index=False)

print("Feature Engineering Completed Successfully!")
print(df.shape)