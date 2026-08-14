import feedparser
import pandas as pd
import urllib.parse
import re
from html import unescape
import time

# ==========================================
# SETTINGS
# ==========================================

SITE = "newsweek.com"          # Change to any website
MAX_ARTICLES = 300

queries = [
    "",
    "world",
    "business",
    "technology",
    "politics",
    "health",
    "science",
    "sports",
    "entertainment",
    "economy",
    "finance"
]

# ==========================================
# CLEAN FUNCTION
# ==========================================

def clean_text(text):
    if text is None:
        return ""

    text = unescape(text)

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# SCRAPER
# ==========================================

articles = []

for q in queries:

    search_query = f"site:{SITE} {q}"

    rss_url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(search_query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )

    print("Reading:", rss_url)

    feed = feedparser.parse(rss_url)

    for entry in feed.entries:

        articles.append({

            "source_url": entry.get("link", ""),

            "publish_date": entry.get("published", ""),

            "news_paper_name": SITE.replace(".com","").upper(),

            "title": clean_text(entry.get("title","")),

            "first_paragraph": clean_text(entry.get("summary",""))

        })

    time.sleep(1)

# ==========================================
# DATAFRAME
# ==========================================

df = pd.DataFrame(
    articles,
    columns=[
        "source_url",
        "publish_date",
        "news_paper_name",
        "title",
        "first_paragraph"
    ]
)

# Remove duplicates
df.drop_duplicates(subset=["source_url"], inplace=True)

# Remove empty rows
df = df[df["source_url"].str.strip() != ""]
df = df[df["title"].str.strip() != ""]

# Limit rows
df = df.head(MAX_ARTICLES)

# Reset index
df.reset_index(drop=True, inplace=True)

# Save
filename = SITE.replace(".com","") + "_google_news.csv"

df.to_csv(
    filename,
    index=False,
    encoding="utf-8-sig"
)

print("="*60)
print("Unique Articles:", len(df))
print("Saved:", filename)
print("="*60)