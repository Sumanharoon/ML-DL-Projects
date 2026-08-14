import feedparser
import pandas as pd
import re
from html import unescape

# ---------------------------------
# Clean Text Function
# ---------------------------------
def clean_text(text):
    if not text:
        return ""

    text = unescape(text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove new lines and tabs
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------
# DW RSS Feeds
# ---------------------------------
rss_feeds = [
    "https://rss.dw.com/xml/rss-en-all",
    "https://rss.dw.com/xml/rss-en-world",
    "https://rss.dw.com/xml/rss-en-business",
    "https://rss.dw.com/xml/rss-en-science",
    "https://rss.dw.com/xml/rss-en-top-stories",
]

articles = []

# ---------------------------------
# Scrape RSS
# ---------------------------------
for url in rss_feeds:

    print(f"Scraping: {url}")

    feed = feedparser.parse(url)

    for entry in feed.entries:

        articles.append({
            "source_url": entry.get("link", ""),
            "publish_date": clean_text(entry.get("published", "")),
            "news_paper_name": "DW",
            "title": clean_text(entry.get("title", "")),
            "first_paragraph": clean_text(entry.get("summary", ""))
        })


# ---------------------------------
# Create DataFrame
# ---------------------------------
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

# Reset index
df.reset_index(drop=True, inplace=True)

# Save CSV
df.to_csv(
    "dw_articles.csv",
    index=False,
    encoding="utf-8-sig"
)

print("=" * 50)
print(f"Total Unique Articles: {len(df)}")
print("Saved as: dw_articles.csv")
print("=" * 50)