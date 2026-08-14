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

    # Remove new lines, tabs
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------
# CNN RSS Feeds
# ---------------------------------
rss_feeds = {
    "World": "http://rss.cnn.com/rss/edition_world.rss",
    "US": "http://rss.cnn.com/rss/cnn_us.rss",
    "Politics": "http://rss.cnn.com/rss/cnn_allpolitics.rss",
    "Business": "http://rss.cnn.com/rss/money_latest.rss",
    "Technology": "http://rss.cnn.com/rss/edition_technology.rss",
    "Health": "http://rss.cnn.com/rss/edition_health.rss",
    "Entertainment": "http://rss.cnn.com/rss/edition_entertainment.rss",
    "Travel": "http://rss.cnn.com/rss/edition_travel.rss",
    "Sport": "http://rss.cnn.com/rss/edition_sport.rss",
}

articles = []

# ---------------------------------
# Scrape RSS
# ---------------------------------
for category, url in rss_feeds.items():

    print(f"Scraping {category}...")

    feed = feedparser.parse(url)

    for entry in feed.entries:

        articles.append({
            "source_url": entry.get("link", ""),
            "publish_date": clean_text(entry.get("published", "")),
            "news_paper_name": "CNN",
            "title": clean_text(entry.get("title", "")),
            "first_paragraph": clean_text(entry.get("summary", ""))
        })

# ---------------------------------
# DataFrame
# ---------------------------------
df = pd.DataFrame(
    articles,
    columns=[
        "source_url",
        "publish_date",
        "news_paper_name",
        "title",
        "first_paragraph",
    ],
)

# Remove duplicates
df.drop_duplicates(subset=["source_url"], inplace=True)

# Remove empty rows
df = df[df["title"].str.strip() != ""]
df = df[df["source_url"].str.strip() != ""]

# Reset Index
df.reset_index(drop=True, inplace=True)

# Save CSV
df.to_csv(
    "cnn_articles.csv",
    index=False,
    encoding="utf-8-sig"
)

print("=" * 50)
print(f"Total Articles: {len(df)}")
print("Saved: cnn_articles.csv")
print("=" * 50)