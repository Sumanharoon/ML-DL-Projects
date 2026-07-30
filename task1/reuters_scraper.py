import feedparser
import csv
import re
import urllib.parse
from html import unescape

queries = [
    "Reuters",
    "Reuters world news",
    "Reuters breaking news",
    "Reuters politics",
    "Reuters business",
    "Reuters technology",
    "Reuters science",
    "Reuters health",
    "Reuters sports",
    "Reuters economy",
    "Reuters finance",
    "Reuters markets",
    "Reuters climate",
    "Reuters energy",
    "Reuters Asia",
    "Reuters Europe",
    "Reuters Africa",
    "Reuters Middle East",
    "Reuters Americas",
    "Reuters US",
    "Reuters China",
    "Reuters India",
    "Reuters Ukraine",
    "Reuters Russia",
    "Reuters elections",
    "Reuters AI",
    "Reuters cryptocurrency",
    "Reuters banking",
    "Reuters companies",
    "Reuters stocks",
    "Reuters oil",
    "Reuters environment"
]

articles = []
seen_urls = set()

for query in queries:

    rss_url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )

    feed = feedparser.parse(rss_url)

    for item in feed.entries:

        url = item.get("link", "")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        summary = unescape(item.get("summary", ""))
        summary = re.sub(r"<[^>]+>", "", summary)
        summary = re.sub(r"\s+", " ", summary).strip()

        articles.append({
            "source_url": url,
            "publish_date": item.get("published", ""),
            "news_paper_name": "Reuters",
            "title": item.get("title", "").strip(),
            "first_paragraph": summary
        })

        # stop after 1500
        if len(articles) >= 1500:
            break

    if len(articles) >= 1500:
        break


with open(
    "reuters_news_1500.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source_url",
            "publish_date",
            "news_paper_name",
            "title",
            "first_paragraph"
        ]
    )

    writer.writeheader()
    writer.writerows(articles)


print("Total Unique Articles:", len(articles))