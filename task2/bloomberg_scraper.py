import feedparser
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

url = "https://feeds.bloomberg.com/markets/news.rss"
news_feed = feedparser.parse(url)

articles = []

for item in news_feed.entries[:10]:
    published_time = item.get("published")

    if published_time:
        parsed_time = datetime(*item.published_parsed[:6], tzinfo=ZoneInfo("UTC"))
        new_york_time = parsed_time.astimezone(ZoneInfo("America/New_York"))
        publish_date = new_york_time.strftime("%Y-%m-%d %I:%M:%S %p")
    else:
        publish_date = datetime.now(
            ZoneInfo("America/New_York")
        ).strftime("%Y-%m-%d %I:%M:%S %p")

    articles.append({
        "source_url": item.link,
        "publish_date": publish_date,
        "news_paper_name": "Bloomberg",
        "title": item.title,
        "first_paragraph": item.get("summary", "Not available")
    })

with open("bloomberg_news.csv", "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "source_url",
        "publish_date",
        "news_paper_name",
        "title",
        "first_paragraph"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(articles)
publish_date = new_york_time.strftime("%m/%d/%Y %I:%M %p")
print("Bloomberg scraping completed!")
print("Total articles:", len(articles))
