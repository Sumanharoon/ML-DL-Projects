import feedparser
import csv
from datetime import datetime

url = "https://www.cbsnews.com/latest/rss/main"
news_feed = feedparser.parse(url)

with open("cbs_news.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "source_url",
        "publish_date",
        "news_paper_name",
        "title",
        "first_paragraph"
    ])

    for item in news_feed.entries[:10]:
        writer.writerow([
            item.link,
            item.get("published", datetime.now().strftime("%Y-%m-%d")),
            "CBS News",
            item.title,
            item.get("summary", "Not available")
        ])

print("Scraping completed!")