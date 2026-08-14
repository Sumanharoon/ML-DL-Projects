import pandas as pd
import psycopg

df = pd.read_csv("processed_articles.csv")

conn = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="akc_dogs",
    user="postgres",
    password="5432"
)

cur = conn.cursor()

for _, row in df.iterrows():
    cur.execute("""
    INSERT INTO processed_articles
    (
        source_url,
        publish_date,
        news_paper_name,
        title,
        first_paragraph,
        clean_content,
        clean_title
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s)
    """,
    (
        row["source_url"],
        row["publish_date"],
        row["news_paper_name"],
        row["title"],
        row["first_paragraph"],
        row["clean_content"],
        row["clean_title"]
    ))

conn.commit()

cur.close()
conn.close()

print("Data uploaded successfully")