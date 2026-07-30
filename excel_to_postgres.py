import pandas as pd
import psycopg

# ==========================
# Read CSV File
# ==========================
df = pd.read_csv(
    r"C:\Users\suman\OneDrive\Desktop\Newsscraper\phase0_articles.csv"
)


# ==========================
# Connect PostgreSQL
# ==========================
conn = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="akc_dogs",
    user="postgres",
    password="5432"
)

cur = conn.cursor()


# ==========================
# Remove Old Data
# ==========================
cur.execute("""
    TRUNCATE TABLE phase0_articles;
""")


# ==========================
# Insert New 1500 Records
# ==========================
for _, row in df.iterrows():

    cur.execute("""
        INSERT INTO phase0_articles
        (
            source_url,
            publish_date,
            news_paper_name,
            title,
            first_paragraph
        )
        VALUES (%s, %s, %s, %s, %s)
    """,
    (
        row["source_url"],
        row["publish_date"],
        row["news_paper_name"],
        row["title"],
        row["first_paragraph"]
    ))


# ==========================
# Commit
# ==========================
conn.commit()

print(f"{len(df)} records inserted successfully!")


# ==========================
# Close
# ==========================
cur.close()
conn.close()