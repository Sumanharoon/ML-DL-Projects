import pandas as pd
import psycopg

# ==========================
# Read CSV
# ==========================
df = pd.read_csv("processed_articles_with_embeddings.csv")

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
# Create Table Automatically
# ==========================

columns = []

for col in df.columns:
    if col.startswith("embedding_"):
        columns.append(f'"{col}" DOUBLE PRECISION')
    else:
        columns.append(f'"{col}" TEXT')

create_table_query = f"""
CREATE TABLE IF NOT EXISTS sentence_embeddings (
    {', '.join(columns)}
);
"""

cur.execute(create_table_query)
conn.commit()

print("Table created successfully!")

# ==========================
# Insert Data
# ==========================

column_names = ",".join(f'"{col}"' for col in df.columns)
placeholders = ",".join(["%s"] * len(df.columns))

insert_query = f"""
INSERT INTO sentence_embeddings ({column_names})
VALUES ({placeholders})
"""

for _, row in df.iterrows():
    cur.execute(insert_query, tuple(row))

conn.commit()

cur.close()
conn.close()

print("Sentence Embeddings uploaded successfully!")