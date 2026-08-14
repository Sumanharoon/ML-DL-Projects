import pandas as pd
import json

csv_file = "phase0_articles.csv"
json_file = "combined.json"

df = pd.read_csv(csv_file)

data = df.to_dict(orient="records")

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("JSON file created successfully!")