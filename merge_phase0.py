import pandas as pd

# =====================================================
# CSV Files to Merge
# =====================================================

csv_files = [
    "reuters_news_1500.csv",
    "bloomberg_news.csv",
    "cbs_news.csv",
    "guardian_news.csv",
    "cnbc_news.csv",
    "bbc_news.csv",
    "ap_news.csv",
    "cnn_articles.csv",
    "dw_articles.csv",
    "newsweek_google_news.csv"
]

# Required columns
required_columns = [
    "source_url",
    "publish_date",
    "news_paper_name",
    "title",
    "first_paragraph"
]

all_data = []

print("=" * 60)
print("Reading CSV Files...")
print("=" * 60)


# =====================================================
# Read CSV Files
# =====================================================

for file in csv_files:

    try:
        df = pd.read_csv(file)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Check columns
        missing_cols = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_cols:
            print(f"❌ {file} skipped - Missing: {missing_cols}")
            continue


        # Keep required columns only
        df = df[required_columns]


        # Convert to string and clean
        for col in required_columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )


        # Remove empty articles
        df = df[
            (df["source_url"] != "") &
            (df["title"] != "")
        ]


        # Remove duplicates inside file
        df.drop_duplicates(
            subset=["source_url"],
            keep="first",
            inplace=True
        )


        all_data.append(df)

        print(f"✓ {file:<30} {len(df)} articles")


    except FileNotFoundError:
        print(f"❌ {file} not found")


    except Exception as e:
        print(f"❌ Error in {file}: {e}")



# =====================================================
# Merge All Files
# =====================================================

if not all_data:
    print("No CSV files loaded.")
    exit()


merged_df = pd.concat(
    all_data,
    ignore_index=True
)



# =====================================================
# Remove Duplicate Articles
# =====================================================

merged_df.drop_duplicates(
    subset=["source_url"],
    keep="first",
    inplace=True
)



# =====================================================
# Remove Blank Rows
# =====================================================

merged_df = merged_df[
    (merged_df["source_url"] != "") &
    (merged_df["title"] != "")
]



print("\nBefore Limiting:", len(merged_df))



# =====================================================
# Keep Maximum 1500 Articles
# Balanced by Newspaper
# =====================================================

if len(merged_df) > 1500:

    sampled_data = []

    total_articles = len(merged_df)

    for source, group in merged_df.groupby("news_paper_name"):

        source_count = len(group)

        # proportional allocation
        take = int(
            (source_count / total_articles) * 1500
        )

        if take > 0:
            sampled_data.append(
                group.sample(
                    n=take,
                    random_state=42
                )
            )


    merged_df = pd.concat(
        sampled_data,
        ignore_index=True
    )


    # If less than 1500 due to rounding, fill remaining
    if len(merged_df) < 1500:

        remaining = 1500 - len(merged_df)

        extra = pd.concat(all_data)

        extra = extra[
            ~extra["source_url"].isin(
                merged_df["source_url"]
            )
        ]

        extra = extra.sample(
            n=remaining,
            random_state=42
        )

        merged_df = pd.concat(
            [merged_df, extra],
            ignore_index=True
        )


    # Exactly 1500
    merged_df = merged_df.head(1500)



# =====================================================
# Sort and Reset Index
# =====================================================

merged_df.sort_values(
    by="news_paper_name",
    inplace=True
)

merged_df.reset_index(
    drop=True,
    inplace=True
)



# =====================================================
# Save Final CSV
# =====================================================

output_file = "phase0_articles.csv"

merged_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)



# =====================================================
# Summary
# =====================================================

print("\n" + "=" * 60)
print("Merge Completed Successfully")
print("=" * 60)

print(f"Total Unique Articles : {len(merged_df)}")
print(f"Output File           : {output_file}")

print("\nArticles per Source:")
print(
    merged_df["news_paper_name"]
    .value_counts()
)

print("=" * 60)