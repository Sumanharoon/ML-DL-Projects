# generate_steps_csv.py
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Step 1: Raw Dataset
df_raw = pd.read_excel('News Classification Labeling.xlsx')
df_raw.to_csv('step1_raw_dataset.csv', index=False)
print("Saved: step1_raw_dataset.csv")

# Step 2: Data Understanding
step2_df = pd.DataFrame({
    'Metric': ['Total Rows', 'Total Columns', 'URL Nulls', 'Title Nulls', 'Duplicate Titles'],
    'Value': [len(df_raw), len(df_raw.columns), df_raw['URL'].isnull().sum(), df_raw['Title'].isnull().sum(), df_raw['Title'].duplicated().sum()]
})
step2_df.to_csv('step2_data_understanding.csv', index=False)
print("Saved: step2_data_understanding.csv")

# Step 3: Data Cleaning
df_clean = df_raw.drop_duplicates(subset=['Title']).dropna(subset=['Title']).copy()
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df_clean['Title_clean'] = df_clean['Title'].apply(clean_text)
df_clean.to_csv('step3_data_cleaning.csv', index=False)
print("Saved: step3_data_cleaning.csv")

# Step 4: Feature Engineering
df_clean['char_count'] = df_clean['Title'].apply(len)
df_clean['word_count'] = df_clean['Title'].apply(lambda x: len(x.split()))

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_clean['Title_clean']).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()])

step4_df = pd.concat([df_clean[['Title', 'char_count', 'word_count']].reset_index(drop=True), tfidf_df], axis=1)
step4_df.to_csv('step4_feature_engineering.csv', index=False)
print("Saved: step4_feature_engineering.csv")

# Step 5: Candidate Features (SVD Reduction)
svd = TruncatedSVD(n_components=20, random_state=42)
svd_features = svd.fit_transform(tfidf_matrix)
svd_df = pd.DataFrame(svd_features, columns=[f"svd_comp_{i+1}" for i in range(20)])
svd_df.to_csv('step5_candidate_features.csv', index=False)
print("Saved: step5_candidate_features.csv")

# Step 6: Feature Selection
svd_df.to_csv('step6_feature_selection.csv', index=False)
print("Saved: step6_feature_selection.csv")

# Step 7: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(svd_features)
scaled_df = pd.DataFrame(X_scaled, columns=[f"scaled_comp_{i+1}" for i in range(20)])
scaled_df.to_csv('step7_feature_scaling.csv', index=False)
print("Saved: step7_feature_scaling.csv")

# Step 8: Exploratory Data Analysis (EDA)
eda_summary = pd.DataFrame({
    'Metric': ['Total Variance Explained', 'Mean Scaled Range Min', 'Mean Scaled Range Max'],
    'Value': [f"{np.sum(svd.explained_variance_ratio_):.4f}", f"{scaled_df.mean().min():.4f}", f"{scaled_df.mean().max():.4f}"]
})
eda_summary.to_csv('step8_eda_summary.csv', index=False)
print("Saved: step8_eda_summary.csv")

# Step 9 & 10: Candidate Models & Evaluation
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
agg_clustering = AgglomerativeClustering(n_clusters=5)

clusters_km = kmeans.fit_predict(X_scaled)
clusters_agg = agg_clustering.fit_predict(X_scaled)

eval_df = pd.DataFrame({
    'Model': ['K-Means', 'Agglomerative Clustering'],
    'Silhouette_Score': [silhouette_score(X_scaled, clusters_km), silhouette_score(X_scaled, clusters_agg)],
    'Calinski_Harabasz_Score': [calinski_harabasz_score(X_scaled, clusters_km), calinski_harabasz_score(X_scaled, clusters_agg)]
})
eval_df.to_csv('step9_10_model_evaluation.csv', index=False)
print("Saved: step9_10_model_evaluation.csv")

# Step 11: Cluster Profiling
df_clean['Cluster'] = clusters_km
terms = tfidf.get_feature_names_out()

profile_list = []
for cluster_id in range(5):
    cluster_docs = df_clean[df_clean['Cluster'] == cluster_id]['Title_clean']
    cluster_tfidf = tfidf.transform(cluster_docs).mean(axis=0)
    top_indices = np.argsort(np.asarray(cluster_tfidf).ravel())[::-1][:5]
    top_terms = ", ".join([terms[idx] for idx in top_indices])
    profile_list.append({
        'Cluster_ID': cluster_id,
        'Article_Count': len(cluster_docs),
        'Top_Keywords': top_terms
    })

profile_df = pd.DataFrame(profile_list)
profile_df.to_csv('step11_cluster_profiling.csv', index=False)
print("Saved: step11_cluster_profiling.csv")

# Step 12: Deployment
df_clean.to_csv('step12_final_deployment_output.csv', index=False)
print("Saved: step12_final_deployment_output.csv")