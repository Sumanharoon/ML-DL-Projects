import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. LOAD DATA & CLEANING
# ==========================================
df = pd.read_csv('merged_deduplicated_data.csv')

df['clean_title'] = df['Title'].astype(str).str.strip().str.lower()
df['url_clean'] = df['URL'].astype(str).str.lower().str.replace(r'https?://', '', regex=True).str.replace(r'www\.', '', regex=True)

# Remove Conflicting Labels
conflicts = df.groupby('clean_title')['Category'].nunique()
clean_df = df[~df['clean_title'].isin(conflicts[conflicts > 1].index)].drop_duplicates('clean_title').copy()

# Merge Categories
clean_df['Category'] = clean_df['Category'].replace({
    'Business': 'Business & Market', 
    'Market': 'Business & Market'
})

# Split 90% Train / 10% Test
train_df, test_df = train_test_split(
    clean_df, test_size=0.10, random_state=42, stratify=clean_df['Category']
)

# ==========================================
# 2. ENHANCED TARGETED MINORITY SYNTHESIS
# ==========================================
extra_samples = [
    # Energy Samples
    {'clean_title': 'crude oil prices surge amid global supply chain disruptions', 'url_clean': 'reuters.com/business/energy/crude-oil-prices-surge', 'Category': 'Energy'},
    {'clean_title': 'solar energy adoption expands with new renewable grid projects', 'url_clean': 'bloomberg.com/news/energy/solar-power-expansion', 'Category': 'Energy'},
    {'clean_title': 'natural gas pipeline supply tightens ahead of winter demand', 'url_clean': 'cnbc.com/energy/natural-gas-prices-oil', 'Category': 'Energy'},
    {'clean_title': 'clean energy transition accelerates with offshore wind farms', 'url_clean': 'wsj.com/articles/renewable-energy-wind-solar', 'Category': 'Energy'},
    {'clean_title': 'opec oil production cuts impact international fuel markets', 'url_clean': 'ft.com/content/opec-oil-output-cuts', 'Category': 'Energy'},
    {'clean_title': 'nuclear power generation rises as electricity demand peaks', 'url_clean': 'reuters.com/energy/nuclear-power-electric-grid', 'Category': 'Energy'},
    {'clean_title': 'green hydrogen projects receive billion dollar investment boost', 'url_clean': 'bloomberg.com/energy/hydrogen-power-clean-tech', 'Category': 'Energy'},
    {'clean_title': 'electricity grid modernization speeds up renewable power access', 'url_clean': 'cnbc.com/energy/electric-grid-power-supply', 'Category': 'Energy'},
    {'clean_title': 'petroleum drillers expand offshore rigs as fuel demand rebounds', 'url_clean': 'reuters.com/business/energy/petroleum-oil-rigs', 'Category': 'Energy'},
    {'clean_title': 'renewable energy subsidies boost wind turbine and solar cell manufacturing', 'url_clean': 'wsj.com/articles/energy-subsidies-wind-solar', 'Category': 'Energy'},
    {'clean_title': 'fossil fuel emissions spark global energy transition debate', 'url_clean': 'bloomberg.com/energy/emissions-oil-gas-power', 'Category': 'Energy'},
    {'clean_title': 'geothermal energy projects unlock clean power for urban grids', 'url_clean': 'cnbc.com/energy/geothermal-power-grid-green', 'Category': 'Energy'},

    # Health Samples
    {'clean_title': 'fda approves breakthrough therapy for chronic disease treatment', 'url_clean': 'reuters.com/business/healthcare-pharmaceuticals/fda-approval', 'Category': 'Health'},
    {'clean_title': 'new vaccine clinical trials show strong protection against virus', 'url_clean': 'bloomberg.com/news/health/vaccine-trial-results', 'Category': 'Health'},
    {'clean_title': 'cancer research study reveals novel target for immunotherapy', 'url_clean': 'wsj.com/articles/medical-health-cancer-research', 'Category': 'Health'},
    {'clean_title': 'hospital health systems report lower clinical infection rates', 'url_clean': 'cnbc.com/health/hospital-patient-care-medical', 'Category': 'Health'},
    {'clean_title': 'pharmaceutical companies launch global public health initiative', 'url_clean': 'ft.com/content/pharma-health-medicine-drugs', 'Category': 'Health'},
    {'clean_title': 'mental health treatment access improves with digital therapy apps', 'url_clean': 'reuters.com/health/mental-healthcare-medical-apps', 'Category': 'Health'},
    {'clean_title': 'cardiovascular disease risk factors identified in comprehensive study', 'url_clean': 'bloomberg.com/health/heart-disease-medical-study', 'Category': 'Health'},
    {'clean_title': 'medical device startup secures funding for diagnostic health tools', 'url_clean': 'cnbc.com/health/medical-devices-healthcare-funding', 'Category': 'Health'},
    {'clean_title': 'biotech firm develops oral drug for rare genetic diseases', 'url_clean': 'reuters.com/business/healthcare-pharmaceuticals/biotech-drug', 'Category': 'Health'},
    {'clean_title': 'clinical healthcare trials evaluate novel antibiotic resistance treatments', 'url_clean': 'wsj.com/articles/health-medicine-antibiotic-trials', 'Category': 'Health'},
    {'clean_title': 'pediatric medical care improves through expanded hospital funding', 'url_clean': 'bloomberg.com/health/pediatric-care-hospitals', 'Category': 'Health'},
    {'clean_title': 'neurology research breakthrough offers hope for alzheimers treatment', 'url_clean': 'cnbc.com/health/neurology-alzheimers-drug-study', 'Category': 'Health'}
]

extra_df = pd.DataFrame(extra_samples)
train_df_balanced = pd.concat([train_df[['clean_title', 'url_clean', 'Category']], extra_df], ignore_index=True)

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
title_word = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', sublinear_tf=True, min_df=1)
title_char = TfidfVectorizer(ngram_range=(2, 6), analyzer='char_wb', sublinear_tf=True, min_df=1)

url_word = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', token_pattern=r'(?u)\b\w+\b', sublinear_tf=True, min_df=1)
url_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', sublinear_tf=True, min_df=1)

X_train = hstack([
    title_word.fit_transform(train_df_balanced['clean_title']),
    title_char.fit_transform(train_df_balanced['clean_title']),
    url_word.fit_transform(train_df_balanced['url_clean']),
    url_char.fit_transform(train_df_balanced['url_clean'])
])

X_test = hstack([
    title_word.transform(test_df['clean_title']),
    title_char.transform(test_df['clean_title']),
    url_word.transform(test_df['url_clean']),
    url_char.transform(test_df['url_clean'])
])

# ==========================================
# 4. ENSEMBLE TRAINING
# ==========================================
m1 = LinearSVC(class_weight='balanced', C=0.35, random_state=42, dual='auto')
m2 = LogisticRegression(class_weight='balanced', C=3.0, max_iter=1000, random_state=42)

model = VotingClassifier(estimators=[('svc', m1), ('lr', m2)], voting='hard')
model.fit(X_train, train_df_balanced['Category'])

# ==========================================
# 5. EVALUATION
# ==========================================
predictions = model.predict(X_test)
score = accuracy_score(test_df['Category'], predictions) * 100

print("\n" + "="*50)
print(f"FINAL BALANCED ML ACCURACY: {score:.2f}%")
print("="*50)
print(classification_report(test_df['Category'], predictions))

# ==========================================
# 6. GENERATE & SAVE CSV OUTPUTS
# ==========================================

# 1. Predictions ki CSV File Banana
results_df = test_df[['Title', 'URL', 'Category']].copy()
results_df.rename(columns={'Category': 'Actual_Category'}, inplace=True)
results_df['Predicted_Category'] = predictions

# Correct / Incorrect Flag (Matching Check)
results_df['Is_Correct'] = results_df['Actual_Category'] == results_df['Predicted_Category']

# Save Predictions CSV
results_df.to_csv('predictions_output.csv', index=False)
print("\n[SUCCESS] Predictions CSV saved as 'predictions_output.csv'!")

# 2. Classification Report (Metrics) ki CSV File Banana
report_dict = classification_report(test_df['Category'], predictions, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df.rename(columns={'index': 'Category_or_Metric'}, inplace=True)

# Save Metrics CSV
report_df.to_csv('category_metrics_report.csv', index=False)
print("[SUCCESS] Classification Report CSV saved as 'category_metrics_report.csv'!")