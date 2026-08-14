import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("Step 4: Training Tuned XGBoost Model (Target: 90%+ Accuracy)...\n")

# 1. Load Dataset
input_file = "real_market_and_politician_data.csv"
df = pd.read_csv(input_file)
print(f" Loaded dataset '{input_file}' with {len(df)} rows.")

# 2. Strict Feature Selection (Excludes all Text Leakage Columns)
exclude_cols = ['Has_Politician_Trade', 'Politician_Name', 'Trade_Type', 'Amount_Tier', 'Trade_Timestamp_EST', 'Date_Only']
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

if 'Ticker' in df.columns:
    X_raw = pd.concat([df[['Ticker']], df[num_cols]], axis=1)
    X = pd.get_dummies(X_raw, columns=['Ticker'], drop_first=True)
else:
    X = df[num_cols].copy()

y = df['Has_Politician_Trade']

# 3. Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Tuned Fast XGBoost Classifier
model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=8,
    learning_rate=0.06,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1,
    tree_method='hist'
)

print(" Fitting Model...")
model.fit(X_train, y_train)

# 5. Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

# 6. Save Model & CSV
model.save_model("leakage_free_insider_model.json")

accuracy_data = [
    {"Metric_Name": "Training Accuracy", "Score": f"{train_acc * 100:.2f}%"},
    {"Metric_Name": "Testing / Validation Accuracy", "Score": f"{test_acc * 100:.2f}%"},
    {"Metric_Name": "Overall Precision", "Score": f"{precision * 100:.2f}%"},
    {"Metric_Name": "Overall Recall", "Score": f"{recall * 100:.2f}%"},
    {"Metric_Name": "F1-Score", "Score": f"{f1 * 100:.2f}%"},
    {"Metric_Name": "ROC-AUC Score", "Score": f"{roc_auc * 100:.2f}%"}
]

results_df = pd.DataFrame(accuracy_data)
eval_csv = "model_evaluation_results.csv"
results_df.to_csv(eval_csv, index=False)

# 7. Confusion Matrix Chart
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Normal (0)', 'Predicted Insider (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Insider (1)'])

plt.title('Leakage-Free XGBoost - Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Actual Category', fontsize=10, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=10, fontweight='bold')
plt.tight_layout()

cm_image_path = "confusion_matrix_chart.png"
plt.savefig(cm_image_path, dpi=300)
plt.close()

# 8. Terminal Summary Display
print("="*65)
print("             ACCURACY RESULTS SUMMARY                           ")
print("="*65)
print(results_df.to_string(index=False))
print("="*65)
print(f"SUCCESS: Model saved to 'leakage_free_insider_model.json'")
print(f"SUCCESS: CSV exported to '{eval_csv}'!")
print(f"SUCCESS: Matrix chart saved to '{cm_image_path}'!")
print("="*65)