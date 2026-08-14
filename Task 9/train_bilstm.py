from collections import Counter
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Set Reproducibility Seeds
torch.manual_seed(42)
np.random.seed(42)

# =============================================================
# 1. LOAD DATA & MERGE CATEGORIES
# =============================================================
file_name = "new data label news.csv"
df = pd.read_csv(file_name).dropna(subset=["Title", "Category"])

df["Category"] = df["Category"].astype(str).str.strip()
df["Category"] = df["Category"].replace(
    {
        "Business": "Business & Markets",
        "Markets": "Business & Markets",
        "business": "Business & Markets",
        "markets": "Business & Markets",
    }
)

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Category"])


# STEP 1: TOKENS
def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


df["tokens"] = df["Title"].apply(clean_and_tokenize)

# STEP 2: IDs (Vocabulary)
all_tokens = [token for tokens in df["tokens"] for token in tokens]
vocab_counts = Counter(all_tokens)

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in vocab_counts.items():
    if count >= 2:  # Low-frequency noise filter
        vocab[word] = len(vocab)

VOCAB_SIZE = len(vocab)
MAX_LEN = 20  # Optimized sequence length


# STEP 3: PADDING
def tokens_to_ids(tokens, vocab, max_len):
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


df["ids"] = df["tokens"].apply(lambda x: tokens_to_ids(x, vocab, MAX_LEN))

# Split Data (85% Train, 15% Test)
X_train, X_test, y_train, y_test = train_test_split(
    df["ids"].tolist(),
    df["label"].values,
    test_size=0.15,
    random_state=42,
    stratify=df["label"],
)


class NewsDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Class Weights (Balanced Focus)
raw_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights_tensor = torch.tensor(np.sqrt(raw_weights), dtype=torch.float)


# =============================================================
# 2. EXACT SLIDE PIPELINE ARCHITECTURE
# =============================================================
class BiLSTMClassifier(nn.Module):

    def __init__(
        self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.5
    ):
        super(BiLSTMClassifier, self).__init__()

        # STEP 4: EMBEDDING
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=vocab["<PAD>"]
        )

        # STEP 5: BiLSTM LAYER
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # STEP 6: DROPOUT LAYER
        self.dropout = nn.Dropout(dropout_rate)

        # STEP 7: LINEAR CLASSIFIER
        self.linear = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        lstm_out, (hn, cn) = self.bilstm(embedded)

        # Average pooling across time dimension
        pooled = torch.mean(lstm_out, dim=1)

        out = self.dropout(pooled)

        # STEP 8: CATEGORY OUTPUT
        logits = self.linear(out)
        return logits


# Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 96
NUM_CLASSES = len(label_encoder.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier(
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, dropout_rate=0.5
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=3e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=1
)

# =============================================================
# 3. TRAINING LOOP (RECORDING BEST ACCURACIES)
# =============================================================
epochs = 25
best_loss = float("inf")
patience, counter = 4, 0

# Variables to store best model's accuracy
best_train_acc = 0.0
best_test_acc = 0.0
best_epoch_num = 0

print(f"\nTraining Optimized BiLSTM Pipeline on {device}...")

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = (correct / total) * 100

    # Validation Phase
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = (val_correct / val_total) * 100
    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch+1:02d}/{epochs}] | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}"
    )

    # Save Best Metrics
    if val_loss < best_loss:
        best_loss = val_loss
        best_train_acc = train_acc
        best_test_acc = val_acc
        best_epoch_num = epoch + 1
        torch.save(model.state_dict(), "best_bilstm_model.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("\nEarly stopping triggered! Best model saved.")
            break

# =============================================================
# 4. EXPORT PROMINENT FINAL ACCURACY SUMMARY CSV
# =============================================================
final_accuracy_df = pd.DataFrame(
    [
        {
            "Metric": "Final Training Accuracy",
            "Value (%)": round(best_train_acc, 2),
        },
        {
            "Metric": "Final Testing Accuracy",
            "Value (%)": round(best_test_acc, 2),
        },
        {
            "Metric": "Overfitting Gap (Train - Test)",
            "Value (%)": round(best_train_acc - best_test_acc, 2),
        },
        {"Metric": "Best Saved Epoch", "Value (%)": best_epoch_num},
    ]
)

final_accuracy_df.to_csv("final_accuracy_summary.csv", index=False)

# =============================================================
# 5. EVALUATION & EXPORT CLASSIFICATION REPORTS
# =============================================================
model.load_state_dict(torch.load("best_bilstm_model.pth"))
model.eval()

y_preds, y_true, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)
        y_preds.extend(preds.cpu().numpy())
        y_true.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# Save Detailed Prediction Results
test_results_df = pd.DataFrame(
    {
        "News_Title": X_test,
        "Actual_Category": label_encoder.inverse_transform(y_true),
        "Predicted_Category": label_encoder.inverse_transform(y_preds),
        "Confidence_Score": np.max(all_probs, axis=1),
        "Is_Correct": np.array(y_true) == np.array(y_preds),
    }
)
test_results_df.to_csv("bilstm_predictions_output.csv", index=False)

# Classification Report CSV
report_dict = classification_report(
    y_true,
    y_preds,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv("bilstm_classification_report.csv")

# Prominent Screen Output
print("\n" + "=" * 50)
print("             FINAL ACCURACY SUMMARY             ")
print("=" * 50)
print(f"  Training Accuracy : {best_train_acc:.2f}%")
print(f"  Testing Accuracy  : {best_test_acc:.2f}%")
print(f"  Accuracy Gap      : {best_train_acc - best_test_acc:.2f}%")
print("=" * 50)
print("✓ 'final_accuracy_summary.csv' exported successfully!")