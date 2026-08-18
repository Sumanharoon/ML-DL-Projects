import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. DATA PREPROCESSING ---
df = pd.read_csv("new data label news.csv").dropna(subset=["Title", "Category"])
df["Category"] = df["Category"].astype(str).str.strip().replace(
    {"Business": "Business & Markets", "Markets": "Business & Markets", 
     "business": "Business & Markets", "markets": "Business & Markets"}
)

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Category"])

def clean_and_tokenize(text):
    text = re.sub(r"[^a-z0-9\s]", "", str(text).lower())
    return text.split()

df["tokens"] = df["Title"].apply(clean_and_tokenize)
vocab_counts = Counter([token for tokens in df["tokens"] for token in tokens])
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in vocab_counts.items():
    if count >= 2: vocab[word] = len(vocab)

VOCAB_SIZE, MAX_LEN = len(vocab), 20

def tokens_to_ids(tokens, vocab, max_len):
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return (ids + [vocab["<PAD>"]] * max_len)[:max_len]

df["ids"] = df["tokens"].apply(lambda x: tokens_to_ids(x, vocab, MAX_LEN))

X_train, X_test, y_train, y_test = train_test_split(
    df["ids"].tolist(), df["label"].values, test_size=0.15, random_state=42, stratify=df["label"]
)

class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(NewsDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(NewsDataset(X_test, y_test), batch_size=64, shuffle=False)

raw_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(np.sqrt(raw_weights), dtype=torch.float)

# --- 2. LSTM MODEL ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<PAD>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        return self.linear(self.dropout(pooled))

# --- 3. TRAINING & EVALUATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(VOCAB_SIZE, 128, 96, len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=3e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

epochs, patience, best_loss, counter = 25, 4, float("inf"), 0
save_path = "best_lstm_model.pth"

for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            val_loss += criterion(model(inputs.to(device)), labels.to(device)).item()
    
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), save_path)
        counter = 0
    else:
        counter += 1
        if counter >= patience: break

model.load_state_dict(torch.load(save_path))
model.eval()
y_preds, y_true = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.to(device))
        y_preds.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().numpy())
        y_true.extend(labels.numpy())

# --- 4. EXPORT ---
pd.DataFrame({
    "Actual_Category": label_encoder.inverse_transform(y_true),
    "Predicted_Category": label_encoder.inverse_transform(y_preds),
    "Is_Correct": np.array(y_true) == np.array(y_preds)
}).to_csv("lstm_predictions_output.csv", index=False)

pd.DataFrame(classification_report(y_true, y_preds, target_names=label_encoder.classes_, output_dict=True)).transpose().to_csv("lstm_classification_report.csv")