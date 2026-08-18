import pandas as pd
import numpy as np
import random
import os
from collections import deque

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# 0. REPRODUCIBILITY & DEVICE
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("REALISTIC REINFORCEMENT LEARNING NEWS RECOMMENDATION SYSTEM")
print("=" * 70)
print(f"Using device: {device}")

# Create output directory for exported files
output_dir = "task_11_outputs"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 1. LOAD DATASET
# =========================================================
file_path = "News Classification Labeling.xlsx"
df = pd.read_excel(file_path)

df = df.dropna(subset=["Title"]).copy()
df["Title"] = df["Title"].astype(str).str.strip()
df = df.reset_index(drop=True)

num_titles = len(df)
print(f"Total Clean Titles: {num_titles}")

# =========================================================
# 2. TF-IDF & CLUSTERING
# =========================================================
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
tfidf_matrix = vectorizer.fit_transform(df["Title"]).toarray()

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
df["cluster_id"] = kmeans.fit_predict(tfidf_matrix)

cluster_categories = {
    0: "Trade Policy & Global Markets",
    1: "Cryptocurrency & Aviation",
    2: "Corporate Financial Earnings",
    3: "Corporate Governance & Legal",
    4: "Geopolitics & International Relations"
}
df["assigned_category"] = df["cluster_id"].map(cluster_categories)

# Save Clustered Dataset CSV
df.to_csv(os.path.join(output_dir, "clustered_news_dataset.csv"), index=False)

# =========================================================
# 3. TRAIN / TEST SPLIT FOR TITLES
# =========================================================
train_indices, test_indices = train_test_split(
    np.arange(num_titles), test_size=0.20, random_state=SEED, stratify=df["cluster_id"]
)

print(f"Training Titles: {len(train_indices)} | Testing Titles: {len(test_indices)}")

# =========================================================
# 4. USER PREFERENCES
# =========================================================
num_users = 10
user_preferences = {user_id: np.random.randint(0, n_clusters) for user_id in range(num_users)}

# Export User Ground-Truth Preferences
user_pref_df = pd.DataFrame([
    {"User_ID": u, "Preferred_Cluster": c, "Preferred_Category": cluster_categories[c]}
    for u, c in user_preferences.items()
])
user_pref_df.to_csv(os.path.join(output_dir, "user_preferences.csv"), index=False)

def get_user_reward(user_id, title_id, noise=True):
    recommended_category = df.loc[title_id, "cluster_id"]
    target_category = user_preferences[user_id]
    
    if recommended_category == target_category:
        reward = 10.0
    else:
        reward = -5.0
        
    if noise and np.random.rand() < 0.10:
        reward = -reward

    return reward

# =========================================================
# 5. TABULAR MODELS (Q-LEARNING, SARSA, EXPECTED SARSA)
# =========================================================
alpha = 0.05
gamma = 0.90
epsilon = 0.20
training_steps = 1000

q_table = np.zeros((num_users, n_clusters), dtype=np.float32)
sarsa_table = np.zeros((num_users, n_clusters), dtype=np.float32)
expected_sarsa_table = np.zeros((num_users, n_clusters), dtype=np.float32)

# Train Q-Learning
for step in range(training_steps):
    for user_id in range(num_users):
        sampled_title = np.random.choice(train_indices)
        
        if np.random.rand() < epsilon:
            chosen_cat = np.random.randint(0, n_clusters)
        else:
            chosen_cat = np.argmax(q_table[user_id])
            
        reward = 10.0 if chosen_cat == user_preferences[user_id] else -5.0
        
        old_q = q_table[user_id, chosen_cat]
        max_next_q = np.max(q_table[user_id])
        q_table[user_id, chosen_cat] = old_q + alpha * (reward + gamma * max_next_q - old_q)

# Train SARSA
for step in range(training_steps):
    for user_id in range(num_users):
        chosen_cat = np.random.randint(0, n_clusters) if np.random.rand() < epsilon else np.argmax(sarsa_table[user_id])
        reward = 10.0 if chosen_cat == user_preferences[user_id] else -5.0
        next_cat = np.random.randint(0, n_clusters) if np.random.rand() < epsilon else np.argmax(sarsa_table[user_id])
        
        old_q = sarsa_table[user_id, chosen_cat]
        next_q = sarsa_table[user_id, next_cat]
        sarsa_table[user_id, chosen_cat] = old_q + alpha * (reward + gamma * next_q - old_q)

# Train Expected SARSA
for step in range(training_steps):
    for user_id in range(num_users):
        chosen_cat = np.random.randint(0, n_clusters) if np.random.rand() < epsilon else np.argmax(expected_sarsa_table[user_id])
        reward = 10.0 if chosen_cat == user_preferences[user_id] else -5.0
        
        q_vals = expected_sarsa_table[user_id]
        best_act = np.argmax(q_vals)
        exp_q = (epsilon / n_clusters) * np.sum(q_vals) + (1 - epsilon) * q_vals[best_act]
        
        old_q = expected_sarsa_table[user_id, chosen_cat]
        expected_sarsa_table[user_id, chosen_cat] = old_q + alpha * (reward + gamma * exp_q - old_q)

# =========================================================
# 6. DEEP Q-NETWORK (DQN)
# =========================================================
class ContextualDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

state_dim = num_users
action_dim = n_clusters

dqn = ContextualDQN(state_dim, action_dim).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.MSELoss()

memory = deque(maxlen=2000)
batch_size = 32

dqn.train()
for epoch in range(500):
    for user_id in range(num_users):
        state = np.zeros(num_users, dtype=np.float32)
        state[user_id] = 1.0
        
        if np.random.rand() < epsilon:
            action = np.random.randint(0, n_clusters)
        else:
            with torch.no_grad():
                st_tensor = torch.tensor(state, device=device).unsqueeze(0)
                action = torch.argmax(dqn(st_tensor), dim=1).item()
                
        reward = 10.0 if action == user_preferences[user_id] else -5.0
        memory.append((state, action, reward, state))
        
        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            b_s = torch.tensor(np.array([m[0] for m in minibatch]), device=device)
            b_a = torch.tensor([m[1] for m in minibatch], device=device).unsqueeze(1)
            b_r = torch.tensor([m[2] for m in minibatch], dtype=torch.float32, device=device)
            
            q_eval = dqn(b_s).gather(1, b_a).squeeze(1)
            with torch.no_grad():
                q_next = dqn(b_s).max(1)[0]
                q_target = b_r + 0.90 * q_next
                
            loss = loss_fn(q_eval, b_r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Extract Q-Matrix from Trained DQN Model
dqn.eval()
dqn_q_table = np.zeros((num_users, n_clusters), dtype=np.float32)
with torch.no_grad():
    for u in range(num_users):
        st = np.zeros(num_users, dtype=np.float32)
        st[u] = 1.0
        st_tensor = torch.tensor(st, device=device).unsqueeze(0)
        dqn_q_table[u] = dqn(st_tensor).cpu().numpy()[0]

# =========================================================
# 7. SAVE Q-MATRICES & POLICY TABLES TO CSV
# =========================================================
cols = [f"Category_{i}" for i in range(n_clusters)]

pd.DataFrame(q_table, columns=cols).to_csv(os.path.join(output_dir, "q_table_q_learning.csv"), index_label="User_ID")
pd.DataFrame(sarsa_table, columns=cols).to_csv(os.path.join(output_dir, "q_table_sarsa.csv"), index_label="User_ID")
pd.DataFrame(expected_sarsa_table, columns=cols).to_csv(os.path.join(output_dir, "q_table_expected_sarsa.csv"), index_label="User_ID")
pd.DataFrame(dqn_q_table, columns=cols).to_csv(os.path.join(output_dir, "q_table_dqn.csv"), index_label="User_ID")

# Policy Summary (Recommended Category per User for each model)
policy_summary = pd.DataFrame({
    "User_ID": list(range(num_users)),
    "Ground_Truth_Pref": [user_preferences[u] for u in range(num_users)],
    "Q_Learning_Action": np.argmax(q_table, axis=1),
    "SARSA_Action": np.argmax(sarsa_table, axis=1),
    "Expected_SARSA_Action": np.argmax(expected_sarsa_table, axis=1),
    "DQN_Action": np.argmax(dqn_q_table, axis=1),
})
policy_summary.to_csv(os.path.join(output_dir, "user_policy_recommendations.csv"), index=False)

# =========================================================
# 8. UNSEEN TEST DATA EVALUATION
# =========================================================
print("\n" + "=" * 70)
print("EVALUATING MODELS ON UNSEEN TEST TITLES (20% HELD OUT)")
print("=" * 70)

def evaluate_on_test_set(policy_type):
    total_correct = 0
    total_evals = 0
    rewards_list = []

    for user_id in range(num_users):
        target_cat = user_preferences[user_id]
        
        for title_id in test_indices:
            actual_cat = df.loc[title_id, "cluster_id"]
            
            if policy_type == "q_learning":
                pred_cat = np.argmax(q_table[user_id])
            elif policy_type == "sarsa":
                pred_cat = np.argmax(sarsa_table[user_id])
            elif policy_type == "expected_sarsa":
                pred_cat = np.argmax(expected_sarsa_table[user_id])
            elif policy_type == "dqn":
                pred_cat = np.argmax(dqn_q_table[user_id])
            
            reward = get_user_reward(user_id, title_id, noise=False) if pred_cat == actual_cat else -5.0
            
            if pred_cat == target_cat:
                total_correct += 1
            
            rewards_list.append(reward)
            total_evals += 1

    accuracy = (total_correct / total_evals) * 100
    avg_reward = np.mean(rewards_list)
    return round(accuracy, 2), round(avg_reward, 2)

# Run Evaluation
q_acc, q_rew = evaluate_on_test_set("q_learning")
sarsa_acc, sarsa_rew = evaluate_on_test_set("sarsa")
exp_acc, exp_rew = evaluate_on_test_set("expected_sarsa")
dqn_acc, dqn_rew = evaluate_on_test_set("dqn")

# Results Dataframe
results_df = pd.DataFrame([
    {"Model": "Q-Learning", "Test Accuracy (%)": q_acc, "Avg Test Reward": q_rew},
    {"Model": "SARSA", "Test Accuracy (%)": sarsa_acc, "Avg Test Reward": sarsa_rew},
    {"Model": "Expected SARSA", "Test Accuracy (%)": exp_acc, "Avg Test Reward": exp_rew},
    {"Model": "DQN", "Test Accuracy (%)": dqn_acc, "Avg Test Reward": dqn_rew}
])

print(results_df.to_string(index=False))

# Save Benchmark Results CSV
results_df.to_csv(os.path.join(output_dir, "unseen_test_rl_results.csv"), index=False)

print("\n" + "=" * 70)
print(f"[SUCCESS] All CSV files and matrices saved in folder: '{output_dir}/'")
print("=" * 70)