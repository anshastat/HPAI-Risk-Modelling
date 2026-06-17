
#--------------------------------------------- GCN ----------------------------------------------------------#


# =========================================
# SEED (REPRODUCIBILITY)
# =========================================
import os, random, numpy as np, torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================
# LOAD DATA
# =========================================
import pandas as pd

train_df = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\train_district_7030.csv")
test_df  = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\test_district_7030.csv")

df_all = pd.concat([train_df, test_df]).reset_index(drop=True)

train_idx = np.arange(len(train_df))
test_idx  = np.arange(len(train_df), len(df_all))


# =========================================
# FEATURES + TARGET
# =========================================
from sklearn.preprocessing import StandardScaler

X = df_all.drop(columns=[
    'Status', 'State', 'District',
    'Longitude', 'Latitude',
    'chicken_den_2010', 'duck_den_2010'
])

y = df_all['Status']

scaler = StandardScaler()

X_train = scaler.fit_transform(X.iloc[train_idx])
X_test  = scaler.transform(X.iloc[test_idx])

X_scaled = np.vstack([X_train, X_test])

x_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y.values, dtype=torch.float)


# =========================================
# CREATE MASKS
# =========================================
train_mask = torch.zeros(len(df_all), dtype=torch.bool)
test_mask  = torch.zeros(len(df_all), dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx]   = True


# =========================================
# LOAD SHAPEFILE + CLEAN
# =========================================
import geopandas as gpd

shp = gpd.read_file(
    r"C:\Users\HPC\Downloads\HPAI_Project\district_shape_file_new\india_districts_2011.shp"
)

shp = shp.rename(columns={'ST_NM': 'State', 'DISTRICT': 'District'})

def clean_text(x):
    if pd.isnull(x):
        return ""
    return x.upper().strip()

shp['State'] = shp['State'].apply(clean_text)
shp['District'] = shp['District'].apply(clean_text)


# =========================================
# GRAPH ALIGNMENT 
# =========================================
df_ids = df_all[['State', 'District']].copy()
df_ids['State'] = df_ids['State'].apply(clean_text)
df_ids['District'] = df_ids['District'].apply(clean_text)

df_ids['node_id'] = np.arange(len(df_ids))

gdf = shp.merge(df_ids, on=['State', 'District'], how='right')

# remove missing geometry
gdf = gdf.dropna(subset=['geometry'])

# restore correct order
gdf = gdf.sort_values('node_id').reset_index(drop=True)

print("Nodes in graph :", len(gdf))
print("Expected nodes :", len(df_all))

# CHECK
if len(gdf) != len(df_all):
    raise ValueError("Graph nodes != feature rows. Fix district matching.")


# =========================================
# BUILD GRAPH
# =========================================
from libpysal.weights import Queen

w = Queen.from_dataframe(gdf)

edges = []
for i, neighbors in w.neighbors.items():
    for j in neighbors:
        edges.append([i, j])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# self-loops
self_loops = torch.arange(len(gdf))
self_loops = torch.stack([self_loops, self_loops])

edge_index = torch.cat([edge_index, self_loops], dim=1)


# =========================================
# PYG DATA OBJECT
# =========================================
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = Data(
    x=x_tensor,
    edge_index=edge_index,
    y=y_tensor
)

data.train_mask = train_mask
data.test_mask  = test_mask

data = data.to(device)


# =========================================
# MODEL
# =========================================
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatialGCN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.out   = GCNConv(32, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.out(x, edge_index)
        return x.view(-1)

model = SpatialGCN(in_channels=data.x.shape[1]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=3e-4)

# imbalance handling
y_train = y.iloc[train_idx]
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# =========================================
# TRAINING
# =========================================
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)

    loss = criterion(
        logits[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# =========================================
# PREDICTIONS
# =========================================
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits)

train_probs = probs[data.train_mask].cpu().numpy()
test_probs  = probs[data.test_mask].cpu().numpy()

y_train = y.iloc[train_idx].values
y_test  = y.iloc[test_idx].values


# =========================================
# THRESHOLD 
# =========================================
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_train, train_probs)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print(" Best threshold:", best_threshold)

# MUST recompute predictions HERE
train_preds = (train_probs > best_threshold).astype(int)
test_preds  = (test_probs  > best_threshold).astype(int)


# =========================================
# SAFE ROC FUNCTION
# =========================================
def safe_roc_auc(y_true, probs):
    if len(np.unique(y_true)) == 1:
        return np.nan
    return roc_auc_score(y_true, probs)


# =========================================
# EVALUATION FUNCTION
# =========================================
def evaluate(y_true, probs, preds, name="DATA"):
    acc = accuracy_score(y_true, preds)
    roc = safe_roc_auc(y_true, probs)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    cm = confusion_matrix(y_true, preds)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = np.nan

    print(f"\n=== {name} METRICS ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"ROC-AUC     : {roc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"Specificity : {specificity:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, preds, zero_division=0))


# =========================================
# RUN EVALUATION
# =========================================
evaluate(y_train, train_probs, train_preds, "TRAIN")
evaluate(y_test, test_probs, test_preds, "TEST")