
#---------------------------------------------------------------------------- GAT -------------------------------------------------------------------#

# =========================================================
# Imports
# =========================================================
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import random

# =========================================================
# Set Seed
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================
# Load Data
# =========================================================
train_df = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\train_district_7030.csv")
test_df  = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\test_district_7030.csv")

y_train = train_df['Status']
y_test  = test_df['Status']

# Drop unwanted columns
exclude_cols = ['Latitude', 'Longitude', 'chicken_den_2010', 'duck_den_2010']
X_train = train_df.drop(columns=['Status', 'State', 'District'] + exclude_cols)
X_test  = test_df.drop(columns=['Status', 'State', 'District'] + exclude_cols)

# Combine for scaling
X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_all = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# =========================================================
# Load Shapefile & Build Graph
# =========================================================
shp = gpd.read_file(r"C:\Users\HPC\Downloads\HPAI_Project\district_shape_file_new\india_districts_2011.shp")
shp = shp.rename(columns={'ST_NM': 'State', 'DISTRICT': 'District'})

def clean_name(name):
    if pd.isnull(name):
        return ""
    return name.upper().replace(" ", "").strip()

all_ids = pd.concat([train_df[['State','District']], test_df[['State','District']]]).drop_duplicates().reset_index(drop=True)
all_ids['State_clean'] = all_ids['State'].apply(clean_name)
all_ids['District_clean'] = all_ids['District'].apply(clean_name)
shp['State_clean'] = shp['State'].apply(clean_name)
shp['District_clean'] = shp['District'].apply(clean_name)

all_ids = all_ids.merge(shp[['State_clean','District_clean','geometry']], on=['State_clean','District_clean'], how='left')
all_ids = gpd.GeoDataFrame(all_ids, geometry='geometry')

# Build adjacency edges
edges = []
for i, geom_i in enumerate(all_ids.geometry):
    for j, geom_j in enumerate(all_ids.geometry):
        if i < j and geom_i is not None and geom_j is not None:
            if geom_i.touches(geom_j):
                edges.append((i, j))
edge_index = torch.tensor(edges + [(j, i) for i, j in edges], dtype=torch.long).t().to(device)

# =========================================================
# Prepare PyTorch Geometric Data
# =========================================================
data = Data(
    x=torch.tensor(X_scaled, dtype=torch.float).to(device),
    edge_index=edge_index,
    y=torch.tensor(y_all.values, dtype=torch.float).to(device)
)

# =========================================================
# Class imbalance weight
# =========================================================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
pos_weight = torch.tensor([neg/pos], dtype=torch.float).to(device)

# =========================================================
# Define GAT Model
# =========================================================
class GATNet(nn.Module):
    def __init__(self, in_feats, hidden_feats=16, out_feats=1, heads=4, dropout=0.01):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_feats*heads, out_feats, heads=1, concat=False, dropout=dropout)
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

# =========================================================
# Initialize Model
# =========================================================
model = GATNet(in_feats=X_scaled.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# =========================================================
# Train Model
# =========================================================
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index).squeeze()
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

# =========================================================
# Evaluate
# =========================================================
threshold = 0.68
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index).squeeze()
    probs = torch.sigmoid(logits).cpu().numpy()

train_probs = probs[:len(X_train)]
test_probs  = probs[len(X_train):]
train_preds = (train_probs >= threshold).astype(int)
test_preds  = (test_probs >= threshold).astype(int)

def print_metrics(name, y_true, y_prob, y_pred):
    print(f"\n===== {name} METRICS (Threshold = {threshold}) =====")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_true, y_prob))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

print_metrics("TRAIN", y_train.values, train_probs, train_preds)
print_metrics("TEST", y_test.values, test_probs, test_preds)

# =========================================================
# Permutation Feature Importance
# =========================================================
def permutation_importance(model, X, y, edge_index, device, metric=roc_auc_score):
    model.eval()
    with torch.no_grad():
        baseline_logits = model(X.to(device), edge_index).squeeze()
        baseline_probs = torch.sigmoid(baseline_logits).detach().cpu().numpy()
    baseline_score = metric(y.detach().cpu().numpy(), baseline_probs)

    importances = []
    for i in range(X.shape[1]):
        X_permuted = X.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(X_permuted.size(0)), i]
        with torch.no_grad():
            logits = model(X_permuted.to(device), edge_index).squeeze()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        score = metric(y.detach().cpu().numpy(), probs)
        importances.append(baseline_score - score)
    return np.array(importances)

# Convert y_all to tensor
y_all_tensor = torch.tensor(y_all.values, dtype=torch.float).to(device)

# Compute feature importance
feat_imp = permutation_importance(model, data.x, y=y_all_tensor, edge_index=data.edge_index, device=device)

# Plot
plt.figure(figsize=(10,6))
plt.barh(X_all.columns, feat_imp)
plt.xlabel("Drop in ROC-AUC when permuted")
plt.title("Permutation Feature Importance - GAT (Filtered Features)")
plt.gca().invert_yaxis()
plt.show()


#===================== FEATURE IMPORTANCE PLOT ========================#
# Print all feature names
print("Feature names in the dataset:")
for i, col in enumerate(X_all.columns):
    print(f"{i+1}. {col}")

custom_labels = {
    "Migratory_Path": "Migratory Path",
    "Annual_Mean_Temperature": "Annual Mean Temperature",
    "Mean_Diurnal_Range": "Mean Diurnal Range",
    "Isothermality": "Isothermality",
    "Temperature_Seasonality": "Temperature Seasonality",
    "Max_Temperature_of_Warmest_Month": "Max Temperature Warmest Month",
    "Min_Temperature_of_Coldest_Month": "Min Temperature Coldest Month",
    "Temperature_Annual_Range": "Temperature Annual Range",
    "Mean_Temperature_of_Wettest_Quarter": "Mean Temperature Wettest Quarter",
    "Mean_Temperature_of_Driest_Quarter": "Mean Temperature Driest Quarter",
    "Mean_Temperature_of_Warmest_Quarter": "Mean Temperature Warmest Quarter",
    "Mean_Temperature_of_Coldest_Quarter": "Mean Temperature Coldest Quarter",
    "Precipitation_Seasonality": "Precipitation Seasonality",
    "EVI": "EVI",
    "LST": "LST",
    "NDVI": "NDVI",
    "Potential_Evaporation": "Potential Evaporation",
    "Air_Temperature": "Air Temperature",
    "Wind_Speed": "Wind Speed",
    "Surface_Pressure": "Surface Pressure",
    "Specific_Humidity": "Specific Humidity",
    "Soil_Moisture": "Soil Moisture",
    "chicken_den_2015": "Chicken Density 2015",
    "duck_den_2015": "Duck Density 2015",
    "Precipitation_of_Coldest_Quarter": "Precipitation Coldest Quarter",
    "Precipitation_of_Driest_Month": "Precipitation Driest Month",
    "Precipitation_of_Warmest_Quarter": "Precipitation Warmest Quarter",
    "Precipitation_of_Wettest_Month": "Precipitation Wettest Month",
    "Precipitation_of_Driest_Quarter": "Precipitation Driest Quarter",
    "Precipitation_of_Wettest_Quarter": "Precipitation Wettest Quarter",
    "Annual_Precipitation": "Annual Precipitation",
    "LAI": "Leaf Area Index (LAI)",
    "Potential_Evapotranspiration": "Potential Evapotranspiration",
    "Water_canal": "Water Canal",
    "Water_drain": "Water Drain",
    "Water_river": "Water River",
    "Water_stream": "Water Stream",
    "dist_wetland_km": "Distance to Wetland (km)",
    "wetland_1km": "Wetland 1km Buffer"
}

# Create plot labels in order of X_all columns
plot_labels = [custom_labels.get(col, col) for col in X_all.columns]

plt.figure(figsize=(12,10))
plt.barh(plot_labels, feat_imp, color="#F7879A")
plt.xlabel("Drop in ROC-AUC when permuted")
plt.title("Permutation Feature Importance - GAT")
plt.gca().invert_yaxis()
plt.show()

#================= Top 20 Fatures =====================#

top_n = 20
sorted_idx = np.argsort(feat_imp)[::-1]  # descending order
top_idx = sorted_idx[:top_n]

plt.figure(figsize=(12,8))
plt.barh([plot_labels[i] for i in top_idx], feat_imp[top_idx], color="steelblue")
plt.xlabel("Drop in ROC-AUC when permuted")
#plt.title("Permutation Feature Importance (Top 20)")
plt.gca().invert_yaxis()

# Save figure
#plt.savefig(r"C:\Users\HPC\Downloads\feature_importance_top20.png", dpi=300, bbox_inches="tight")

plt.show()


top_n = 42
sorted_idx = np.argsort(feat_imp)[::-1]  # descending order
top_idx = sorted_idx[:top_n]

plt.figure(figsize=(12,8))
plt.barh([plot_labels[i] for i in top_idx], feat_imp[top_idx], color="steelblue")
plt.xlabel("Drop in ROC-AUC when permuted")
#plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()

# Save figure
#plt.savefig(r"C:\Users\HPC\Downloads\feature_importance.png", dpi=300, bbox_inches="tight")

plt.show()



#====================================== SHAP Analysis =======================================#

import shap
import torch
import numpy as np

# -------------------------------
# Wrapper for GAT model
# -------------------------------
class GATWrapper(torch.nn.Module):
    def __init__(self, model, edge_index):
        super().__init__()
        self.model = model
        self.edge_index = edge_index

    def forward(self, x):
        logits = self.model(x, self.edge_index).squeeze()
        return torch.sigmoid(logits)

wrapped_model = GATWrapper(model, data.edge_index).to(device)
wrapped_model.eval()


# -------------------------------
# Prediction function for SHAP
# -------------------------------
def predict_fn(X_numpy):

    X_tensor = torch.tensor(X_numpy, dtype=torch.float).to(device)
    outputs = []

    for row in X_tensor:

        # copy original node features
        full_x = data.x.clone()

        # replace the node features we want to explain
        full_x[0] = row

        with torch.no_grad():
            logits = model(full_x, data.edge_index).squeeze()
            prob = torch.sigmoid(logits)[0].cpu().numpy()

        outputs.append(prob)

    return np.array(outputs)


# -------------------------------
# Background and sample data
# -------------------------------
background = X_scaled[np.random.choice(X_scaled.shape[0], 50, replace=False)]
samples = X_scaled[:50]


# -------------------------------
# SHAP Explainer
# -------------------------------
explainer = shap.KernelExplainer(predict_fn, background)

shap_values = explainer.shap_values(
    samples,
    nsamples=100
)


# -------------------------------
# SHAP Summary Plot
# -------------------------------
shap.summary_plot(
    shap_values,
    samples,
    feature_names=X_all.columns
)

#===================== Bar Plot ====================#

import shap
import numpy as np

# 100 background samples
background = X_scaled[np.random.choice(X_scaled.shape[0], 100, replace=False)]

# Samples we want to explain
samples = X_scaled[:100]

explainer = shap.KernelExplainer(predict_fn, background)

shap_values = explainer.shap_values(
    samples,
    nsamples=100
)

# SHAP SUMMARY PLOT

shap.summary_plot(
    shap_values,
    samples,
    feature_names=X_all.columns
)

# SHAP FEATURE IMPORTNACE BAR PLOT

shap.summary_plot(
    shap_values,
    samples,
    feature_names=X_all.columns,
    plot_type="bar"
)
