
#--------------------------------- XGBoost -----------------------------------#

import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product
from sklearn.metrics import (
    precision_recall_curve, accuracy_score, roc_auc_score,
    recall_score, precision_score, f1_score
)

# ================================
# Load data
# ================================
train_df = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\train_district_7030.csv")
test_df  = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\test_district_7030.csv")

train_ids = train_df[['State', 'District']]
test_ids  = test_df[['State', 'District']]

X_train = train_df.drop(columns=['Status', 'State', 'District'])
y_train = train_df['Status']

X_test  = test_df.drop(columns=['Status', 'State', 'District'])
y_test  = test_df['Status']

# ================================
# Handle class imbalance
# ================================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)

# ================================
# Threshold & metric functions
# ================================
def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def get_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return (
        accuracy_score(y_true, y_pred),
        roc_auc_score(y_true, y_prob),
        recall_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )

# ================================
# XGBoost parameter grid
# ================================
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.5, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0]
}

param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["learning_rate"],
    param_grid["subsample"],
    param_grid["colsample_bytree"]
))

print("Total XGBoost models:", len(param_combinations))

# ================================
# Grid search loop
# ================================
results = []

for n_est, depth, lr, subsample, colsample in param_combinations:

    model = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        subsample=subsample,
        colsample_bytree=colsample,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        tree_method="hist"   # faster on CPU
    )

    model.fit(X_train, y_train)

    # =====================
    # Probabilities
    # =====================
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob  = model.predict_proba(X_test)[:, 1]

    # =====================
    # Best threshold from TRAIN
    # =====================
    best_threshold = find_best_threshold(y_train, train_prob)

    # =====================
    # Metrics
    # =====================
    train_metrics = get_metrics(y_train, train_prob, best_threshold)
    test_metrics  = get_metrics(y_test, test_prob, best_threshold)

    results.append({
        "n_estimators": n_est,
        "max_depth": depth,
        "learning_rate": lr,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "Best_Threshold": best_threshold,

        # Train metrics
        "Train_Accuracy": train_metrics[0],
        "Train_AUC": train_metrics[1],
        "Train_Recall": train_metrics[2],
        "Train_Precision": train_metrics[3],
        "Train_F1": train_metrics[4],

        # Test metrics
        "Test_Accuracy": test_metrics[0],
        "Test_AUC": test_metrics[1],
        "Test_Recall": test_metrics[2],
        "Test_Precision": test_metrics[3],
        "Test_F1": test_metrics[4],
    })

# ================================
# Save results
# ================================
results_df = pd.DataFrame(results)

results_df.to_csv(
    r"C:\Users\HPC\Downloads\HPAI_Project\ML_Models\District level modeling\xgboost_param_grid_best_threshold_results_7030.csv",
    index=False
)

print("All XGBoost parameter results saved.")


#=====================================================================================================
# **********************************Retraining best model with CV*************************************


# ===============================
# Set seeds
# ===============================
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# ===============================
# Imports
# ===============================
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score
)

# ===============================
# Handle class imbalance
# ===============================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)

# ===============================
# Model parameters
# ===============================
xgb_params = dict(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.01,
    subsample=0.5,
    colsample_bytree=0.5,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

threshold = 0.5

from sklearn.metrics import confusion_matrix

# ===============================
# Stratified 5-Fold CV
# ===============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_metrics_all = []
val_metrics_all   = []

fold = 1

for train_idx, val_idx in skf.split(X_train, y_train):

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr)

    # ======================
    # TRAIN predictions
    # ======================
    y_tr_prob = model.predict_proba(X_tr)[:, 1]
    y_tr_pred = (y_tr_prob >= threshold).astype(int)

    train_cm = confusion_matrix(y_tr, y_tr_pred)

    train_metrics = [
        accuracy_score(y_tr, y_tr_pred),
        roc_auc_score(y_tr, y_tr_prob),
        recall_score(y_tr, y_tr_pred),
        precision_score(y_tr, y_tr_pred),
        f1_score(y_tr, y_tr_pred)
    ]

    train_metrics_all.append(train_metrics)

    # ======================
    # VALIDATION predictions
    # ======================
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= threshold).astype(int)

    val_cm = confusion_matrix(y_val, y_val_pred)

    val_metrics = [
        accuracy_score(y_val, y_val_pred),
        roc_auc_score(y_val, y_val_prob),
        recall_score(y_val, y_val_pred),
        precision_score(y_val, y_val_pred),
        f1_score(y_val, y_val_pred)
    ]

    val_metrics_all.append(val_metrics)

    # ======================
    # PRINT FOLD RESULTS
    # ======================
    print(f"\n================ Fold {fold} ================")

    print("TRAIN → Acc:", train_metrics[0], "AUC:", train_metrics[1],
          "Recall:", train_metrics[2], "F1:", train_metrics[4])
    print("TRAIN Confusion Matrix:\n", train_cm)

    print("\nVAL   → Acc:", val_metrics[0], "AUC:", val_metrics[1],
          "Recall:", val_metrics[2], "F1:", val_metrics[4])
    print("VAL Confusion Matrix:\n", val_cm)

    fold += 1

# ===============================
# Mean CV Results
# ===============================
train_metrics_all = np.array(train_metrics_all)
val_metrics_all   = np.array(val_metrics_all)

print("\n ===== 5-Fold CV MEAN TRAIN =====")
print("Accuracy :", train_metrics_all[:,0].mean())
print("ROC AUC  :", train_metrics_all[:,1].mean())
print("Recall   :", train_metrics_all[:,2].mean())
print("Precision:", train_metrics_all[:,3].mean())
print("F1 Score :", train_metrics_all[:,4].mean())

print("\n ===== 5-Fold CV MEAN VALIDATION =====")
print("Accuracy :", val_metrics_all[:,0].mean())
print("ROC AUC  :", val_metrics_all[:,1].mean())
print("Recall   :", val_metrics_all[:,2].mean())
print("Precision:", val_metrics_all[:,3].mean())
print("F1 Score :", val_metrics_all[:,4].mean())

# ===============================
# FINAL MODEL → TRAIN on FULL TRAIN SET
# ===============================
final_model = xgb.XGBClassifier(**xgb_params)
final_model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix

# ===============================
# FINAL MODEL → TRAIN on FULL TRAIN SET
# ===============================
final_model = xgb.XGBClassifier(**xgb_params)
final_model.fit(X_train, y_train)

# ===============================
# TRAIN PERFORMANCE (FINAL MODEL)
# ===============================
y_train_prob = final_model.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_prob >= threshold).astype(int)

train_cm = confusion_matrix(y_train, y_train_pred)

print("\n ===== FINAL TRAIN RESULTS =====")
print("Accuracy :", accuracy_score(y_train, y_train_pred))
print("ROC AUC  :", roc_auc_score(y_train, y_train_prob))
print("Recall   :", recall_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("F1 Score :", f1_score(y_train, y_train_pred))
print("Train Confusion Matrix:\n", train_cm)

# ===============================
# TEST PERFORMANCE (FINAL MODEL)
# ===============================
y_test_prob = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= threshold).astype(int)

test_cm = confusion_matrix(y_test, y_test_pred)

print("\n ===== FINAL TEST RESULTS =====")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_test_prob))
print("Recall   :", recall_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("F1 Score :", f1_score(y_test, y_test_pred))
print("Test Confusion Matrix:\n", test_cm)