
#------------------------------- Random Forest ---------------------------------#

import pandas as pd 
import numpy as np

train_df = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\train_district_7030.csv")
test_df = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\test_district_7030.csv")

# Save identifiers for later
train_ids = train_df[['State', 'District']]
test_ids  = test_df[['State', 'District']]

# define features and target
X_train = train_df.drop(columns=['Status', 'State', 'District'])
y_train = train_df['Status']

X_test  = test_df.drop(columns=['Status', 'State', 'District'])
y_test  = test_df['Status']

# compute class weights
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

class_weights = dict(zip(classes, weights))
print("Class weights:", class_weights)


# Hyperparamteric Tuning
from sklearn.metrics import (
    precision_recall_curve, accuracy_score, roc_auc_score,
    recall_score, precision_score, f1_score
)

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


from itertools import product

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 3]
}

param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_split"],
    param_grid["min_samples_leaf"]
))

print("Total models:", len(param_combinations))

from sklearn.ensemble import RandomForestClassifier

results = []

for n_est, depth, min_split, min_leaf in param_combinations:

    rf_model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        random_state=42,
        class_weight=class_weights,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    # =====================
    # Probabilities
    # =====================
    train_prob = rf_model.predict_proba(X_train)[:, 1]
    test_prob  = rf_model.predict_proba(X_test)[:, 1]

    # =====================
    # Find best threshold from TRAIN only
    # =====================
    best_threshold = find_best_threshold(y_train, train_prob)

    # =====================
    # Metrics at best threshold
    # =====================
    train_metrics = get_metrics(y_train, train_prob, best_threshold)
    test_metrics  = get_metrics(y_test, test_prob, best_threshold)

    # =====================
    # Store results
    # =====================
    results.append({
        "n_estimators": n_est,
        "max_depth": depth,
        "min_samples_split": min_split,
        "min_samples_leaf": min_leaf,
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

results_df = pd.DataFrame(results)

results_df.to_csv(
    r"C:\Users\HPC\Downloads\HPAI_Project\ML_Models\District level modeling\rf_param_grid_best_threshold_results_7030.csv",
    index=False
)

print("All parameter results with best thresholds saved.")



#=======================================================================================================
#**************************************** Retraining Best Model with CV *********************************#

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, confusion_matrix
)

threshold = 0.5

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

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        class_weight=class_weights,
        n_jobs=-1
    )

    rf_model.fit(X_tr, y_tr)

    # ======================
    # TRAIN predictions
    # ======================
    y_tr_prob = rf_model.predict_proba(X_tr)[:, 1]
    y_tr_pred = (y_tr_prob >= threshold).astype(int)
    train_cm  = confusion_matrix(y_tr, y_tr_pred)

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
    y_val_prob = rf_model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= threshold).astype(int)
    val_cm     = confusion_matrix(y_val, y_val_pred)

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
# 3Mean CV Results
# ===============================
train_metrics_all = np.array(train_metrics_all)
val_metrics_all   = np.array(val_metrics_all)

print("\n ===== RF 5-Fold CV MEAN TRAIN =====")
print("Accuracy :", train_metrics_all[:,0].mean())
print("ROC AUC  :", train_metrics_all[:,1].mean())
print("Recall   :", train_metrics_all[:,2].mean())
print("Precision:", train_metrics_all[:,3].mean())
print("F1 Score :", train_metrics_all[:,4].mean())

print("\n ===== RF 5-Fold CV MEAN VALIDATION =====")
print("Accuracy :", val_metrics_all[:,0].mean())
print("ROC AUC  :", val_metrics_all[:,1].mean())
print("Recall   :", val_metrics_all[:,2].mean())
print("Precision:", val_metrics_all[:,3].mean())
print("F1 Score :", val_metrics_all[:,4].mean())

# ===============================
# FINAL MODEL → TRAIN on FULL TRAIN SET
# ===============================
final_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    class_weight=class_weights,
    n_jobs=-1
)

final_rf.fit(X_train, y_train)

# ===============================
# FINAL TRAIN PERFORMANCE
# ===============================
y_train_prob = final_rf.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_prob >= threshold).astype(int)
train_cm = confusion_matrix(y_train, y_train_pred)

print("\n ===== RF FINAL TRAIN RESULTS =====")
print("Accuracy :", accuracy_score(y_train, y_train_pred))
print("ROC AUC  :", roc_auc_score(y_train, y_train_prob))
print("Recall   :", recall_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("F1 Score :", f1_score(y_train, y_train_pred))
print("Train Confusion Matrix:\n", train_cm)

# ===============================
# FINAL TEST PERFORMANCE
# ===============================
y_test_prob = final_rf.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= threshold).astype(int)
test_cm = confusion_matrix(y_test, y_test_pred)

print("\n ===== RF FINAL TEST RESULTS =====")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_test_prob))
print("Recall   :", recall_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("F1 Score :", f1_score(y_test, y_test_pred))
print("Test Confusion Matrix:\n", test_cm)