
#----------------------------------- BRT ---------------------------------------#

import numpy as np
import pandas as pd
import random

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

# ===============================
# Set seed
# ===============================
np.random.seed(42)
random.seed(42)

# ===============================
# Sample weights
# ===============================
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# ===============================
# Model
# ===============================
gbm = GradientBoostingClassifier(random_state=42)

# ===============================
# Hyperparameter grid
# ===============================
param_grid = {
    "n_estimators": [100, 300, 500, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.6, 0.7, 1],
    "min_samples_leaf": [1, 3, 5]
}

# ===============================
# Stratified CV
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===============================
# Grid Search
# ===============================
grid = GridSearchCV(
    estimator=gbm,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid.fit(X_train, y_train, sample_weight=sample_weights)

print("Best Parameters:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# ===============================
# Extract results
# ===============================
cv_results = pd.DataFrame(grid.cv_results_)

all_results = []

for i, params in enumerate(cv_results["params"]):

    model = GradientBoostingClassifier(random_state=42, **params)

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Probabilities
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Predictions
    train_preds = (train_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)

    # Confusion matrices (safe)
    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(
        y_train, train_preds, labels=[0,1]
    ).ravel()

    tn_te, fp_te, fn_te, tp_te = confusion_matrix(
        y_test, test_preds, labels=[0,1]
    ).ravel()

    result = {
        **params,

        "CV_ROC_AUC": cv_results["mean_test_score"][i],

        # TRAIN
        "Train_Accuracy": accuracy_score(y_train, train_preds),
        "Train_ROC_AUC": roc_auc_score(y_train, train_probs),
        "Train_PR_AUC": average_precision_score(y_train, train_probs),
        "Train_Precision": precision_score(y_train, train_preds, zero_division=0),
        "Train_Recall": recall_score(y_train, train_preds, zero_division=0),
        "Train_F1": f1_score(y_train, train_preds, zero_division=0),
        "Train_TP": tp_tr,
        "Train_TN": tn_tr,
        "Train_FP": fp_tr,
        "Train_FN": fn_tr,

        # TEST
        "Test_Accuracy": accuracy_score(y_test, test_preds),
        "Test_ROC_AUC": roc_auc_score(y_test, test_probs),
        "Test_PR_AUC": average_precision_score(y_test, test_probs),
        "Test_Precision": precision_score(y_test, test_preds, zero_division=0),
        "Test_Recall": recall_score(y_test, test_preds, zero_division=0),
        "Test_F1": f1_score(y_test, test_preds, zero_division=0),
        "Test_TP": tp_te,
        "Test_TN": tn_te,
        "Test_FP": fp_te,
        "Test_FN": fn_te
    }

    all_results.append(result)

# ===============================
# Final Results
# ===============================
results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by="Test_ROC_AUC",
    ascending=False
)

# ===============================
# Save results
# ===============================
results_df.to_csv(
    r"C:\Users\HPC\Downloads\HPAI_Project\ML_Models\District level modeling\Final_Models\BRT\BRT_final_result.csv",
    index=False
)


# =====================================================================================================================================#
#***************************************** Retraining on best model with cv ************************************************************#



selected_params = [
   
    {"learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 1, "n_estimators": 300, "subsample": 0.6},
    
]

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix
)
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, params in enumerate(selected_params, 1):

    print(f"\n==============================")
    print(f"Model {i} | Params: {params}")
    print(f"==============================")

    model = GradientBoostingClassifier(
        random_state=42,
        **params
    )

    # MANUAL CV WITH SAMPLE WEIGHTS
    cv_probs = np.zeros(len(y_train))

    for train_idx, val_idx in cv.split(X_train, y_train):

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        w_tr = sample_weights[train_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr)

        cv_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    cv_auc = roc_auc_score(y_train, cv_probs)
    cv_pr  = average_precision_score(y_train, cv_probs)

    #  Fit on full training data
    model.fit(X_train, y_train, sample_weight=sample_weights)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs  = model.predict_proba(X_test)[:, 1]

    # Threshold = 0.5
    train_preds = (train_probs >= 0.5).astype(int)
    test_preds  = (test_probs  >= 0.5).astype(int)

    # Confusion matrices
    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, train_preds).ravel()
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, test_preds).ravel()

    #  CV METRICS
    print("\n CV METRICS (5-Fold on Train)")
    print("CV ROC-AUC :", cv_auc)
    print("CV PR-AUC  :", cv_pr)

    #  TRAIN METRICS
    print("\n TRAIN METRICS")
    print("Accuracy :", accuracy_score(y_train, train_preds))
    print("ROC-AUC  :", roc_auc_score(y_train, train_probs))
    print("PR-AUC   :", average_precision_score(y_train, train_probs))
    print("Precision:", precision_score(y_train, train_preds, zero_division=0))
    print("Recall   :", recall_score(y_train, train_preds))
    print("F1-score :", f1_score(y_train, train_preds))
    print("Confusion Matrix (TN FP FN TP):", tn_tr, fp_tr, fn_tr, tp_tr)

    #  TEST METRICS
    print("\n TEST METRICS")
    print("Accuracy :", accuracy_score(y_test, test_preds))
    print("ROC-AUC  :", roc_auc_score(y_test, test_probs))
    print("PR-AUC   :", average_precision_score(y_test, test_probs))
    print("Precision:", precision_score(y_test, test_preds, zero_division=0))
    print("Recall   :", recall_score(y_test, test_preds))
    print("F1-score :", f1_score(y_test, test_preds))
    print("Confusion Matrix (TN FP FN TP):", tn_te, fp_te, fn_te, tp_te)



    #=================================================================================================================#
    #**************************************** Feature importance plot ************************************************#


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=1,
    n_estimators=300,
    subsample=0.6,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)


from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=20,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

feat_imp = perm.importances_mean
plot_labels = X_train.columns

top_n = 42

sorted_idx = np.argsort(feat_imp)[::-1]   # descending
top_idx = sorted_idx[:top_n]

plt.figure(figsize=(12,8))

plt.barh(
    [plot_labels[i] for i in top_idx],
    feat_imp[top_idx],
    color="steelblue"
)

plt.xlabel("Drop in ROC-AUC when permuted")
plt.title("Permutation Feature Importance")

plt.gca().invert_yaxis()

# Save figure
plt.savefig(
    r"C:\Users\HPC\Downloads\BRT_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#===================== Top 20 features =====================3

top_n = 20

sorted_idx = np.argsort(feat_imp)[::-1]   # descending
top_idx = sorted_idx[:top_n]

plt.figure(figsize=(12,8))

plt.barh(
    [plot_labels[i] for i in top_idx],
    feat_imp[top_idx],
    color="steelblue"
)

plt.xlabel("Drop in ROC-AUC when permuted")
#plt.title("Permutation Feature Importance")

plt.gca().invert_yaxis()

# Save figure
plt.savefig(
    r"C:\Users\HPC\Downloads\BRT_feature_importance_top20.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



#=====================================================================================================================================#
#************************************************************** SHAP Analysis ********************************************************#

import shap
import matplotlib.pyplot as plt

# Create explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_train)

# Create a matplotlib figure
fig, ax = plt.subplots(figsize=(12,8))

# Summary plot on the figure
shap.summary_plot(
    shap_values,
    X_train,
    max_display=42,
    show=False,      # IMPORTANT: prevents SHAP from calling plt.show() immediately
    plot_type="dot"  # optional, default is "dot"
)

# Save the figure
plt.savefig(
    r"C:\Users\HPC\Downloads\BRT_shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

# Now display
plt.show()


# bar plot 

import shap
import matplotlib.pyplot as plt

# Create SHAP bar plot
shap.plots.bar(
    shap_values,
    max_display=42,
    show=False   # prevents SHAP from displaying immediately
)

# Save the figure
plt.savefig(
    r"C:\Users\HPC\Downloads\BRT_shap_bar_plot.png",
    dpi=300,
    bbox_inches="tight"
)

# Now display
plt.show()




