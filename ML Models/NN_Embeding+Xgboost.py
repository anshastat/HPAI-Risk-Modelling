
#----------------------------------- NN Embeding + XGBoost ----------------------------------#

import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.random.set_seed(SEED)

# =========================================================
#  Load train and test data
# =========================================================
train_df = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\train_district_7030.csv")
test_df  = pd.read_csv(r"C:\Users\HPC\Downloads\HPAI_Project\Psudo_data_prep\test_district_7030.csv")

train_ids = train_df[['State', 'District']]
test_ids  = test_df[['State', 'District']]

# =========================================================
# SET SEED
# =========================================================
import os
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import random
random.seed(SEED)

import numpy as np
import tensorflow as tf

np.random.seed(SEED)
tf.random.set_seed(SEED)


import numpy as np
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# LOAD DATA
# =========================================================
X_train = train_df.drop(columns=['Status', 'State', 'District']).values
y_train = train_df['Status'].values

X_test  = test_df.drop(columns=['Status', 'State', 'District']).values
y_test  = test_df['Status'].values

# =========================================================
# CLASS WEIGHTS
# =========================================================
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights_dict = dict(zip(classes, class_weights))

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

# =========================================================
# NN MODEL
# =========================================================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_nn(input_dim):
    inputs = Input(shape=(input_dim,))

    x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(inputs)
    x = Dropout(0.5)(x)

    x = Dense(16, activation="relu", kernel_regularizer=l2(0.01))(x)

    embedding = Dense(8, activation="relu", name="embedding")(x)

    outputs = Dense(1, activation="sigmoid")(embedding)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )

    return model

early_stop = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=0
)

# =========================================================
# XGBOOST MODEL
# =========================================================
def build_xgb():
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=SEED,
        tree_method="hist",
        n_jobs=-1
    )

# =========================================================
#  METRIC FUNCTION
# =========================================================
def get_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

# =========================================================
# CV PIPELINE
# =========================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    print(f"\n Fold {fold}")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    #  Scale inside fold
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    #  Train NN on fold-train
    nn_model = build_nn(X_tr_scaled.shape[1])

    history = nn_model.fit(
        X_tr_scaled, y_tr,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=0
    )

    stopped_epoch = len(history.history["loss"])
    print("Stopped at epoch:", stopped_epoch)

    #  Extract embeddings
    embedding_model = Model(
        inputs=nn_model.input,
        outputs=nn_model.get_layer("embedding").output
    )

    X_tr_emb = embedding_model.predict(X_tr_scaled, verbose=0)
    X_val_emb = embedding_model.predict(X_val_scaled, verbose=0)

    #  Train XGBoost
    xgb_model = build_xgb()
    xgb_model.fit(X_tr_emb, y_tr)

    #  Train metrics
    train_prob = xgb_model.predict_proba(X_tr_emb)[:, 1]
    train_m = get_metrics(y_tr, train_prob)

    #  Validation metrics
    val_prob = xgb_model.predict_proba(X_val_emb)[:, 1]
    val_m = get_metrics(y_val, val_prob)

    print("Train AUC:", train_m["roc_auc"], "Val AUC:", val_m["roc_auc"])

    row = {"fold": fold, **{f"train_{k}": v for k, v in train_m.items()},
                          **{f"val_{k}": v for k, v in val_m.items()},
           "stopped_epoch": stopped_epoch}

    cv_results.append(row)

cv_results_df = pd.DataFrame(cv_results)

print("\n Mean CV AUC:", cv_results_df["val_roc_auc"].mean())

# =========================================================
# FINAL MODEL ON FULL TRAIN
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

nn_model = build_nn(X_train_scaled.shape[1])

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    shuffle=False,   
    verbose=0
)

print("Final NN stopped at epoch:", len(history.history["loss"]))

embedding_model = Model(
    inputs=nn_model.input,
    outputs=nn_model.get_layer("embedding").output
)

X_train_emb = embedding_model.predict(X_train_scaled, verbose=0)
X_test_emb  = embedding_model.predict(X_test_scaled, verbose=0)

xgb_model = build_xgb()
xgb_model.fit(X_train_emb, y_train)

# =========================================================
# FINAL METRICS (THRESHOLD = 0.5)
# =========================================================
train_prob = xgb_model.predict_proba(X_train_emb)[:, 1]
test_prob  = xgb_model.predict_proba(X_test_emb)[:, 1]

train_m = get_metrics(y_train, train_prob)
test_m  = get_metrics(y_test, test_prob)

print("\n FINAL TRAIN METRICS:", train_m)
print(" FINAL TEST METRICS:", test_m)

# =========================================================
# SAVE RESULTS
# =========================================================
cv_results_df.to_csv("NN_XGB_CV_results.csv", index=False)

final_results = pd.DataFrame([{
    **{f"train_{k}": v for k, v in train_m.items()},
    **{f"test_{k}": v for k, v in test_m.items()}
}])

final_results.to_csv("NN_XGB_final_results.csv", index=False)