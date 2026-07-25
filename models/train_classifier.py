"""
===========================================================
Honeywell Grade Change Intelligence

Future Off-Spec Prediction Model

Author : Team
===========================================================
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

# ===========================================================
# PATHS
# ===========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "paper_grade_dataset_features.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "saved_models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "future_offspec_classifier.pkl"
)

# ===========================================================
# LOAD DATA
# ===========================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(df.shape)

# ===========================================================
# REMOVE LEAKAGE COLUMNS
# ===========================================================

drop_columns = [

    "Timestamp",

    "Transition_ID",

    "Recipe_ID",

    "From_Grade",

    "To_Grade",

    "Operator_Action",

    "Recommendation_Source",

    "Suggested_Setpoint",

    "Future_Basis_Weight",

    "Future_Moisture",

    "Future_Ash",

    "Off_Spec"

]

drop_columns = [

    c for c in drop_columns

    if c in df.columns

]

df = df.drop(columns=drop_columns)

# ===========================================================
# TARGET
# ===========================================================

TARGET = "Future_OffSpec"

X = df.drop(columns=[TARGET])

y = df[TARGET]

# ===========================================================
# ENCODE CATEGORICALS
# ===========================================================

categorical = X.select_dtypes(include=["object","category"]).columns

for col in categorical:

    X[col] = X[col].astype("category").cat.codes

# ===========================================================
# TRAIN TEST SPLIT
# ===========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print()

print("Training Samples :", len(X_train))

print("Testing Samples :", len(X_test))

# ===========================================================
# MODEL
# ===========================================================

model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"

)

# ===========================================================
# TRAIN
# ===========================================================

print()

print("Training Model...")

model.fit(

    X_train,

    y_train

)

# ===========================================================
# PREDICTION
# ===========================================================

pred = model.predict(X_test)

# ===========================================================
# METRICS
# ===========================================================

acc = accuracy_score(y_test,pred)

precision = precision_score(y_test,pred)

recall = recall_score(y_test,pred)

f1 = f1_score(y_test,pred)

print()

print("="*60)

print("MODEL PERFORMANCE")

print("="*60)

print(f"Accuracy  : {acc:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print()

print("Confusion Matrix")

print(

    confusion_matrix(

        y_test,

        pred

    )

)

print()

print(

    classification_report(

        y_test,

        pred

    )

)

# ===========================================================
# FEATURE IMPORTANCE
# ===========================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(

    "Importance",

    ascending=False

)

print()

print("="*60)

print("TOP 15 IMPORTANT FEATURES")

print("="*60)

print(

    importance.head(15)

)

# ===========================================================
# SAVE MODEL
# ===========================================================

joblib.dump(

    model,

    MODEL_PATH

)

print()

print("Model Saved")

print(MODEL_PATH)

print("="*60)