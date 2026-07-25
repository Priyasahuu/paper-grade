"""
===========================================================
Honeywell Grade Change Intelligence

Future Basis Weight Prediction Model

Author : Team
===========================================================
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

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
    "future_bw_regressor.pkl"
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
# REMOVE DATA LEAKAGE
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

    "Future_OffSpec",

    "Future_Moisture",

    "Future_Ash"

]

drop_columns = [c for c in drop_columns if c in df.columns]

df = df.drop(columns=drop_columns)

TARGET = "Future_Basis_Weight"

X = df.drop(columns=[TARGET])

y = df[TARGET]

# ===========================================================
# ENCODE CATEGORICAL FEATURES
# ===========================================================

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical_cols:

    X[col] = X[col].astype("category").cat.codes

# ===========================================================
# TRAIN TEST SPLIT
# ===========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ===========================================================
# MODEL
# ===========================================================

model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42

)

# ===========================================================
# TRAIN MODEL
# ===========================================================

print("\nTraining Regressor...\n")

model.fit(

    X_train,

    y_train

)

# ===========================================================
# PREDICTIONS
# ===========================================================

predictions = model.predict(X_test)

# ===========================================================
# METRICS
# ===========================================================

mae = mean_absolute_error(

    y_test,

    predictions

)

mse = mean_squared_error(

    y_test,

    predictions

)

rmse = mse ** 0.5

r2 = r2_score(

    y_test,

    predictions

)

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ===========================================================
# FEATURE IMPORTANCE
# ===========================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 15 Important Features\n")

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

print("\nModel Saved Successfully")

print(MODEL_PATH)

print("=" * 60)