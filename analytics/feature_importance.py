"""
===========================================================
Honeywell Grade Change Intelligence

Feature Importance Analysis

Uses both trained XGBoost models

===========================================================
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ===========================================================
# PATHS
# ===========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "saved_models"
)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "paper_grade_dataset_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "analytics",
    "results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================
# LOAD MODELS
# ===========================================================

classifier = joblib.load(

    os.path.join(
        MODEL_DIR,
        "future_offspec_classifier.pkl"
    )

)

regressor = joblib.load(

    os.path.join(
        MODEL_DIR,
        "future_bw_regressor.pkl"
    )

)

# ===========================================================
# LOAD DATA
# ===========================================================

df = pd.read_csv(DATA_PATH)

# ===========================================================
# ENCODE CATEGORICALS
# ===========================================================

categorical = df.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical:

    df[col] = df[col].astype("category").cat.codes

# ===========================================================
# CLASSIFIER FEATURES
# ===========================================================

classifier_features = classifier.get_booster().feature_names

classifier_importance = pd.DataFrame({

    "Feature": classifier_features,

    "Importance": classifier.feature_importances_

})

classifier_importance = classifier_importance.sort_values(

    by="Importance",

    ascending=False

)

classifier_importance.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "classifier_feature_importance.csv"

    ),

    index=False

)

# ===========================================================
# REGRESSOR FEATURES
# ===========================================================

regressor_features = regressor.get_booster().feature_names

regressor_importance = pd.DataFrame({

    "Feature": regressor_features,

    "Importance": regressor.feature_importances_

})

regressor_importance = regressor_importance.sort_values(

    by="Importance",

    ascending=False

)

regressor_importance.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "regressor_feature_importance.csv"

    ),

    index=False

)

# ===========================================================
# PLOT CLASSIFIER
# ===========================================================

plt.figure(figsize=(10,8))

top_classifier = classifier_importance.head(15)

plt.barh(

    top_classifier["Feature"],

    top_classifier["Importance"]

)

plt.gca().invert_yaxis()

plt.title("Top Features - Future OffSpec Classifier")

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "classifier_feature_importance.png"

    )

)

plt.close()

# ===========================================================
# PLOT REGRESSOR
# ===========================================================

plt.figure(figsize=(10,8))

top_regressor = regressor_importance.head(15)

plt.barh(

    top_regressor["Feature"],

    top_regressor["Importance"]

)

plt.gca().invert_yaxis()

plt.title("Top Features - Future Basis Weight Regressor")

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "regressor_feature_importance.png"

    )

)

plt.close()

# ===========================================================
# PRINT
# ===========================================================

print("="*60)

print("TOP FEATURES - CLASSIFIER")

print("="*60)

print(

    classifier_importance.head(15)

)

print()

print("="*60)

print("TOP FEATURES - REGRESSOR")

print("="*60)

print(

    regressor_importance.head(15)

)

print()

print("Results saved to:")

print(OUTPUT_DIR)

print("="*60)