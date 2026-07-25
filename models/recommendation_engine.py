"""
===========================================================
Honeywell Grade Change Intelligence

Recommendation Engine

===========================================================
"""

import os
import joblib
import pandas as pd

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

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "future_offspec_classifier.pkl"
)

REGRESSOR_PATH = os.path.join(
    MODEL_DIR,
    "future_bw_regressor.pkl"
)

OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "recommendation_results.csv"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ===========================================================
# LOAD MODELS
# ===========================================================

classifier = joblib.load(CLASSIFIER_PATH)
regressor = joblib.load(REGRESSOR_PATH)

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
# USE EXACT TRAINING FEATURES
# ===========================================================

classifier_features = classifier.get_booster().feature_names
regressor_features = regressor.get_booster().feature_names

X_classifier = df[classifier_features].copy()
X_regressor = df[regressor_features].copy()

# ===========================================================
# PREDICTIONS
# ===========================================================

predicted_offspec = classifier.predict(X_classifier)
predicted_bw = regressor.predict(X_regressor)

# ===========================================================
# RECOMMENDATION ENGINE
# ===========================================================

recommendations = []
sources = []
expected_results = []

for i in range(len(df)):

    bw = predicted_bw[i]

    upper = df.loc[i, "BW_Upper_Limit"]
    lower = df.loc[i, "BW_Lower_Limit"]

    moisture = df.loc[i, "Moisture"]
    ash = df.loc[i, "Ash"]

    recommendation = "Maintain Current Settings"
    source = "Recipe Rules"
    expected = "Stable Operation"

    if predicted_offspec[i] == 1:

        if bw > upper:

            recommendation = "Reduce Stock Flow by 25 L/min"

            source = "ML Prediction"

            expected = "Decrease Basis Weight"

        elif bw < lower:

            recommendation = "Increase Stock Flow by 25 L/min"

            source = "ML Prediction"

            expected = "Increase Basis Weight"

        elif moisture > 7:

            recommendation = "Increase Steam Pressure by 8 kPa"

            source = "Historical Data"

            expected = "Reduce Moisture"

        elif ash > 15:

            recommendation = "Reduce Filler Flow by 10 L/min"

            source = "Correlation Engine"

            expected = "Reduce Ash"

        else:

            recommendation = "Reduce Machine Speed by 15 m/min"

            source = "Historical Data"

            expected = "Improve Stability"

    recommendations.append(recommendation)
    sources.append(source)
    expected_results.append(expected)

# ===========================================================
# OUTPUT
# ===========================================================

output = df.copy()

output["Predicted_Future_BW"] = predicted_bw
output["Predicted_OffSpec"] = predicted_offspec
output["Recommendation"] = recommendations
output["Recommendation_Source"] = sources
output["Expected_Result"] = expected_results

# ===========================================================
# SAVE
# ===========================================================

output.to_csv(
    OUTPUT_PATH,
    index=False
)

print("=" * 60)
print("Recommendation Engine Completed")
print("=" * 60)

print(output[[
    "Predicted_Future_BW",
    "Predicted_OffSpec",
    "Recommendation",
    "Recommendation_Source",
    "Expected_Result"
]].head())

print()
print("Saved to")
print(OUTPUT_PATH)
print("=" * 60)