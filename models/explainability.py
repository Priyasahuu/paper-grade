"""
===========================================================
Honeywell Grade Change Intelligence

Explainability Engine (SHAP)

===========================================================
"""

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "paper_grade_dataset_features.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "future_offspec_classifier.pkl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "analytics",
    "results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================

print("=" * 60)
print("Loading Model")
print("=" * 60)

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

# ==========================================================
# ENCODE CATEGORICALS
# ==========================================================

categorical = df.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical:
    df[col] = df[col].astype("category").cat.codes

# ==========================================================
# MODEL FEATURES
# ==========================================================

feature_names = model.get_booster().feature_names

X = df[feature_names]

# ==========================================================
# SHAP
# ==========================================================

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X.sample(500, random_state=42))

# ==========================================================
# SUMMARY PLOT
# ==========================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X.sample(500, random_state=42),
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "shap_summary.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# BAR PLOT
# ==========================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X.sample(500, random_state=42),
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "shap_feature_importance.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# EXPLANATION CSV
# ==========================================================

importance = pd.DataFrame({

    "Feature": feature_names,

    "Mean_SHAP_Importance":

    abs(shap_values).mean(axis=0)

})

importance = importance.sort_values(

    by="Mean_SHAP_Importance",

    ascending=False

)

importance.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "shap_feature_importance.csv"

    ),

    index=False

)

print()

print("=" * 60)
print("Explainability Completed")
print("=" * 60)

print(importance.head(15))

print()

print("Generated Files")

print("✔ shap_summary.png")
print("✔ shap_feature_importance.png")
print("✔ shap_feature_importance.csv")

print("=" * 60)