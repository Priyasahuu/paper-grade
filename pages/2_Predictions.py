import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Predictions",
    layout="wide"
)

# ======================================================
# PATHS
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "paper_grade_dataset_features.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
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

# ======================================================
# LOAD
# ======================================================

df = pd.read_csv(DATA_PATH)

classifier = joblib.load(CLASSIFIER_PATH)
regressor = joblib.load(REGRESSOR_PATH)

# ======================================================
# ENCODE
# ======================================================

# ======================================================
# ENCODE
# ======================================================

categorical = df.select_dtypes(
    include=["object","category"]
).columns

# Do NOT encode grade names
exclude = ["From_Grade", "To_Grade"]

for col in categorical:

    if col not in exclude:

        df[col] = df[col].astype("category").cat.codes

# ======================================================
# FEATURES
# ======================================================

classifier_features = classifier.get_booster().feature_names

regressor_features = regressor.get_booster().feature_names

X_classifier = df[classifier_features]

X_regressor = df[regressor_features]

# ======================================================
# PREDICT
# ======================================================

df["Predicted_OffSpec"] = classifier.predict(X_classifier)

df["Predicted_Basis_Weight"] = regressor.predict(X_regressor)

# ======================================================
# TITLE
# ======================================================

st.title("🤖 AI Prediction Dashboard")

st.markdown("---")

# ======================================================
# SIDEBAR
# ======================================================

# ----------------------------------------------------
# Grade Transition Dropdown
# ----------------------------------------------------

transition_df = (
    df[["Transition_ID", "From_Grade", "To_Grade"]]
    .drop_duplicates()
    .sort_values("Transition_ID")
)

transition_df["Display"] = (
    transition_df["From_Grade"].astype(str)
    + " GSM → "
    + transition_df["To_Grade"].astype(str)
    + " GSM"
)

display_transition = st.sidebar.selectbox(

    "Grade Transition",

    transition_df["Display"]

)

transition = transition_df.loc[
    transition_df["Display"] == display_transition,
    "Transition_ID"
].iloc[0]

selected = df[df["Transition_ID"] == transition]

# ======================================================
# KPIs
# ======================================================

c1,c2,c3 = st.columns(3)

c1.metric(

    "Predicted Basis Weight",

    round(

        selected["Predicted_Basis_Weight"].iloc[-1],

        2

    )

)

c2.metric(

    "Future Off Spec",

    int(

        selected["Predicted_OffSpec"].iloc[-1]

    )

)

risk = "HIGH" if selected["Predicted_OffSpec"].iloc[-1]==1 else "LOW"

c3.metric(

    "Risk Level",

    risk

)

st.markdown("---")

# ======================================================
# BW
# ======================================================

fig = px.line(

    selected,

    x="Step",

    y=[

        "Basis_Weight",

        "Predicted_Basis_Weight",

        "Target_BW"

    ],

    title="Actual vs Predicted Basis Weight"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ======================================================
# OFF SPEC
# ======================================================

fig = px.area(

    selected,

    x="Step",

    y="Predicted_OffSpec",

    title="Predicted Off Spec"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ======================================================
# TABLE
# ======================================================

st.subheader("Prediction Results")

st.dataframe(

    selected[[

        "Step",

        "Basis_Weight",

        "Predicted_Basis_Weight",

        "Target_BW",

        "Predicted_OffSpec"

    ]],

    use_container_width=True

)

# ======================================================
# ALERT
# ======================================================

if selected["Predicted_OffSpec"].iloc[-1]==1:

    st.error(

        "⚠ Future Off-Spec Predicted. Corrective Action Recommended."

    )

else:

    st.success(

        "✅ Process Expected To Remain Stable."

    )