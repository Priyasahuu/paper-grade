import os
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Recommendations",
    layout="wide"
)

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "recommendation_results.csv"
)

FEEDBACK_PATH = os.path.join(
    BASE_DIR,
    "results",
    "operator_feedback.csv"
)

os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(RESULT_PATH)

st.title("🤖 AI Recommendation Engine")

st.markdown("---")

# ==========================================================
# GRADE TRANSITION SIDEBAR
# ==========================================================

transition_df = (
    df[["Transition_ID", "From_Grade", "To_Grade"]]
    .drop_duplicates()
    .sort_values("Transition_ID")
)

# If your dataset has encoded values (0,1,2,3)
grade_map = {
    0: "42 GSM",
    1: "70 GSM",
    2: "90 GSM",
    3: "120 GSM"
}

transition_df["Display"] = (
    transition_df["From_Grade"].map(grade_map)
    + " → "
    + transition_df["To_Grade"].map(grade_map)
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

latest = selected.iloc[-1]

# ==========================================================
# KPIs
# ==========================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Predicted Future BW",
    round(latest["Predicted_Future_BW"],2)
)

risk = "HIGH" if latest["Predicted_OffSpec"]==1 else "LOW"

c2.metric(
    "Future Off Spec",
    risk
)

c3.metric(
    "Current BW",
    round(latest["Basis_Weight"],2)
)

st.markdown("---")

# ==========================================================
# RECOMMENDATION
# ==========================================================

st.subheader("AI Recommendation")

st.success(latest["Recommendation"])

st.write("### Source of Inference")

st.info(latest["Recommendation_Source"])

st.write("### Expected Result")

st.warning(latest["Expected_Result"])

# ==========================================================
# PROCESS VARIABLES
# ==========================================================

st.subheader("Current Process Variables")

st.dataframe(

    latest[[
        "Stock_Flow",
        "Filler_Flow",
        "Steam_Pressure",
        "Machine_Speed",
        "Basis_Weight",
        "Moisture",
        "Ash",
        "Caliper"
    ]].to_frame().T,

    use_container_width=True

)

# ==========================================================
# ACCEPT / REJECT
# ==========================================================

st.markdown("---")

st.subheader("Operator Decision")

decision = st.radio(

    "Accept Recommendation?",

    [

        "Accept",

        "Reject"

    ]

)

comments = st.text_area(

    "Operator Comments"

)

if st.button("Submit Feedback"):

    feedback = pd.DataFrame({

        "Timestamp":[pd.Timestamp.now()],

        "Transition_ID":[transition],

        "Recommendation":[latest["Recommendation"]],

        "Decision":[decision],

        "Comments":[comments]

    })

    if os.path.exists(FEEDBACK_PATH):

        old = pd.read_csv(FEEDBACK_PATH)

        feedback = pd.concat(

            [

                old,

                feedback

            ],

            ignore_index=True

        )

    feedback.to_csv(

        FEEDBACK_PATH,

        index=False

    )

    st.success("Feedback Saved Successfully")

# ==========================================================
# HISTORY
# ==========================================================

if os.path.exists(FEEDBACK_PATH):

    st.markdown("---")

    st.subheader("Previous Feedback")

    history = pd.read_csv(FEEDBACK_PATH)

    st.dataframe(

        history,

        use_container_width=True

    )