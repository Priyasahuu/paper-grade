import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Operator Feedback",
    layout="wide"
)

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEEDBACK_PATH = os.path.join(
    BASE_DIR,
    "results",
    "operator_feedback.csv"
)

st.title("📝 Operator Feedback Analytics")

st.markdown("---")

# ==========================================================
# CHECK FILE
# ==========================================================

if not os.path.exists(FEEDBACK_PATH):

    st.warning("No feedback available yet.")

    st.stop()

# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(FEEDBACK_PATH)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# ==========================================================
# KPIs
# ==========================================================

accept = (df["Decision"]=="Accept").sum()

reject = (df["Decision"]=="Reject").sum()

total = len(df)

c1,c2,c3 = st.columns(3)

c1.metric(
    "Total Feedback",
    total
)

c2.metric(
    "Accepted",
    accept
)

c3.metric(
    "Rejected",
    reject
)

st.markdown("---")

# ==========================================================
# PIE CHART
# ==========================================================

fig = px.pie(

    df,

    names="Decision",

    title="Recommendation Acceptance"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ==========================================================
# BAR
# ==========================================================

transition = (

    df.groupby("Transition_ID")

    .size()

    .reset_index(name="Count")

)

fig = px.bar(

    transition,

    x="Transition_ID",

    y="Count",

    title="Feedback per Transition"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ==========================================================
# TIMELINE
# ==========================================================

fig = px.scatter(

    df,

    x="Timestamp",

    y="Decision",

    color="Decision",

    hover_data=["Transition_ID"],

    title="Operator Decisions Timeline"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ==========================================================
# COMMENTS
# ==========================================================

st.subheader("Operator Comments")

st.dataframe(

    df,

    use_container_width=True

)

# ==========================================================
# DOWNLOAD
# ==========================================================

st.download_button(

    label="📥 Download Feedback CSV",

    data=df.to_csv(index=False),

    file_name="operator_feedback.csv",

    mime="text/csv"

)