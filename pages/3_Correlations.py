import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Correlation Analysis",
    layout="wide"
)

# ======================================================
# PATHS
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = os.path.join(
    BASE_DIR,
    "analytics",
    "results"
)

CORR_PATH = os.path.join(
    RESULT_DIR,
    "correlation_matrix.csv"
)

STRONG_PATH = os.path.join(
    RESULT_DIR,
    "strong_correlations.csv"
)

HIDDEN_PATH = os.path.join(
    RESULT_DIR,
    "hidden_relationship_explanations.csv"
)

# ======================================================
# LOAD
# ======================================================

corr = pd.read_csv(CORR_PATH, index_col=0)

strong = pd.read_csv(STRONG_PATH)

hidden = pd.read_csv(HIDDEN_PATH)

# ======================================================
# TITLE
# ======================================================

st.title("🔍 Correlation Discovery Engine")

st.markdown("---")

# ======================================================
# HEATMAP
# ======================================================

fig = px.imshow(

    corr,

    color_continuous_scale="RdBu_r",

    aspect="auto",

    title="Correlation Heatmap"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ======================================================
# TOP CORRELATIONS
# ======================================================

st.subheader("Top Strong Correlations")

st.dataframe(

    strong,

    use_container_width=True,

    height=450

)

# ======================================================
# HIDDEN RELATIONSHIPS
# ======================================================

st.subheader("AI Discovered Hidden Relationships")

st.dataframe(

    hidden,

    use_container_width=True

)

# ======================================================
# BAR CHART
# ======================================================

if "Importance" in hidden.columns:

    fig = px.bar(

        hidden,

        x="Importance",

        y="Hidden_Feature",

        orientation="h",

        title="Hidden Feature Importance"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ======================================================
# INSIGHTS
# ======================================================

st.subheader("AI Insights")

for _, row in hidden.iterrows():

    st.success(

        f"✅ {row['Hidden_Feature']} significantly influences Basis Weight."

    )

st.info(
    """
These relationships are automatically discovered from historical
process data and may not be explicitly defined in the recipe.
"""
)