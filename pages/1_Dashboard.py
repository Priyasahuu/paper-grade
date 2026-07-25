import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Process Dashboard",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "paper_grade_dataset_features.csv"
)

df = pd.read_csv(DATA_PATH)

st.title("🏭 Paper Grade Change Process Dashboard")

st.markdown("---")

# ======================================================
# SIDEBAR
# ======================================================

# Create display labels
transition_df = (
    df[["Transition_ID", "From_Grade", "To_Grade"]]
    .drop_duplicates()
    .sort_values("Transition_ID")
)

transition_df["Display"] = (
    transition_df["From_Grade"] +
    "  →  " +
    transition_df["To_Grade"]
)

# Dropdown shows GSM transition
display_transition = st.sidebar.selectbox(
    "Grade Transition",
    transition_df["Display"]
)

# Get corresponding Transition_ID
transition = transition_df.loc[
    transition_df["Display"] == display_transition,
    "Transition_ID"
].iloc[0]

# Filter data
selected = df[df["Transition_ID"] == transition]
# ======================================================
# KPIs
# ======================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Target BW",
    round(selected["Target_BW"].iloc[-1],2)
)

c2.metric(
    "Current BW",
    round(selected["Basis_Weight"].iloc[-1],2)
)

c3.metric(
    "Moisture",
    round(selected["Moisture"].iloc[-1],2)
)

c4.metric(
    "Ash",
    round(selected["Ash"].iloc[-1],2)
)

st.markdown("---")

# ======================================================
# BASIS WEIGHT
# ======================================================

fig = px.line(
    selected,
    x="Step",
    y=["Basis_Weight","Target_BW"],
    title="Basis Weight vs Target"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# STOCK FLOW
# ======================================================

fig = px.line(
    selected,
    x="Step",
    y="Stock_Flow",
    title="Stock Flow"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# STEAM PRESSURE
# ======================================================

fig = px.line(
    selected,
    x="Step",
    y="Steam_Pressure",
    title="Steam Pressure"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# MACHINE SPEED
# ======================================================

fig = px.line(
    selected,
    x="Step",
    y="Machine_Speed",
    title="Machine Speed"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# MOISTURE
# ======================================================

fig = px.line(
    selected,
    x="Step",
    y="Moisture",
    title="Moisture"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# ASH
# ======================================================

fig = px.line(
    selected,
    x="Step",
    y="Ash",
    title="Ash Content"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# CALIPER
# ======================================================

fig = px.line(
    selected,
    x="Step",
    y="Caliper",
    title="Caliper"
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
    y="Off_Spec",
    title="Off Spec Timeline"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# PROCESS VARIABLES
# ======================================================

st.subheader("Current Process Variables")

st.dataframe(

    selected[[
        "Stock_Flow",
        "Filler_Flow",
        "Steam_Pressure",
        "Machine_Speed",
        "Basis_Weight",
        "Moisture",
        "Ash",
        "Caliper"
    ]].tail(10),

    use_container_width=True

)