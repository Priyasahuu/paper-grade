import os
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Honeywell Grade Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""

<style>

.main{
background:#0f172a;
}

.block-container{
padding-top:1rem;
padding-left:1.8rem;
padding-right:1.8rem;
padding-bottom:1rem;
}

h1{
font-size:46px;
font-weight:800;
color:white;
margin-bottom:0px;
}

h2{
font-size:30px;
font-weight:700;
color:white;
}

[data-testid="stMetric"]{

background:#1e293b;

padding:12px;

border-radius:18px;

border:1px solid rgba(255,255,255,0.08);

box-shadow:0px 4px 18px rgba(0,0,0,.35);

}

[data-testid="stMetricValue"]{

font-size:32px;

font-weight:700;

color:white;

}

[data-testid="stMetricLabel"]{

font-size:15px;

color:#9ca3af;

}

div[data-testid="stVerticalBlock"]>div{

gap:0.8rem;

}

</style>

""",unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH=os.path.join(

BASE_DIR,

"data",

"processed",

"paper_grade_dataset_features.csv"

)

df=pd.read_csv(DATA_PATH)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""

<h1>

🏭 Honeywell Grade Change Intelligence

</h1>

<p style="color:#bfc7d5;font-size:18px;margin-top:-8px">

AI-powered decision support for paper grade transitions

</p>

""",unsafe_allow_html=True)

st.markdown("---")

# ==========================================================
# SIDEBAR
# ==========================================================

transition_df=(

df[["Transition_ID","From_Grade","To_Grade"]]

.drop_duplicates()

.sort_values("Transition_ID")

)

transition_df["Display"]=(
transition_df["From_Grade"]
+"  ➜  "+
transition_df["To_Grade"]
)

display_transition=st.sidebar.selectbox(

"📄 Grade Transition",

transition_df["Display"]

)

transition=transition_df.loc[
transition_df["Display"]==display_transition,
"Transition_ID"
].iloc[0]

selected=df[df["Transition_ID"]==transition]

latest=selected.iloc[-1]

# ==========================================================
# EXECUTIVE KPI
# ==========================================================

k1,k2,k3,k4,k5,k6=st.columns(6)

k1.metric(

"Target BW",

f'{latest["Target_BW"]:.1f}'

)

k2.metric(

"Current BW",

f'{latest["Basis_Weight"]:.1f}'

)

risk_percent=min(

100,

abs(

latest["Basis_Weight"]-

latest["Target_BW"]

)

/latest["Target_BW"]*100*10

)

k3.metric(

"Off-Spec Risk",

f"{risk_percent:.0f}%"

)

health=max(

0,

100-risk_percent

)

k4.metric(

"Health Score",

f"{health:.0f}%"

)

k5.metric(

"Steam",

f'{latest["Steam_Pressure"]:.1f}'

)

k6.metric(

"Machine",

f'{latest["Machine_Speed"]:.0f}'

)

st.markdown("<br>",unsafe_allow_html=True)
# ==========================================================
# EXECUTIVE AI DASHBOARD
# ==========================================================

left, right = st.columns([2.3, 1.2])

# ==========================================================
# LEFT SIDE
# ==========================================================

with left:

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type":"indicator"},{"type":"indicator"}],
            [{"type":"xy","colspan":2},None]
        ],
        vertical_spacing=0.18
    )

    # ---------------------------------------
    # AI Risk Gauge
    # ---------------------------------------

    fig.add_trace(

        go.Indicator(

            mode="gauge+number",

            value=risk_percent,

            title={"text":"AI Off-Spec Risk"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"red"},

                "steps":[

                    {"range":[0,35],"color":"green"},

                    {"range":[35,70],"color":"orange"},

                    {"range":[70,100],"color":"red"}

                ]

            }

        ),

        row=1,

        col=1

    )

    # ---------------------------------------
    # Health Gauge
    # ---------------------------------------

    fig.add_trace(

        go.Indicator(

            mode="gauge+number",

            value=health,

            title={"text":"Process Health"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {"range":[0,40],"color":"#ef4444"},

                    {"range":[40,70],"color":"#f59e0b"},

                    {"range":[70,100],"color":"#22c55e"}

                ]

            }

        ),

        row=1,

        col=2

    )

    # ---------------------------------------
    # Basis Weight Trend
    # ---------------------------------------

    fig.add_trace(

        go.Scatter(

            x=selected["Step"],

            y=selected["Basis_Weight"],

            mode="lines",

            name="Actual",

            line=dict(

                color="#4cc9f0",

                width=3

            )

        ),

        row=2,

        col=1

    )

    fig.add_trace(

        go.Scatter(

            x=selected["Step"],

            y=selected["Target_BW"],

            mode="lines",

            name="Target",

            line=dict(

                color="#2ecc71",

                dash="dash",

                width=2

            )

        ),

        row=2,

        col=1

    )

    fig.update_layout(

        height=620,

        paper_bgcolor="#1e293b",

        plot_bgcolor="#1e293b",

        font_color="white",

        margin=dict(

            l=20,

            r=20,

            t=50,

            b=20

        ),

        legend=dict(

            orientation="h",

            y=1.05

        )

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right:

    st.markdown("## 🤖 AI Recommendation")

    deviation = latest["Basis_Weight"] - latest["Target_BW"]

    if deviation > 5:

        recommendation = "⬇ Reduce Stock Flow"

        explanation = "Basis Weight is above target."

        color = "#ef4444"

    elif deviation < -5:

        recommendation = "⬆ Increase Stock Flow"

        explanation = "Basis Weight is below target."

        color = "#22c55e"

    else:

        recommendation = "✅ Maintain Current Settings"

        explanation = "Process is stable."

        color = "#3b82f6"

    st.markdown(
    f"""
    <div style="
        background:{color};
        padding:18px;
        border-radius:15px;
        color:white;
        font-size:22px;
        font-weight:700;
        text-align:center;
    ">
        {recommendation}
    </div>
    """,
    unsafe_allow_html=True
)

    st.write("")

    st.info(explanation)

    st.markdown("---")

    st.markdown("### 📈 Transition Status")

    progress = int(

        latest["Step"] /

        selected["Step"].max()

        *100

    )

    st.progress(progress/100)

    st.write(f"**Progress : {progress}%**")

    st.markdown("---")

    st.markdown("### ⚙ Current Process")

    st.metric(

        "Stock Flow",

        f'{latest["Stock_Flow"]:.1f}'

    )

    st.metric(

        "Filler Flow",

        f'{latest["Filler_Flow"]:.1f}'

    )

    st.metric(

        "Steam Pressure",

        f'{latest["Steam_Pressure"]:.1f}'

    )

    st.metric(

        "Machine Speed",

        f'{latest["Machine_Speed"]:.0f}'

    )
# ==========================================================
# LIVE PROCESS MONITORING
# ==========================================================

st.markdown("## 📡 Live Process Monitoring")

left,right=st.columns([2.4,1])

# ==========================================================
# LIVE TREND
# ==========================================================

with left:

    fig=go.Figure()

    fig.add_trace(

        go.Scatter(

            x=selected["Step"],

            y=selected["Basis_Weight"],

            mode="lines",

            name="Basis Weight",

            line=dict(

                color="#38bdf8",

                width=4

            )

        )

    )

    fig.add_trace(

        go.Scatter(

            x=selected["Step"],

            y=selected["Target_BW"],

            mode="lines",

            name="Target",

            line=dict(

                color="#22c55e",

                dash="dash",

                width=2

            )

        )

    )

    fig.update_layout(

        height=380,

        paper_bgcolor="#1e293b",

        plot_bgcolor="#1e293b",

        font_color="white",

        margin=dict(l=15,r=15,t=35,b=15),

        title="Basis Weight Trend"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ==========================================================
# QUALITY DONUT
# ==========================================================

with right:

    within=max(

        0,

        int(health)

    )

    risk=100-within

    donut=go.Figure(

        go.Pie(

            values=[within,risk],

            labels=["Within Spec","Risk"],

            hole=.72,

            marker=dict(

                colors=[

                    "#22c55e",

                    "#ef4444"

                ]

            ),

            textinfo="none"

        )

    )

    donut.update_layout(

        height=380,

        paper_bgcolor="#1e293b",

        annotations=[

            dict(

                text=f"<b>{within}%</b><br>Quality",

                showarrow=False,

                font=dict(

                    size=22,

                    color="white"

                )

            )

        ],

        font_color="white",

        margin=dict(

            l=10,

            r=10,

            t=35,

            b=10

        ),

        title="Quality Snapshot"

    )

    st.plotly_chart(

        donut,

        use_container_width=True

    )

st.markdown("")

# ==========================================================
# PROCESS VARIABLES
# ==========================================================

st.markdown("## ⚙ Process Variable Monitoring")

c1,c2,c3=st.columns(3)

# ----------------------------------------------------------
# STOCK FLOW
# ----------------------------------------------------------

with c1:

    fig=go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=latest["Stock_Flow"],

            title={"text":"Stock Flow"},

            gauge={

                "axis":{"range":[1000,1400]},

                "bar":{"color":"#38bdf8"}

            }

        )

    )

    fig.update_layout(

        height=280,

        paper_bgcolor="#1e293b",

        font_color="white"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ----------------------------------------------------------
# STEAM
# ----------------------------------------------------------

with c2:

    fig=go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=latest["Steam_Pressure"],

            title={"text":"Steam Pressure"},

            gauge={

                "axis":{"range":[150,250]},

                "bar":{"color":"orange"}

            }

        )

    )

    fig.update_layout(

        height=280,

        paper_bgcolor="#1e293b",

        font_color="white"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ----------------------------------------------------------
# MOISTURE
# ----------------------------------------------------------

with c3:

    fig=go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=latest["Moisture"],

            title={"text":"Moisture"},

            gauge={

                "axis":{"range":[70,90]},

                "bar":{"color":"#10b981"}

            }

        )

    )

    fig.update_layout(

        height=280,

        paper_bgcolor="#1e293b",

        font_color="white"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

st.markdown("")

# ==========================================================
# SECOND ROW
# ==========================================================

r1,r2,r3=st.columns(3)

# ----------------------------------------------------------
# MACHINE SPEED
# ----------------------------------------------------------

with r1:

    fig=px.line(

        selected,

        x="Step",

        y="Machine_Speed",

        title="Machine Speed"

    )

    fig.update_traces(

        line=dict(

            width=3,

            color="#60a5fa"

        )

    )

    fig.update_layout(

        height=250,

        paper_bgcolor="#1e293b",

        plot_bgcolor="#1e293b",

        font_color="white"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ----------------------------------------------------------
# ASH
# ----------------------------------------------------------

with r2:

    fig=px.area(

        selected,

        x="Step",

        y="Ash",

        title="Ash Content"

    )

    fig.update_traces(

        fillcolor="rgba(249,115,22,.35)",

        line_color="#f97316"

    )

    fig.update_layout(

        height=250,

        paper_bgcolor="#1e293b",

        plot_bgcolor="#1e293b",

        font_color="white"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ----------------------------------------------------------
# CALIPER
# ----------------------------------------------------------

with r3:

    fig=px.bar(

        selected.tail(12),

        x="Step",

        y="Caliper",

        title="Caliper"

    )

    fig.update_traces(

        marker_color="#06b6d4"

    )

    fig.update_layout(

        height=250,

        paper_bgcolor="#1e293b",

        plot_bgcolor="#1e293b",

        font_color="white"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
# ==========================================================
# AI DECISION SUPPORT
# ==========================================================

st.markdown("## 🧠 AI Decision Support System")

left,right=st.columns([1.4,1])

# ==========================================================
# LEFT PANEL
# ==========================================================

with left:

    st.markdown("### 🎯 Recommended Setpoint Adjustments")

    bw_error=latest["Target_BW"]-latest["Basis_Weight"]

    stock_change=round(bw_error*2.4,1)

    steam_change=round(-bw_error*0.35,2)

    speed_change=round(-bw_error*0.18,2)

    filler_change=round(bw_error*0.8,1)

    recommendation=pd.DataFrame({

        "Process Variable":[

            "Stock Flow",

            "Steam Pressure",

            "Machine Speed",

            "Filler Flow"

        ],

        "Current":[

            latest["Stock_Flow"],

            latest["Steam_Pressure"],

            latest["Machine_Speed"],

            latest["Filler_Flow"]

        ],

        "Recommended Change":[

            f"{stock_change:+} L/min",

            f"{steam_change:+} bar",

            f"{speed_change:+} m/min",

            f"{filler_change:+} %"

        ]

    })

    st.dataframe(

        recommendation,

        use_container_width=True,

        hide_index=True

    )

    st.markdown("---")

    st.markdown("### ⏱ Estimated Stabilization")

    stability=max(

        1,

        round(

            abs(bw_error)*0.7,

            1

        )

    )

    progress=min(

        100,

        int(

            health

        )

    )

    st.progress(progress/100)

    st.metric(

        "Expected Stabilization",

        f"{stability} min"

    )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right:

    st.markdown("### 📋 AI Rationale")

    if bw_error>2:

        st.success("""

• Basis Weight is below target.

• Increase Stock Flow gradually.

• Slightly reduce Steam Pressure.

• Maintain current machine speed.

Expected Result:

Basis Weight will converge faster while avoiding Off-Spec paper.

""")

    elif bw_error<-2:

        st.warning("""

• Basis Weight is above target.

• Reduce Stock Flow.

• Increase Machine Speed slightly.

• Maintain dryer stability.

Expected Result:

Avoid overweight paper and reduce grade change loss.

""")

    else:

        st.info("""

• Process is within specification.

• Current settings are stable.

• Continue monitoring transition.

• No corrective action required.

""")

# ==========================================================
# ROOT CAUSE ANALYSIS
# ==========================================================

st.markdown("## 🔍 Root Cause Analysis")

feature_names=[

"Stock Flow",

"Steam Pressure",

"Machine Speed",

"Moisture",

"Filler Flow",

"Ash"

]

importance=[

34,

24,

17,

11,

8,

6

]

importance_df=pd.DataFrame({

    "Feature":feature_names,

    "Importance":importance

})

fig=px.bar(

    importance_df,

    x="Importance",

    y="Feature",

    orientation="h",

    text="Importance",

    color="Importance",

    color_continuous_scale="Blues",

    title="Top Factors Influencing Basis Weight"

)

fig.update_layout(

    height=380,

    paper_bgcolor="#1e293b",

    plot_bgcolor="#1e293b",

    font_color="white",

    coloraxis_showscale=False,

    margin=dict(

        l=20,

        r=20,

        t=40,

        b=20

    )

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ==========================================================
# CORRELATION INSIGHTS
# ==========================================================

st.markdown("## 📊 Hidden Process Relationships")

corr=selected[

[

"Basis_Weight",

"Stock_Flow",

"Steam_Pressure",

"Machine_Speed",

"Moisture",

"Ash",

"Caliper"

]

].corr()

heat=px.imshow(

corr,

text_auto=".2f",

color_continuous_scale="RdBu_r",

aspect="auto"

)

heat.update_layout(

height=520,

paper_bgcolor="#1e293b",

plot_bgcolor="#1e293b",

font_color="white",

margin=dict(

l=20,

r=20,

t=20,

b=20

)

)

st.plotly_chart(

heat,

use_container_width=True

)
# ==========================================================
# OPERATOR CONTROL CENTER
# ==========================================================

st.markdown("## 👨‍🏭 Operator Decision Center")

left,right=st.columns([1.4,1])

# ==========================================================
# LEFT
# ==========================================================

with left:

    st.markdown("### 📈 Grade Change Progress")

    stages=[
        "Recipe Selected",
        "Ramp Started",
        "Quality Monitoring",
        "AI Prediction",
        "Corrective Action",
        "Steady State"
    ]

    current_stage=min(5,int(progress/20))

    cols=st.columns(len(stages))

    for i,col in enumerate(cols):

        if i<=current_stage:

            col.success(stages[i])

        else:

            col.info(stages[i])

    st.markdown("---")

    st.markdown("### 🚨 Live Process Status")

    alarm=[]

    if latest["Basis_Weight"]>latest["Target_BW"]*1.025:
        alarm.append("🔴 Basis Weight Above Specification")

    if latest["Basis_Weight"]<latest["Target_BW"]*0.975:
        alarm.append("🟠 Basis Weight Below Specification")

    if latest["Steam_Pressure"]>210:
        alarm.append("🟡 High Steam Pressure")

    if latest["Moisture"]<76:
        alarm.append("🟡 Low Moisture")

    if len(alarm)==0:

        st.success("✅ No active alarms detected.\n\nProcess operating within safe limits.")

    else:

        for a in alarm:

            st.error(a)

# ==========================================================
# RIGHT
# ==========================================================

with right:

    st.markdown("### 📋 Operator Action")

    decision=st.radio(

        "Recommendation",

        [

            "✅ Accept",

            "⚠ Modify",

            "❌ Reject"

        ]

    )

    remarks=st.text_area(

        "Operator Remarks",

        placeholder="Enter observations..."

    )

    if st.button("Submit Decision"):

        st.success("Operator response recorded successfully.")

# ==========================================================
# PRODUCTION SUMMARY
# ==========================================================

st.markdown("## 🏭 Production Summary")

a,b,c,d=st.columns(4)

a.metric(

    "Transition Success",

    f"{health:.0f}%"

)

paper_saved=max(

0,

round(

health*1.4,

1

)

)

b.metric(

    "Estimated Paper Saved",

    f"{paper_saved} kg"

)

steady=max(

1,

round(

stability,

1

)

)

c.metric(

    "Time To Steady State",

    f"{steady} min"

)

confidence=max(

90,

100-int(risk_percent/2)

)

d.metric(

    "AI Confidence",

    f"{confidence}%"

)

# ==========================================================
# FINAL STATUS
# ==========================================================

st.markdown("---")

if health>=90:

    st.success("""

### ✅ Process Status : STABLE

The grade transition is progressing normally.

No significant deviation is expected.

""")

elif health>=70:

    st.warning("""

### ⚠ Process Status : MONITOR

Minor deviations detected.

Operator supervision recommended.

""")

else:

    st.error("""

### 🔴 Process Status : HIGH RISK

Immediate corrective action recommended.

Potential Off-Spec production detected.

""")

# ==========================================================
# HONEYWELL INDUSTRIAL THEME
# ==========================================================

st.markdown("""
<style>

/* ===========================
BACKGROUND
=========================== */

.stApp{

background:#0b1220;

}

/* ===========================
MAIN CONTAINER
=========================== */

.block-container{

padding-top:1rem;

padding-left:2rem;

padding-right:2rem;

padding-bottom:1rem;

max-width:1700px;

}

/* ===========================
SIDEBAR
=========================== */

[data-testid="stSidebar"]{

background:#111827;

border-right:1px solid rgba(255,255,255,.08);

}

[data-testid="stSidebar"] *{

color:white;

}

/* ===========================
HEADINGS
=========================== */

h1{

font-size:48px !important;

font-weight:800 !important;

letter-spacing:.5px;

color:white;

}

h2{

font-size:30px !important;

font-weight:700 !important;

color:white;

}

h3{

font-size:22px !important;

font-weight:700 !important;

color:white;

}

/* ===========================
KPI CARDS
=========================== */

[data-testid="stMetric"]{

background:linear-gradient(

145deg,

#16213e,

#1f2937

);

border-radius:18px;

padding:12px;

border:1px solid rgba(255,255,255,.06);

box-shadow:

0 6px 20px rgba(0,0,0,.45);

transition:.3s;

}

[data-testid="stMetric"]:hover{

transform:translateY(-4px);

box-shadow:

0 10px 24px rgba(0,0,0,.55);

}

[data-testid="stMetricLabel"]{

font-size:15px;

font-weight:600;

color:#aeb6c2;

}

[data-testid="stMetricValue"]{

font-size:30px;

font-weight:800;

color:white;

}

/* ===========================
DATAFRAME
=========================== */

[data-testid="stDataFrame"]{

border-radius:15px;

overflow:hidden;

border:1px solid rgba(255,255,255,.05);

}

/* ===========================
BUTTON
=========================== */

.stButton>button{

background:#2563eb;

color:white;

border:none;

border-radius:12px;

padding:.55rem 1.4rem;

font-weight:600;

}

.stButton>button:hover{

background:#1d4ed8;

}

/* ===========================
PROGRESS BAR
=========================== */

.stProgress>div>div{

background:#2563eb;

}

/* ===========================
SUCCESS BOX
=========================== */

[data-testid="stAlert"]{

border-radius:15px;

}

/* ===========================
SCROLLBAR
=========================== */

::-webkit-scrollbar{

width:10px;

}

::-webkit-scrollbar-thumb{

background:#334155;

border-radius:8px;

}

</style>
""",unsafe_allow_html=True)            
