import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Grade Change Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.stApp{

background-color:#0F172A;

}

.block-container{

padding-top:1rem;
padding-left:2rem;
padding-right:2rem;
max-width:1500px;

}

section[data-testid="stSidebar"]{

background:#111827;
border-right:1px solid #253046;

}

section[data-testid="stSidebar"] *{

color:white;

}

.hero{

background:linear-gradient(135deg,#0F172A,#1E3A8A);

padding:35px;

border-radius:18px;

margin-bottom:25px;

box-shadow:0px 8px 25px rgba(0,0,0,0.35);

}

.hero-title{

font-size:42px;

font-weight:700;

color:white;

}

.hero-sub{

font-size:18px;

color:#CBD5E1;

margin-top:10px;

}

.card{

background:#1E293B;

padding:14px;

border-radius:16px;

box-shadow:0px 4px 14px rgba(0,0,0,0.25);

border:1px solid #334155;

transition:0.3s;

/* was 160px */
height:90px;

display:flex;
flex-direction:column;
justify-content:center;

}

.card:hover{

border:1px solid #38BDF8;

transform:translateY(-2px);

}

.label{

font-size:14px;

color:#94A3B8;

margin-bottom:4px;

line-height:1.2;

}

.metric{

font-size:24px;

font-weight:700;

color:white;

line-height:1.1;

margin:0;

}

.value{

font-size:12px;

color:#38BDF8;

margin-top:3px;

line-height:1.2;

}

.section{

background:#1E293B;

padding:25px;

border-radius:18px;

margin-top:20px;

border:1px solid #334155;

}

h1,h2,h3,h4{

color:white;

}

p{

color:#CBD5E1;

}

hr{

border:1px solid #334155;

}

</style>

""",unsafe_allow_html=True)

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------







st.sidebar.info(
"""
AI Powered
Grade Change
Decision Support
"""
)

# --------------------------------------------------------
# HERO
# --------------------------------------------------------

st.markdown("""

<div class="hero">

<h1 style="
font-size:60px;
font-weight:900;
color:white;
letter-spacing:-1px;
line-height:1.1;
margin-bottom:12px;
">
Grade Change Intelligence
</h1>

<div class="hero-sub">

AI-driven prediction, recommendation and process analytics for paper manufacturing.

</div>

</div>

""",unsafe_allow_html=True)
# ============================================================
# KPI CARDS
# ============================================================

st.markdown(
"""
<h2 style="
margin-top:-35px;
margin-bottom:12px;
font-size:38px;
font-weight:550;
color:white;">
Process Overview
</h2>
""",
unsafe_allow_html=True
)

col1,col2,col3,col4=st.columns(4)

with col1:

    st.markdown("""

    <div class="card">

    <div class="label">

    Active Transition

    </div>

    <br>

    <div class="metric">

    120

    </div>

    <br>

    <div class="value">

    Running Successfully

    </div>

    </div>

    """,unsafe_allow_html=True)

with col2:

    st.markdown("""

    <div class="card">

    <div class="label">

    Prediction Accuracy

    </div>

    <br>

    <div class="metric">

    97.4%

    </div>

    <br>

    <div class="value">

    XGBoost Models

    </div>

    </div>

    """,unsafe_allow_html=True)

with col3:

    st.markdown("""

    <div class="card">

    <div class="label">

    Future Off-Spec Risk

    </div>

    <br>

    <div class="metric">

    LOW

    </div>

    <br>

    <div class="value">

    Within Limits

    </div>

    </div>

    """,unsafe_allow_html=True)

with col4:

    st.markdown("""

    <div class="card">

    <div class="label">

    Recommendations

    </div>

    <br>

    <div class="metric">

    12

    </div>

    <br>

    <div class="value">

    Generated Today

    </div>

    </div>

    """,unsafe_allow_html=True)

st.write("")

# ============================================================
# SECOND ROW
# ============================================================


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.markdown("## Live Process Monitoring")

left,right=st.columns([3,1])

# ============================================================
# PROCESS TREND
# ============================================================

with left:

    x=np.arange(180)

    target=np.ones(180)*120

    bw=120+2*np.sin(x/12)+1*np.cos(x/20)

    fig=go.Figure()

    fig.add_trace(

        go.Scatter(

            x=x,

            y=bw,

            mode="lines",

            name="Basis Weight",

            line=dict(

                color="#38BDF8",

                width=4

            )

        )

    )

    fig.add_trace(

        go.Scatter(

            x=x,

            y=target,

            mode="lines",

            name="Target",

            line=dict(

                color="#22C55E",

                dash="dash",

                width=2

            )

        )

    )

    fig.update_layout(

        title="Basis Weight Trend",

        paper_bgcolor="#1E293B",

        plot_bgcolor="#1E293B",

        font_color="white",

        height=420,

        margin=dict(

            l=10,

            r=10,

            t=40,

            b=10

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

# ============================================================
# QUALITY SNAPSHOT
# ============================================================

with right:

    fig=go.Figure(

        go.Pie(

            labels=[

                "Within Spec",

                "Risk"

            ],

            values=[92,8],

            hole=0.72,

            marker=dict(

                colors=[

                    "#22C55E",

                    "#EF4444"

                ]

            ),

            textinfo="none"

        )

    )

    fig.update_layout(

        paper_bgcolor="#1E293B",

        height=420,

        font_color="white",

        margin=dict(

            l=10,

            r=10,

            t=40,

            b=10

        ),

        annotations=[

            dict(

                text="<b>92%</b><br>Quality",

                showarrow=False,

                font=dict(

                    size=22,

                    color="white"

                )

            )

        ],

        title="Quality Snapshot"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
left,right=st.columns([2,1])

with left:

    st.markdown("""

    <div class="section">

    <h2>System Overview</h2>

    <p>

    The AI continuously monitors grade transitions,

    predicts future Basis Weight,

    estimates Off-Spec risk,

    discovers hidden process relationships,

    and recommends corrective actions

    before quality limits are exceeded.

    </p>

    </div>

    """,unsafe_allow_html=True)

with right:

    st.markdown("""

    <div class="section">

    <h2>System Health</h2>

    <br>

    <p style="font-size:18px;">

    Models Loaded

    </p>

    <div class="metric">

    2

    </div>

    <br>

    <p style="color:#22C55E;">

    ● Online

    </p>

    <p>

    Last Updated

    </p>

    <p>

    Today

    </p>

    </div>

    """,unsafe_allow_html=True)

st.write("")
# ============================================================
# SECOND ROW
# ============================================================

c1,c2=st.columns(2)

# ============================================================
# STEAM PRESSURE
# ============================================================

with c1:

    steam=260+10*np.sin(x/15)

    fig=px.area(

        x=x,

        y=steam,

        labels={

            "x":"Time",

            "y":"Steam Pressure"

        }

    )

    fig.update_traces(

        line_color="#F59E0B"

    )

    fig.update_layout(

        title="Steam Pressure",

        paper_bgcolor="#1E293B",

        plot_bgcolor="#1E293B",

        font_color="white",

        height=320

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ============================================================
# MACHINE SPEED
# ============================================================

with c2:

    speed=760+15*np.cos(x/14)

    fig=px.line(

        x=x,

        y=speed,

        labels={

            "x":"Time",

            "y":"Machine Speed"

        }

    )

    fig.update_traces(

        line_color="#8B5CF6",

        line_width=4

    )

    fig.update_layout(

        title="Machine Speed",

        paper_bgcolor="#1E293B",

        plot_bgcolor="#1E293B",

        font_color="white",

        height=320

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
    # ============================================================
# AI INSIGHTS
# ============================================================

st.markdown("## AI Decision Support")

left,right=st.columns([2,1])

# ============================================================
# AI RECOMMENDATIONS
# ============================================================

with left:

    st.markdown("""

    <div class="section">

    <h2>AI Recommendation</h2>

    <hr>

    <h3 style="color:#38BDF8;">

    Reduce Stock Flow by 25 L/min

    </h3>

    <p>

    The predicted Basis Weight is expected to exceed
    the upper specification limit during the current
    transition.

    </p>

    <br>

    <table style="width:100%;">

    <tr>

    <td><b>Source</b></td>

    <td>Machine Learning Prediction</td>

    </tr>

    <tr>

    <td><b>Confidence</b></td>

    <td>97.6%</td>

    </tr>

    <tr>

    <td><b>Expected Result</b></td>

    <td>Basis Weight returns inside specification.</td>

    </tr>

    <tr>

    <td><b>Stabilization Gain</b></td>

    <td>18%</td>

    </tr>

    </table>

    </div>

    """,unsafe_allow_html=True)

# ============================================================
# PROCESS HEALTH
# ============================================================

with right:

    st.markdown("""

    <div class="section">

    <h2>Process Health</h2>

    </div>

    """,unsafe_allow_html=True)

    st.progress(0.92)

    st.write("Quality Index : **92%**")

    st.progress(0.84)

    st.write("Prediction Confidence : **84%**")

    st.progress(0.95)

    st.write("Model Availability : **95%**")

    st.progress(0.89)

    st.write("Sensor Health : **89%**")

# ============================================================
# HIDDEN CORRELATIONS
# ============================================================

st.markdown("## Hidden Correlation Discovery")

corr1,corr2,corr3=st.columns(3)

with corr1:

    st.info("""

**Stock Flow ↔ Basis Weight**

Correlation

**0.94**

""")

with corr2:

    st.info("""

**Steam Pressure ↔ Moisture**

Correlation

**-0.87**

""")

with corr3:

    st.info("""

**Machine Speed ↔ Basis Weight**

Correlation

**-0.79**

""")

# ============================================================
# PROCESS STATUS
# ============================================================

st.markdown("## Current Process Status")

status1,status2,status3,status4=status_cols=st.columns(4)

status1.success("Recipe Loaded")

status2.success("Prediction Running")

status3.success("Recommendation Generated")

status4.success("Models Online")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
"""
Grade Change Intelligence Dashboard

Machine Learning • Predictive Analytics • Recommendation Engine • Explainable AI
"""
)