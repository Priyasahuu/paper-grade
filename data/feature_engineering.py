"""
===========================================================
Honeywell Grade Change Intelligence
Feature Engineering Pipeline
===========================================================

Creates process-aware features for
predictive quality control.

Output:
processed/paper_grade_dataset_features.csv
"""

import os
import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

import os
import pandas as pd
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "paper_grade_dataset_processed.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

OUTPUT_FILE = "paper_grade_dataset_features.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading Processed Dataset...")

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# ==========================================================
# GROUP BY TRANSITION
# ==========================================================

transition = df.groupby("Transition_ID")

# ==========================================================
# 1. BASIS WEIGHT ERROR
# ==========================================================

df["BW_Error"] = (

    df["Basis_Weight"]

    -

    df["Target_BW"]

)

df["BW_Error_Percent"] = (

    df["BW_Error"]

    /

    df["Target_BW"]

) * 100

# ==========================================================
# 2. DISTANCE TO LIMITS
# ==========================================================

df["Distance_To_Upper"] = (

    df["BW_Upper_Limit"]

    -

    df["Basis_Weight"]

)

df["Distance_To_Lower"] = (

    df["Basis_Weight"]

    -

    df["BW_Lower_Limit"]

)

# ==========================================================
# 3. LAG FEATURES
# ==========================================================

lag_columns = [

    "Basis_Weight",

    "Stock_Flow",

    "Steam_Pressure",

    "Machine_Speed",

    "Moisture",

    "Ash",

    "Caliper"

]

for col in lag_columns:

    df[f"{col}_Lag1"] = (

        transition[col].shift(1)

    )

    df[f"{col}_Lag2"] = (

        transition[col].shift(2)

    )

# ==========================================================
# 4. RATE OF CHANGE
# ==========================================================

roc_columns = [

    "Stock_Flow",

    "Steam_Pressure",

    "Machine_Speed",

    "Basis_Weight",

    "Moisture"

]

for col in roc_columns:

    df[f"{col}_ROC"] = (

        transition[col].diff()

    )

# ==========================================================
# 5. ROLLING MEAN
# ==========================================================

rolling_columns = [

    "Basis_Weight",

    "Moisture",

    "Steam_Pressure",

    "Machine_Speed"

]

for col in rolling_columns:

    df[f"{col}_RollingMean"] = (

        transition[col]

        .rolling(window=5)

        .mean()

        .reset_index(level=0, drop=True)

    )

# ==========================================================
# 6. ROLLING STD
# ==========================================================

for col in rolling_columns:

    df[f"{col}_RollingStd"] = (

        transition[col]

        .rolling(window=5)

        .std()

        .reset_index(level=0, drop=True)

    )

# ==========================================================
# 7. CUMULATIVE CHANGE
# ==========================================================

df["StockFlow_Deviation"] = (

    df["Stock_Flow"]

    -

    transition["Stock_Flow"]

    .transform("first")

)

df["Steam_Deviation"] = (

    df["Steam_Pressure"]

    -

    transition["Steam_Pressure"]

    .transform("first")

)

df["Speed_Deviation"] = (

    df["Machine_Speed"]

    -

    transition["Machine_Speed"]

    .transform("first")

)

# ==========================================================
# 8. PROCESS INTERACTION FEATURES
# ==========================================================

df["Steam_x_Speed"] = (

    df["Steam_Pressure"]

    *

    df["Machine_Speed"]

)

df["Stock_x_Filler"] = (

    df["Stock_Flow"]

    *

    df["Filler_Flow"]

)

df["Steam_per_Stock"] = (

    df["Steam_Pressure"]

    /

    df["Stock_Flow"]

)

df["Speed_per_Stock"] = (

    df["Machine_Speed"]

    /

    df["Stock_Flow"]

)

# ==========================================================
# 9. QUALITY RATIOS
# ==========================================================

df["Moisture_per_BW"] = (

    df["Moisture"]

    /

    df["Basis_Weight"]

)

df["Ash_per_BW"] = (

    df["Ash"]

    /

    df["Basis_Weight"]

)

# ==========================================================
# 10. GRADE CHANGE PROGRESS
# ==========================================================

df["Transition_Progress"] = (

    transition.cumcount()

    /

    transition["Step"]

    .transform("max")

)

# ==========================================================
# 11. NEAR OFF SPEC
# ==========================================================

df["Near_OffSpec"] = np.where(

    abs(df["BW_Error_Percent"]) > 2,

    1,

    0

)

# ==========================================================
# 12. FUTURE TARGETS
# ==========================================================

forecast_horizon = 5

df["Future_Basis_Weight"] = (

    transition["Basis_Weight"]

    .shift(-forecast_horizon)

)

df["Future_Moisture"] = (

    transition["Moisture"]

    .shift(-forecast_horizon)

)

df["Future_OffSpec"] = (

    transition["Off_Spec"]

    .shift(-forecast_horizon)

)

# ==========================================================
# 13. REMOVE NaN
# ==========================================================

df = df.bfill()

df = df.ffill()

# ==========================================================
# SAVE
# ==========================================================

output_path = os.path.join(

    OUTPUT_DIR,

    OUTPUT_FILE

)

df.to_csv(

    output_path,

    index=False

)

print("="*60)

print("Feature Engineering Completed")

print("="*60)

print(df.shape)

print(output_path)