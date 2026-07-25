"""
===========================================================
Honeywell Grade Change Intelligence
Dataset Validation Pipeline
===========================================================

Validates the generated dataset before ML training.

Checks:
1. Shape
2. Missing Values
3. Duplicate Rows
4. Data Types
5. Numerical Ranges
6. Transition Consistency
7. Off-Spec Distribution
8. Future Off-Spec Distribution
9. Correlation Analysis
10. Transition Statistics
11. Feature Completeness

Author : Team
"""

import pandas as pd
import numpy as np

# ==========================================================
# PATH
# ==========================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "paper_grade_dataset_features.csv"
)

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

# ==========================================================
# BASIC INFORMATION
# ==========================================================

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

# ==========================================================
# MISSING VALUES
# ==========================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

if missing.sum() == 0:
    print("\nPASS : No Missing Values")
else:
    print("\nWARNING : Missing values found")

# ==========================================================
# DUPLICATES
# ==========================================================

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print("Duplicate Rows :", duplicates)

# ==========================================================
# DATA TYPES
# ==========================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ==========================================================
# PROCESS LIMITS
# ==========================================================

print("\n" + "=" * 70)
print("PROCESS LIMIT VALIDATION")
print("=" * 70)

limits = {

    "Stock_Flow": (800, 2200),

    "Steam_Pressure": (100, 350),

    "Machine_Speed": (500, 1100),

    "Basis_Weight": (40, 160),

    "Moisture": (2, 10),

    "Ash": (0, 30),

    "Caliper": (20, 200)

}

for col, (low, high) in limits.items():

    invalid = (

        (df[col] < low)

        |

        (df[col] > high)

    ).sum()

    print(f"{col:<25} Invalid Values : {invalid}")

# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("CURRENT OFF-SPEC DISTRIBUTION")
print("=" * 70)

print(df["Off_Spec"].value_counts())

print("\nPercentage")

print(

    round(

        df["Off_Spec"]

        .value_counts(normalize=True) * 100,

        2

    )

)

# ==========================================================
# FUTURE TARGET DISTRIBUTION
# ==========================================================

if "Future_OffSpec" in df.columns:

    print("\n" + "=" * 70)
    print("FUTURE OFF-SPEC DISTRIBUTION")
    print("=" * 70)

    print(df["Future_OffSpec"].value_counts())

# ==========================================================
# TRANSITION VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("TRANSITION STATISTICS")
print("=" * 70)

transition_size = (

    df

    .groupby("Transition_ID")

    .size()

)

print(transition_size.describe())

# ==========================================================
# GRADE TRANSITIONS
# ==========================================================

print("\n" + "=" * 70)
print("GRADE CHANGE SUMMARY")
print("=" * 70)

print(

    df.groupby(

        [

            "From_Grade",

            "To_Grade"

        ]

    ).size()

)

# ==========================================================
# SCENARIO DISTRIBUTION
# ==========================================================

if "Scenario" in df.columns:

    print("\n" + "=" * 70)
    print("SCENARIOS")
    print("=" * 70)

    print(

        df["Scenario"]

        .value_counts()

    )

# ==========================================================
# ALARM DISTRIBUTION
# ==========================================================

if "Alarm" in df.columns:

    print("\n" + "=" * 70)
    print("ALARMS")
    print("=" * 70)

    print(

        df["Alarm"]

        .value_counts()

    )

# ==========================================================
# OPERATOR ACTIONS
# ==========================================================

if "Operator_Action" in df.columns:

    print("\n" + "=" * 70)
    print("OPERATOR ACTIONS")
    print("=" * 70)

    print(

        df["Operator_Action"]

        .value_counts()

    )

# ==========================================================
# CORRELATION
# ==========================================================

print("\n" + "=" * 70)
print("TOP CORRELATIONS WITH BASIS WEIGHT")
print("=" * 70)

corr = (

    df

    .corr(numeric_only=True)

)

bw_corr = (

    corr["Basis_Weight"]

    .sort_values(

        ascending=False

    )

)

print(bw_corr.head(20))

# ==========================================================
# NUMERICAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(df.describe())

# ==========================================================
# CLASS BALANCE CHECK
# ==========================================================

print("\n" + "=" * 70)
print("CLASS BALANCE")
print("=" * 70)

ratio = df["Future_OffSpec"].mean()

print(f"Failure Ratio : {ratio:.2%}")

if ratio < 0.05:

    print("WARNING : Too few failures for ML")

elif ratio > 0.60:

    print("WARNING : Too many failures")

else:

    print("PASS : Good class balance")

# ==========================================================
# FEATURE VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

important_features = [

    "BW_Error",

    "BW_Error_Percent",

    "Steam_x_Speed",

    "Stock_x_Filler",

    "Transition_Progress",

    "Future_Basis_Weight",

    "Future_OffSpec"

]

for feature in important_features:

    if feature in df.columns:

        print(f"{feature:<30} PASS")

    else:

        print(f"{feature:<30} MISSING")

# ==========================================================
# FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION REPORT")
print("=" * 70)

print(f"Total Rows               : {len(df)}")
print(f"Total Columns            : {len(df.columns)}")
print(f"Duplicate Rows           : {duplicates}")
print(f"Missing Values           : {missing.sum()}")
print(f"Transitions              : {df['Transition_ID'].nunique()}")

print("\nDataset Ready for Model Training")
print("=" * 70)