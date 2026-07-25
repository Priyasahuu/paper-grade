"""
===========================================================
Honeywell Grade Change Intelligence

Correlation Engine

Finds hidden relationships between process variables

===========================================================
"""

import os
import pandas as pd
import numpy as np

# ===========================================================
# PATHS
# ===========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(

    PROJECT_DIR,

    "data",

    "processed",

    "paper_grade_dataset_features.csv"

)

OUTPUT_DIR = os.path.join(

    PROJECT_DIR,

    "analytics",

    "results"

)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================
# LOAD DATA
# ===========================================================

print("="*60)
print("Loading Dataset")
print("="*60)

df = pd.read_csv(DATA_PATH)

# ===========================================================
# KEEP ONLY NUMERIC COLUMNS
# ===========================================================

numeric_df = df.select_dtypes(include=np.number)

# ===========================================================
# CORRELATION MATRIX
# ===========================================================

corr_matrix = numeric_df.corr(method="pearson")

corr_matrix.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "correlation_matrix.csv"

    )

)

print("Correlation Matrix Saved")

# ===========================================================
# STRONG CORRELATIONS
# ===========================================================

pairs = []

columns = corr_matrix.columns

for i in range(len(columns)):

    for j in range(i+1,len(columns)):

        corr = corr_matrix.iloc[i,j]

        if abs(corr) >= 0.75:

            pairs.append({

                "Variable_1":columns[i],

                "Variable_2":columns[j],

                "Correlation":round(corr,4),

                "Strength":

                "Positive"

                if corr>0

                else

                "Negative"

            })

strong_corr = pd.DataFrame(pairs)

strong_corr = strong_corr.sort_values(

    by="Correlation",

    key=abs,

    ascending=False

)

strong_corr.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "strong_correlations.csv"

    ),

    index=False

)

print()

print("="*60)

print("Top Hidden Correlations")

print("="*60)

print(

    strong_corr.head(20)

)

# ===========================================================
# BASIS WEIGHT CORRELATION
# ===========================================================

bw_corr = corr_matrix["Basis_Weight"]

bw_corr = bw_corr.drop("Basis_Weight")

bw_corr = bw_corr.sort_values(

    key=abs,

    ascending=False

)

bw_corr.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "basis_weight_correlations.csv"

    )

)

print()

print("="*60)

print("Top Drivers of Basis Weight")

print("="*60)

print(

    bw_corr.head(15)

)

# ===========================================================
# MOISTURE CORRELATION
# ===========================================================

moisture_corr = corr_matrix["Moisture"]

moisture_corr = moisture_corr.drop("Moisture")

moisture_corr = moisture_corr.sort_values(

    key=abs,

    ascending=False

)

moisture_corr.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "moisture_correlations.csv"

    )

)

# ===========================================================
# ASH CORRELATION
# ===========================================================

ash_corr = corr_matrix["Ash"]

ash_corr = ash_corr.drop("Ash")

ash_corr = ash_corr.sort_values(

    key=abs,

    ascending=False

)

ash_corr.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "ash_correlations.csv"

    )

)

print()

print("="*60)

print("Files Generated")

print("="*60)

print("correlation_matrix.csv")

print("strong_correlations.csv")

print("basis_weight_correlations.csv")

print("moisture_correlations.csv")

print("ash_correlations.csv")

print("="*60)