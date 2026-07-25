"""
===========================================================
Honeywell Grade Change Intelligence

Hidden Relationship Discovery Engine

Discovers process relationships not explicitly defined
in the recipe using Mutual Information.

===========================================================
"""

import os
import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

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
# ENCODE CATEGORICALS
# ===========================================================

categorical = df.select_dtypes(
    include=["object","category"]
).columns

encoder = LabelEncoder()

for col in categorical:

    df[col] = encoder.fit_transform(df[col].astype(str))

# ===========================================================
# TARGET
# ===========================================================

TARGET = "Basis_Weight"

features = [

    c for c in df.columns

    if c != TARGET

    and c != "Future_Basis_Weight"

    and c != "Future_OffSpec"

]

X = df[features]

y = df[TARGET]

# ===========================================================
# MUTUAL INFORMATION
# ===========================================================

mi = mutual_info_regression(

    X,

    y,

    random_state=42

)

importance = pd.DataFrame({

    "Feature":features,

    "Mutual_Information":mi

})

importance = importance.sort_values(

    by="Mutual_Information",

    ascending=False

)

importance.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "hidden_relationships.csv"

    ),

    index=False

)

# ===========================================================
# DISCOVER NEW RELATIONSHIPS
# ===========================================================

known_variables = {

    "Stock_Flow",

    "Steam_Pressure",

    "Machine_Speed",

    "Filler_Flow",

    "Basis_Weight",

    "Moisture",

    "Ash",

    "Caliper"

}

hidden = importance[
    ~importance["Feature"].isin(known_variables)
]

hidden = hidden.head(15)

hidden.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "new_discovered_relationships.csv"

    ),

    index=False

)

# ===========================================================
# GENERATE EXPLANATIONS
# ===========================================================

explanations = []

for _, row in hidden.iterrows():

    explanations.append({

        "Hidden_Feature":row["Feature"],

        "Importance":round(row["Mutual_Information"],4),

        "Reason":

        f"{row['Feature']} shows a strong non-linear "

        f"relationship with Basis Weight and "

        f"should be considered during "

        f"grade transition optimization."

    })

explanations = pd.DataFrame(explanations)

explanations.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "hidden_relationship_explanations.csv"

    ),

    index=False

)

# ===========================================================
# PRINT RESULTS
# ===========================================================

print("="*60)
print("TOP DISCOVERED HIDDEN RELATIONSHIPS")
print("="*60)

print(hidden)

print()

print("="*60)
print("Generated Files")
print("="*60)

print("hidden_relationships.csv")
print("new_discovered_relationships.csv")
print("hidden_relationship_explanations.csv")

print("="*60)
