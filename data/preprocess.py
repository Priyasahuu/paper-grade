"""
===========================================================
Honeywell Grade Change Intelligence
Preprocessing Pipeline
===========================================================

Tasks
-----
1. Load Dataset
2. Remove Duplicates
3. Handle Missing Values
4. Validate Process Limits
5. Convert Datatypes
6. Sort Time-Series
7. Save Processed Dataset

Author : Team
"""

import os
import pandas as pd
import numpy as np

# ===========================================================
# PATHS
# ===========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "synthetic",
    "paper_grade_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

OUTPUT_FILE = "paper_grade_dataset_processed.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================
# LOAD DATA
# ===========================================================

def load_dataset():

    print("Loading Dataset...")

    df = pd.read_csv(INPUT_FILE)

    return df


# ===========================================================
# REMOVE DUPLICATES
# ===========================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Removed {before-after} duplicate rows")

    return df


# ===========================================================
# HANDLE MISSING VALUES
# ===========================================================

def handle_missing(df):

    numeric = df.select_dtypes(include=np.number).columns

    categorical = df.select_dtypes(exclude=np.number).columns

    for col in numeric:

        df[col] = df[col].fillna(df[col].median())

    for col in categorical:

        df[col] = df[col].fillna("Unknown")

    return df


# ===========================================================
# REMOVE IMPOSSIBLE VALUES
# ===========================================================

def validate_process_limits(df):

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

        df[col] = df[col].clip(lower=low, upper=high)

    return df


# ===========================================================
# DATATYPE CONVERSION
# ===========================================================

def convert_types(df):

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    categorical = [

        "Transition_ID",

        "Recipe_ID",

        "From_Grade",

        "To_Grade",

        "Alarm",

        "Operator_Action",

        "Scenario"

    ]

    for col in categorical:

        if col in df.columns:

            df[col] = df[col].astype("category")

    return df


# ===========================================================
# SORT
# ===========================================================

def sort_dataset(df):

    df = df.sort_values(

        by=[

            "Transition_ID",

            "Timestamp"

        ]

    )

    df = df.reset_index(drop=True)

    return df


# ===========================================================
# TIME FEATURES
# ===========================================================

def basic_time_features(df):

    df["Hour"] = df["Timestamp"].dt.hour

    df["Minute"] = df["Timestamp"].dt.minute

    df["Day"] = df["Timestamp"].dt.day

    df["Elapsed_Minutes"] = (

        df

        .groupby("Transition_ID")

        .cumcount()

    )

    return df


# ===========================================================
# DATA QUALITY REPORT
# ===========================================================

def quality_report(df):

    print("\n")

    print("="*60)

    print("DATASET QUALITY REPORT")

    print("="*60)

    print("Shape")

    print(df.shape)

    print("\n")

    print("Missing Values")

    print(df.isnull().sum())

    print("\n")

    print("Duplicate Rows")

    print(df.duplicated().sum())

    print("\n")

    print("Off Spec Distribution")

    print(df["Off_Spec"].value_counts())

    print("\n")

    print("Numerical Summary")

    print(df.describe())

    print("="*60)


# ===========================================================
# SAVE
# ===========================================================

def save(df):

    output_path = os.path.join(

        OUTPUT_DIR,

        OUTPUT_FILE

    )

    df.to_csv(

        output_path,

        index=False

    )

    print("\nSaved")

    print(output_path)


# ===========================================================
# MAIN
# ===========================================================

def preprocess():

    df = load_dataset()

    df = remove_duplicates(df)

    df = handle_missing(df)

    df = validate_process_limits(df)

    df = convert_types(df)

    df = sort_dataset(df)

    df = basic_time_features(df)

    quality_report(df)

    save(df)


if __name__ == "__main__":

    preprocess()