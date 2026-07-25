
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