import pandas as pd
from pathlib import Path


# ============================================================
# LOAD PREPARED DATA
# ============================================================

def load_prepared_data(file_path):

    df = pd.read_csv(file_path)

    return df


# ============================================================
# CREATE STUDENT LEVEL FEATURES
# ============================================================

def create_student_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Create subject-wise columns
    # --------------------------------------------------------

    subject_data = (
        df.pivot_table(
            index=[
                "Student_ID",
                "Name",
                "Branch",
                "Semester",
                "Section"
            ],
            columns="Subject",
            values="Marks",
            aggfunc="mean"
        )
        .reset_index()
    )

    # Remove column index name
    subject_data.columns.name = None

    # --------------------------------------------------------
    # Find subject columns
    # --------------------------------------------------------

    subject_columns = [
        column
        for column in subject_data.columns
        if column not in [
            "Student_ID",
            "Name",
            "Branch",
            "Semester",
            "Section"
        ]
    ]

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    subject_data["Average_Marks"] = (
        subject_data[subject_columns]
        .mean(axis=1)
    )

    subject_data["Highest_Marks"] = (
        subject_data[subject_columns]
        .max(axis=1)
    )

    subject_data["Lowest_Marks"] = (
        subject_data[subject_columns]
        .min(axis=1)
    )

    subject_data["Total_Marks"] = (
        subject_data[subject_columns]
        .sum(axis=1)
    )

    subject_data["Subject_Count"] = (
        subject_data[subject_columns]
        .notna()
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Failed subject count
    # --------------------------------------------------------

    subject_data["Failed_Subjects"] = (
        subject_data[subject_columns]
        .lt(40)
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Passed subject count
    # --------------------------------------------------------

    subject_data["Passed_Subjects"] = (
        subject_data[subject_columns]
        .ge(40)
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Overall percentage
    # --------------------------------------------------------

    subject_data["Percentage"] = (
        subject_data["Average_Marks"]
    )

    # --------------------------------------------------------
    # Performance category
    # --------------------------------------------------------

    def get_performance(marks):

        if marks >= 80:
            return "Excellent"

        elif marks >= 70:
            return "Good"

        elif marks >= 60:
            return "Average"

        elif marks >= 50:
            return "Needs Improvement"

        else:
            return "At Risk"

    subject_data["Performance_Category"] = (
        subject_data["Average_Marks"]
        .apply(get_performance)
    )

    # --------------------------------------------------------
    # Risk target
    # --------------------------------------------------------

    subject_data["Risk_Target"] = (
    subject_data["Average_Marks"]
    .apply(
        lambda marks:
        1 if marks < 70 else 0
        )
    )

    # --------------------------------------------------------
    # Round numerical values
    # --------------------------------------------------------

    numerical_columns = [
        "Average_Marks",
        "Highest_Marks",
        "Lowest_Marks",
        "Total_Marks",
        "Percentage"
    ]

    for column in numerical_columns:

        if column in subject_data.columns:

            subject_data[column] = (
                subject_data[column]
                .round(2)
            )

    return subject_data


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    df,
    output_path
):

    df.to_csv(
        output_path,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # --------------------------------------------------------
    # Input CSV
    # --------------------------------------------------------

    input_file = (
        project_root
        / "data"
        / "student_results.csv"
    )

    # --------------------------------------------------------
    # Output CSV
    # --------------------------------------------------------

    output_file = (
        project_root
        / "data"
        / "student_features.csv"
    )

    print(
        "\nLoading dataset..."
    )

    try:

        # Load
        df = load_prepared_data(
            input_file
        )

        print(
            "Original shape:",
            df.shape
        )

        # Create features
        features = create_student_features(
            df
        )

        # Save
        save_features(
            features,
            output_file
        )

        print(
            "\n===================================="
        )

        print(
            "FEATURE ENGINEERING SUCCESSFUL"
        )

        print(
            "===================================="
        )

        print(
            "\nStudent-level shape:",
            features.shape
        )

        print(
            "\nColumns:"
        )

        print(
            features.columns.tolist()
        )

        print(
            "\nStudent Features:"
        )

        print(
            features.to_string(
                index=False
            )
        )

        print(
            "\nSaved to:"
        )

        print(
            output_file
        )

    except Exception as error:

        print(
            "\nERROR:",
            error
        )