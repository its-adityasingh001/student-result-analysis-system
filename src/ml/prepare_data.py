import pandas as pd
from pathlib import Path


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):

    df = pd.read_csv(file_path)

    return df


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_dataset(df):

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

def convert_numeric_columns(df):

    df = df.copy()

    numeric_columns = [
        "Marks",
        "Credits",
        "Semester"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# REMOVE INVALID DATA
# ============================================================

def validate_data(df):

    df = df.copy()

    # Remove rows where Marks is missing
    df = df.dropna(
        subset=["Marks"]
    )

    # Keep marks between 0 and 100
    df = df[
        (df["Marks"] >= 0)
        &
        (df["Marks"] <= 100)
    ]

    return df


# ============================================================
# CREATE GRADE
# ============================================================

def create_grade(df):

    df = df.copy()

    def get_grade(marks):

        if marks >= 90:
            return "A+"

        elif marks >= 80:
            return "A"

        elif marks >= 70:
            return "B+"

        elif marks >= 60:
            return "B"

        elif marks >= 50:
            return "C"

        elif marks >= 40:
            return "D"

        else:
            return "F"

    df["Grade"] = (
        df["Marks"]
        .apply(get_grade)
    )

    return df


# ============================================================
# CREATE RESULT
# ============================================================

def create_result(df):

    df = df.copy()

    df["Result"] = (
        df["Marks"]
        .apply(
            lambda marks:
            "Pass"
            if marks >= 40
            else "Fail"
        )
    )

    return df


# ============================================================
# CREATE PERFORMANCE CATEGORY
# ============================================================

def create_performance_category(df):

    df = df.copy()

    def get_category(marks):

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

    df["Performance_Category"] = (
        df["Marks"]
        .apply(get_category)
    )

    return df


# ============================================================
# CREATE PASS TARGET
# ============================================================

def create_target(df):

    df = df.copy()

    df["Pass_Target"] = (
        df["Result"]
        .map(
            {
                "Pass": 1,
                "Fail": 0
            }
        )
    )

    return df


# ============================================================
# COMPLETE PREPARATION PIPELINE
# ============================================================

def prepare_dataset(file_path):

    # Load
    df = load_dataset(
        file_path
    )

    # Clean
    df = clean_dataset(
        df
    )

    # Numeric conversion
    df = convert_numeric_columns(
        df
    )

    # Validate
    df = validate_data(
        df
    )

    # Create Grade
    df = create_grade(
        df
    )

    # Create Result
    df = create_result(
        df
    )

    # Create Performance Category
    df = create_performance_category(
        df
    )

    # Create ML target
    df = create_target(
        df
    )

    return df


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    # Find project root automatically
    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # CSV path
    file_path = (
        project_root
        / "data"
        / "student_results.csv"
    )

    print(
        "Loading dataset from:"
    )

    print(
        file_path
    )

    try:

        df = prepare_dataset(
            file_path
        )

        print(
            "\n================================"
        )

        print(
            "DATASET PREPARED SUCCESSFULLY"
        )

        print(
            "================================"
        )

        print(
            "\nDataset Shape:"
        )

        print(
            df.shape
        )

        print(
            "\nColumns:"
        )

        print(
            df.columns.tolist()
        )

        print(
            "\nFirst 5 Rows:"
        )

        print(
            df.head()
        )

        print(
            "\nGrade Distribution:"
        )

        print(
            df["Grade"]
            .value_counts()
        )

        print(
            "\nResult Distribution:"
        )

        print(
            df["Result"]
            .value_counts()
        )

        print(
            "\nPerformance Category:"
        )

        print(
            df["Performance_Category"]
            .value_counts()
        )

        print(
            "\nPass Target:"
        )

        print(
            df["Pass_Target"]
            .value_counts()
        )

    except Exception as error:

        print(
            "\nERROR:",
            error
        )