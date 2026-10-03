import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path="data/student_results.csv"):
    """
    Load student result data from CSV.
    """
    df = pd.read_csv(file_path)
    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    """
    Clean and validate student result data.
    """

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Required columns
    required_columns = [
        "Student_ID",
        "Name",
        "Branch",
        "Semester",
        "Section",
        "Subject",
        "Marks",
        "Credits"
    ]

    # Check missing columns
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    # Remove rows with missing important values
    df = df.dropna(
        subset=[
            "Student_ID",
            "Name",
            "Subject",
            "Marks"
        ]
    )

    # Convert Marks to numeric
    df["Marks"] = pd.to_numeric(
        df["Marks"],
        errors="coerce"
    )

    # Convert Credits to numeric
    df["Credits"] = pd.to_numeric(
        df["Credits"],
        errors="coerce"
    )

    # Remove invalid marks
    df = df[
        (df["Marks"] >= 0)
        & (df["Marks"] <= 100)
    ]

    # Remove invalid credits
    df = df[
        df["Credits"] > 0
    ]

    # Clean text columns
    text_columns = [
        "Student_ID",
        "Name",
        "Branch",
        "Section",
        "Subject"
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# CALCULATE GRADE
# ============================================================

def calculate_grade(marks):
    """
    Convert marks into grade.
    """

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


# ============================================================
# CALCULATE RESULT
# ============================================================

def calculate_result(marks):
    """
    Determine Pass/Fail status.
    """

    if marks >= 40:
        return "Pass"

    return "Fail"


# ============================================================
# PROCESS DATA
# ============================================================

def process_dataframe(df):
    """
    Clean data and generate Grade and Result columns.
    """

    df = clean_data(df)

    df["Grade"] = df["Marks"].apply(
        calculate_grade
    )

    df["Result"] = df["Marks"].apply(
        calculate_result
    )

    return df


# ============================================================
# PROCESS DEFAULT CSV
# ============================================================

def process_data():
    """
    Load and process the default CSV dataset.
    """

    df = load_data()

    df = process_dataframe(df)

    return df


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    df = process_data()

    print("\nProcessed Data:")
    print(df)

    print("\nTotal Records:", len(df))

    print(
        "Total Students:",
        df["Student_ID"].nunique()
    )

    print(
        "Total Subjects:",
        df["Subject"].nunique()
    )