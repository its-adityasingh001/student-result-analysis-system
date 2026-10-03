import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
# ============================================================

def load_student_data():
    """
    Load the original student result CSV.
    """

    file_path = "data/student_results.csv"

    try:
        df = pd.read_csv(file_path)
        return df

    except FileNotFoundError:
        print("ERROR: student_results.csv not found.")
        return pd.DataFrame()

    except Exception as e:
        print(f"ERROR while loading data: {e}")
        return pd.DataFrame()


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

def get_basic_statistics(df):
    """
    Return basic statistics of the dataset.
    """

    if df.empty:
        return {}

    statistics = {
        "total_records": len(df),
        "total_students": df["Student_ID"].nunique(),
        "total_subjects": df["Subject"].nunique(),
        "average_marks": round(df["Marks"].mean(), 2),
        "highest_marks": df["Marks"].max(),
        "lowest_marks": df["Marks"].min(),
        "total_branches": df["Branch"].nunique(),
        "total_semesters": df["Semester"].nunique()
    }

    return statistics


# ============================================================
# STUDENT LEVEL ANALYSIS
# ============================================================

def create_student_summary(df):
    """
    Convert subject-level data into student-level summary.
    """

    if df.empty:
        return pd.DataFrame()

    student_summary = (
        df.groupby(
            ["Student_ID", "Name", "Branch", "Semester", "Section"],
            as_index=False
        )
        .agg(
            Average_Marks=("Marks", "mean"),
            Highest_Marks=("Marks", "max"),
            Lowest_Marks=("Marks", "min"),
            Total_Marks=("Marks", "sum"),
            Subject_Count=("Subject", "count")
        )
    )

    student_summary["Average_Marks"] = student_summary[
        "Average_Marks"
    ].round(2)

    student_summary["Percentage"] = student_summary[
        "Average_Marks"
    ]

    return student_summary


# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

def add_performance_category(df):
    """
    Add performance category based on average marks.
    """

    if df.empty:
        return df

    result = df.copy()

    def category(marks):

        if marks >= 80:
            return "Excellent"

        elif marks >= 70:
            return "Good"

        elif marks >= 60:
            return "Average"

        elif marks >= 50:
            return "Needs Improvement"

        else:
            return "Poor"

    result["Performance_Category"] = result[
        "Average_Marks"
    ].apply(category)

    return result


# ============================================================
# SUBJECT ANALYSIS
# ============================================================

def get_subject_analysis(df):
    """
    Calculate subject-wise performance.
    """

    if df.empty:
        return pd.DataFrame()

    subject_analysis = (
        df.groupby("Subject", as_index=False)
        .agg(
            Average_Marks=("Marks", "mean"),
            Highest_Marks=("Marks", "max"),
            Lowest_Marks=("Marks", "min"),
            Students=("Student_ID", "nunique")
        )
    )

    subject_analysis["Average_Marks"] = subject_analysis[
        "Average_Marks"
    ].round(2)

    return subject_analysis


# ============================================================
# PASS / FAIL ANALYSIS
# ============================================================

def calculate_pass_fail(df, pass_marks=40):
    """
    Calculate pass and fail statistics.

    Pass marks default = 40.
    """

    if df.empty:
        return {}

    total = len(df)

    passed = len(
        df[df["Marks"] >= pass_marks]
    )

    failed = len(
        df[df["Marks"] < pass_marks]
    )

    pass_percentage = (
        passed / total * 100
        if total > 0
        else 0
    )

    fail_percentage = (
        failed / total * 100
        if total > 0
        else 0
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_percentage": round(pass_percentage, 2),
        "fail_percentage": round(fail_percentage, 2)
    }


# ============================================================
# TOP STUDENTS
# ============================================================

def get_top_students(df, n=5):
    """
    Return top students according to average marks.
    """

    if df.empty:
        return pd.DataFrame()

    student_summary = create_student_summary(df)

    student_summary = add_performance_category(
        student_summary
    )

    top_students = student_summary.sort_values(
        by="Average_Marks",
        ascending=False
    ).head(n)

    return top_students


# ============================================================
# LOW PERFORMING STUDENTS
# ============================================================

def get_at_risk_students(df, threshold=60):
    """
    Return students whose average marks are below threshold.
    """

    if df.empty:
        return pd.DataFrame()

    student_summary = create_student_summary(df)

    student_summary = add_performance_category(
        student_summary
    )

    risk_students = student_summary[
        student_summary["Average_Marks"] < threshold
    ].sort_values(
        by="Average_Marks",
        ascending=True
    )

    return risk_students


# ============================================================
# GRADE DISTRIBUTION
# ============================================================

def get_grade_distribution(df):
    """
    Create grade distribution based on marks.
    """

    if df.empty:
        return pd.DataFrame()

    def grade(marks):

        if marks >= 90:
            return "A+"

        elif marks >= 80:
            return "A"

        elif marks >= 70:
            return "B"

        elif marks >= 60:
            return "C"

        elif marks >= 50:
            return "D"

        elif marks >= 40:
            return "E"

        else:
            return "F"

    grade_df = df.copy()

    grade_df["Grade"] = grade_df[
        "Marks"
    ].apply(grade)

    distribution = (
        grade_df["Grade"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "Grade",
        "Students"
    ]

    return distribution


# ============================================================
# SECTION ANALYSIS
# ============================================================

def get_section_analysis(df):
    """
    Analyze performance by section.
    """

    if df.empty:
        return pd.DataFrame()

    section_analysis = (
        df.groupby("Section", as_index=False)
        .agg(
            Average_Marks=("Marks", "mean"),
            Students=("Student_ID", "nunique")
        )
    )

    section_analysis["Average_Marks"] = section_analysis[
        "Average_Marks"
    ].round(2)

    return section_analysis


# ============================================================
# BRANCH ANALYSIS
# ============================================================

def get_branch_analysis(df):
    """
    Analyze performance by branch.
    """

    if df.empty:
        return pd.DataFrame()

    branch_analysis = (
        df.groupby("Branch", as_index=False)
        .agg(
            Average_Marks=("Marks", "mean"),
            Students=("Student_ID", "nunique")
        )
    )

    branch_analysis["Average_Marks"] = branch_analysis[
        "Average_Marks"
    ].round(2)

    return branch_analysis


# ============================================================
# SEMESTER ANALYSIS
# ============================================================

def get_semester_analysis(df):
    """
    Analyze performance by semester.
    """

    if df.empty:
        return pd.DataFrame()

    semester_analysis = (
        df.groupby("Semester", as_index=False)
        .agg(
            Average_Marks=("Marks", "mean"),
            Students=("Student_ID", "nunique")
        )
    )

    semester_analysis["Average_Marks"] = semester_analysis[
        "Average_Marks"
    ].round(2)

    return semester_analysis


# ============================================================
# STUDENT PERFORMANCE REPORT
# ============================================================

def get_student_report(df, student_id):
    """
    Return complete report for a particular student.
    """

    if df.empty:
        return pd.DataFrame()

    student_data = df[
        df["Student_ID"] == student_id
    ].copy()

    if student_data.empty:
        return pd.DataFrame()

    student_data = student_data.sort_values(
        by="Marks",
        ascending=False
    )

    return student_data


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def run_complete_analysis():
    """
    Run complete analysis pipeline.
    """

    df = load_student_data()

    if df.empty:
        print("No data available.")
        return

    print("\n======================================")
    print("STUDENT RESULT ANALYSIS")
    print("======================================")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nBasic Statistics:")
    statistics = get_basic_statistics(df)

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\nSubject Analysis:")
    print(get_subject_analysis(df))

    print("\nTop Students:")
    print(get_top_students(df))

    print("\nAt Risk Students:")
    print(get_at_risk_students(df))

    print("\nGrade Distribution:")
    print(get_grade_distribution(df))

    print("\nSection Analysis:")
    print(get_section_analysis(df))

    print("\nBranch Analysis:")
    print(get_branch_analysis(df))

    print("\nSemester Analysis:")
    print(get_semester_analysis(df))

    print("\n======================================")
    print("ANALYSIS COMPLETED")
    print("======================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_complete_analysis()