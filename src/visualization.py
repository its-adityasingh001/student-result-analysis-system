import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import process_data
from analysis import (
    student_performance,
    subject_performance,
    grade_analysis,
    result_analysis,
    top_students
)


# --------------------------------------------------
# 1. Subject Average Marks
# --------------------------------------------------

def plot_subject_average(df):

    data = subject_performance(df)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=data,
        x="Subject",
        y="Average_Marks"
    )

    plt.title("Average Marks by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Average Marks")

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.show()


# --------------------------------------------------
# 2. Grade Distribution
# --------------------------------------------------

def plot_grade_distribution(df):

    data = grade_analysis(df)

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=data,
        x="Grade",
        y="Count"
    )

    plt.title("Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.show()


# --------------------------------------------------
# 3. Pass / Fail Distribution
# --------------------------------------------------

def plot_result_distribution(df):

    data = result_analysis(df)

    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=data,
        x="Result",
        y="Count"
    )

    plt.title("Pass / Fail Distribution")
    plt.xlabel("Result")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.show()


# --------------------------------------------------
# 4. Student Average Performance
# --------------------------------------------------

def plot_student_performance(df):

    data = student_performance(df)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=data,
        x="Name",
        y="Average_Marks"
    )

    plt.title("Student Average Performance")
    plt.xlabel("Student")
    plt.ylabel("Average Marks")

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.show()


# --------------------------------------------------
# 5. Top 5 Students
# --------------------------------------------------

def plot_top_students(df):

    data = top_students(df, 5)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=data,
        x="Name",
        y="Average_Marks"
    )

    plt.title("Top 5 Students")
    plt.xlabel("Student")
    plt.ylabel("Average Marks")

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.show()


# --------------------------------------------------
# 6. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    # Load and process data
    df = process_data()

    print("\nGenerating visualizations...")

    # Generate charts
    plot_subject_average(df)

    plot_grade_distribution(df)

    plot_result_distribution(df)

    plot_student_performance(df)

    plot_top_students(df)

    print("\nAll visualizations generated successfully!")