import pandas as pd

df = pd.read_csv("data/student_results.csv")

print("Student Result Data")
print("-------------------")

print(df.head())

print("\nTotal Records:", len(df))
print("Total Students:", df["Student_ID"].nunique())
print("Total Subjects:", df["Subject"].nunique())