import pandas as pd
import random
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

OUTPUT_FILE = DATA_DIR / "student_results.csv"


# ============================================================
# SETTINGS
# ============================================================

TOTAL_STUDENTS = 50

SUBJECTS = [
    "DBMS",
    "Operating Systems",
    "DAA",
    "Computer Networks"
]

CREDITS = 4

BRANCHES = [
    "CSE",
    "CSE",
    "CSE",
    "CSE",
    "IT"
]

SECTIONS = [
    "A",
    "B"
]

SEMESTER = 5


# ============================================================
# STUDENT NAMES
# ============================================================

NAMES = [
    "Rahul",
    "Aman",
    "Ravi",
    "Priya",
    "Neha",
    "Arjun",
    "Ananya",
    "Karan",
    "Simran",
    "Rohit",
    "Aditya",
    "Sneha",
    "Vikas",
    "Pooja",
    "Nikhil",
    "Kavya",
    "Ankit",
    "Shreya",
    "Manish",
    "Riya",
    "Ayush",
    "Nisha",
    "Varun",
    "Muskan",
    "Harsh",
    "Ishita",
    "Abhishek",
    "Sakshi",
    "Yash",
    "Tanya",
    "Mohit",
    "Aditi",
    "Saurabh",
    "Komal",
    "Deepak",
    "Mehak",
    "Akash",
    "Payal",
    "Gaurav",
    "Divya",
    "Rohan",
    "Kriti",
    "Shubham",
    "Anjali",
    "Naveen",
    "Preeti",
    "Raj",
    "Swati",
    "Varsha",
    "Vivek"
]


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)


# ============================================================
# GENERATE DATA
# ============================================================

records = []


for i in range(TOTAL_STUDENTS):

    student_id = f"ST{i + 1:03d}"

    name = NAMES[i]

    branch = random.choice(BRANCHES)

    section = random.choice(SECTIONS)

    # --------------------------------------------------------
    # Create different performance groups
    # --------------------------------------------------------

    performance_group = random.choices(
        [
            "Excellent",
            "Good",
            "Average",
            "Weak"
        ],
        weights=[
            15,
            30,
            35,
            20
        ],
        k=1
    )[0]

    # --------------------------------------------------------
    # Select marks range
    # --------------------------------------------------------

    if performance_group == "Excellent":

        minimum_marks = 80
        maximum_marks = 98

    elif performance_group == "Good":

        minimum_marks = 65
        maximum_marks = 85

    elif performance_group == "Average":

        minimum_marks = 50
        maximum_marks = 70

    else:

        minimum_marks = 35
        maximum_marks = 55

    # --------------------------------------------------------
    # Generate four subject records
    # --------------------------------------------------------

    for subject in SUBJECTS:

        marks = random.randint(
            minimum_marks,
            maximum_marks
        )

        # Small variation between subjects
        variation = random.randint(
            -5,
            5
        )

        marks = marks + variation

        # Keep marks between 0 and 100
        marks = max(
            0,
            min(
                100,
                marks
            )
        )

        records.append(
            {
                "Student_ID": student_id,
                "Name": name,
                "Branch": branch,
                "Semester": SEMESTER,
                "Section": section,
                "Subject": subject,
                "Marks": marks,
                "Credits": CREDITS
            }
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# SAVE CSV
# ============================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# OUTPUT INFORMATION
# ============================================================

print("\n======================================")
print("DATASET GENERATED SUCCESSFULLY")
print("======================================")

print(
    f"\nTotal records: {len(df)}"
)

print(
    f"Total students: {df['Student_ID'].nunique()}"
)

print(
    f"Total subjects: {df['Subject'].nunique()}"
)

print(
    f"\nSaved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nFirst 10 records:"
)

print(
    df.head(10).to_string(
        index=False
    )
)

print(
    "\nDataset generation completed!"
)