import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Result Analysis System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DATA_PATH = DATA_DIR / "student_results.csv"
FEATURE_PATH = DATA_DIR / "student_features.csv"
MODEL_PATH = DATA_DIR / "student_risk_model.pkl"
MODEL_RESULTS_PATH = DATA_DIR / "model_results.csv"
CONFUSION_PATH = DATA_DIR / "confusion_matrix.csv"
IMPORTANCE_PATH = DATA_DIR / "feature_importance.csv"


# ============================================================
# LOAD FUNCTIONS
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Unable to load dataset: {e}")
        return None


@st.cache_data
def load_features():

    if not FEATURE_PATH.exists():
        return None

    try:
        return pd.read_csv(FEATURE_PATH)
    except Exception as e:
        st.error(f"Unable to load student features: {e}")
        return None


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Unable to load trained model: {e}")
        return None


@st.cache_data
def load_model_results():

    if not MODEL_RESULTS_PATH.exists():
        return None

    try:
        return pd.read_csv(MODEL_RESULTS_PATH)
    except Exception as e:
        st.error(f"Unable to load model results: {e}")
        return None


@st.cache_data
def load_confusion_matrix():

    if not CONFUSION_PATH.exists():
        return None

    try:
        return pd.read_csv(
            CONFUSION_PATH,
            index_col=0
        )
    except Exception as e:
        st.error(f"Unable to load confusion matrix: {e}")
        return None


@st.cache_data
def load_feature_importance():

    if not IMPORTANCE_PATH.exists():
        return None

    try:
        return pd.read_csv(
            IMPORTANCE_PATH
        )
    except Exception as e:
        st.error(f"Unable to load feature importance: {e}")
        return None


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()
features_df = load_features()
model = load_model()
model_results = load_model_results()
confusion_matrix_df = load_confusion_matrix()
feature_importance_df = load_feature_importance()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 Student Result Analysis")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Student Analysis",
        "Model Performance",
        "Risk Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Student Result Analysis System\n\n"
    "Python • Pandas • Scikit-learn • Streamlit"
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title(
        "🎓 Student Result Analysis System"
    )

    st.subheader(
        "Academic Performance Analysis & ML-Based Risk Prediction"
    )

    st.write(
        """
        This system analyzes student academic performance,
        provides statistical insights, and uses machine
        learning to identify students who may be academically
        at risk.
        """
    )

    st.divider()

    if features_df is not None:

        total_students = len(features_df)

        average_percentage = (
            features_df["Percentage"].mean()
        )

        highest_percentage = (
            features_df["Percentage"].max()
        )

        if "Risk_Target" in features_df.columns:

            risk_students = int(
                features_df["Risk_Target"].sum()
            )

        else:

            risk_students = 0

    else:

        total_students = 0
        average_percentage = 0
        highest_percentage = 0
        risk_students = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👨‍🎓 Total Students",
            total_students
        )

    with col2:

        st.metric(
            "📊 Average Percentage",
            f"{average_percentage:.2f}%"
        )

    with col3:

        st.metric(
            "🏆 Highest Percentage",
            f"{highest_percentage:.2f}%"
        )

    with col4:

        st.metric(
            "⚠️ Students At Risk",
            risk_students
        )

    st.divider()

    st.subheader(
        "🚀 System Features"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 📊 Academic Analysis

            - Complete student dataset
            - Student-wise performance
            - Subject-wise analysis
            - Top student identification
            - Performance categories
            - Grade distribution
            - At-risk student identification
            - Individual student report
            """
        )

    with col2:

        st.markdown(
            """
            ### 🤖 Machine Learning

            - Logistic Regression
            - Decision Tree
            - Random Forest
            - Model comparison
            - Confusion Matrix
            - Feature Importance
            - Accuracy
            - Precision
            - Recall
            - F1 Score
            - Risk prediction
            """
        )

    st.divider()

    st.subheader(
        "📌 Project Workflow"
    )

    st.markdown(
        """
        **CSV Dataset**
        ↓

        **Data Processing**
        ↓

        **Feature Engineering**
        ↓

        **Machine Learning**
        ↓

        **Model Evaluation**
        ↓

        **Risk Prediction**
        ↓

        **Streamlit Dashboard**
        """
    )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.title(
        "📁 Student Dataset"
    )

    if df is None:

        st.error(
            "student_results.csv was not found."
        )

        st.stop()

    st.subheader(
        "Complete Student Result Dataset"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "📌 Dataset Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Records", len(df))

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        st.metric(
            "Students",
            df["Student_ID"].nunique()
        )

    with col4:
        st.metric(
            "Subjects",
            df["Subject"].nunique()
        )

    st.divider()

    st.subheader(
        "🔍 Missing Values"
    )

    missing_values = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_values.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing_values,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# STUDENT ANALYSIS
# ============================================================

elif page == "Student Analysis":

    st.title(
        "📊 Student Performance Analysis"
    )

    if features_df is None:

        st.error(
            "student_features.csv was not found."
        )

        st.stop()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Students",
            len(features_df)
        )

    with col2:
        st.metric(
            "Average %",
            f"{features_df['Percentage'].mean():.2f}%"
        )

    with col3:
        st.metric(
            "Highest %",
            f"{features_df['Percentage'].max():.2f}%"
        )

    with col4:
        st.metric(
            "Lowest %",
            f"{features_df['Percentage'].min():.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader(
        "🔎 Search & Filters"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        search_student = st.text_input(
            "Search Student",
            placeholder="Student ID or Name"
        )

    with col2:

        branch_options = ["All"] + sorted(
            features_df["Branch"]
            .astype(str)
            .unique()
            .tolist()
        )

        selected_branch = st.selectbox(
            "Branch",
            branch_options
        )

    with col3:

        section_options = ["All"] + sorted(
            features_df["Section"]
            .astype(str)
            .unique()
            .tolist()
        )

        selected_section = st.selectbox(
            "Section",
            section_options
        )

    filtered_df = features_df.copy()

    if search_student:

        search_text = search_student.lower()

        filtered_df = filtered_df[
            filtered_df["Student_ID"]
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                na=False
            )
            |
            filtered_df["Name"]
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                na=False
            )
        ]

    if selected_branch != "All":

        filtered_df = filtered_df[
            filtered_df["Branch"].astype(str)
            == selected_branch
        ]

    if selected_section != "All":

        filtered_df = filtered_df[
            filtered_df["Section"].astype(str)
            == selected_section
        ]

    st.divider()

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "👨‍🎓 Student Performance"
    )

    if filtered_df.empty:

        st.warning(
            "No students found."
        )

    else:

        table_columns = [
            "Student_ID",
            "Name",
            "Branch",
            "Semester",
            "Section",
            "Average_Marks",
            "Highest_Marks",
            "Lowest_Marks",
            "Total_Marks",
            "Percentage",
            "Performance_Category"
        ]

        available_columns = [
            c
            for c in table_columns
            if c in filtered_df.columns
        ]

        st.dataframe(
            filtered_df[
                available_columns
            ].sort_values(
                "Percentage",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # TOP STUDENTS
    # --------------------------------------------------------

    st.subheader(
        "🏆 Top Students"
    )

    top_students = (
        features_df
        .sort_values(
            "Percentage",
            ascending=False
        )
        .head(5)
    )

    st.dataframe(
        top_students[
            [
                "Student_ID",
                "Name",
                "Percentage",
                "Performance_Category"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # SUBJECT PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "📚 Subject-wise Performance"
    )

    subjects = [
        "Computer Networks",
        "DAA",
        "DBMS",
        "Operating Systems"
    ]

    available_subjects = [
        s
        for s in subjects
        if s in features_df.columns
    ]

    if available_subjects:

        subject_average = (
            features_df[
                available_subjects
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            subject_average
        )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE CATEGORY
    # --------------------------------------------------------

    st.subheader(
        "📊 Performance Category Distribution"
    )

    category_data = (
        features_df[
            "Performance_Category"
        ]
        .value_counts()
    )

    st.bar_chart(
        category_data
    )

    st.divider()

    # --------------------------------------------------------
    # AT RISK
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Students At Risk"
    )

    if "Risk_Target" in features_df.columns:

        risk_students = features_df[
            features_df["Risk_Target"] == 1
        ]

        if risk_students.empty:

            st.success(
                "No at-risk students found."
            )

        else:

            st.dataframe(
                risk_students[
                    [
                        "Student_ID",
                        "Name",
                        "Percentage",
                        "Performance_Category"
                    ]
                ].sort_values(
                    "Percentage"
                ),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    # --------------------------------------------------------
    # INDIVIDUAL REPORT
    # --------------------------------------------------------

    st.subheader(
        "👤 Individual Student Report"
    )

    student_ids = (
        features_df[
            "Student_ID"
        ]
        .tolist()
    )

    selected_student = st.selectbox(
        "Select Student",
        student_ids
    )

    student_data = features_df[
        features_df["Student_ID"]
        == selected_student
    ]

    if not student_data.empty:

        student = student_data.iloc[0]

        st.markdown(
            f"### 🎓 {student['Name']} "
            f"({student['Student_ID']})"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Percentage",
                f"{student['Percentage']:.2f}%"
            )

        with col2:
            st.metric(
                "Average",
                f"{student['Average_Marks']:.2f}"
            )

        with col3:
            st.metric(
                "Highest",
                student["Highest_Marks"]
            )

        with col4:
            st.metric(
                "Lowest",
                student["Lowest_Marks"]
            )

        st.write(
            f"**Performance:** "
            f"{student['Performance_Category']}"
        )

        student_subjects = {}

        for subject in subjects:

            if subject in student.index:

                student_subjects[
                    subject
                ] = student[subject]

        if student_subjects:

            st.bar_chart(
                pd.DataFrame(
                    {
                        "Marks":
                            student_subjects
                    }
                )
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "🤖 Machine Learning Model Performance"
    )

    st.write(
        """
        This section evaluates the machine learning models
        used for academic risk prediction.
        """
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    if model_results is None:

        st.error(
            "model_results.csv not found."
        )

    else:

        st.subheader(
            "📊 Model Comparison"
        )

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if "F1_Score" in model_results.columns:

            best_row = model_results.loc[
                model_results[
                    "F1_Score"
                ].idxmax()
            ]

            best_model_name = best_row[
                "Model"
            ]

            best_f1 = best_row[
                "F1_Score"
            ]

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Selected Model",
                    best_model_name
                )

            with col2:

                st.metric(
                    "F1 Score",
                    f"{best_f1:.4f}"
                )

        st.divider()

        # ----------------------------------------------------
        # METRIC CHARTS
        # ----------------------------------------------------

        if "Accuracy" in model_results.columns:

            st.subheader(
                "🎯 Accuracy"
            )

            st.bar_chart(
                model_results[
                    [
                        "Model",
                        "Accuracy"
                    ]
                ].set_index(
                    "Model"
                )
            )

        if "Precision" in model_results.columns:

            st.subheader(
                "🎯 Precision"
            )

            st.bar_chart(
                model_results[
                    [
                        "Model",
                        "Precision"
                    ]
                ].set_index(
                    "Model"
                )
            )

        if "Recall" in model_results.columns:

            st.subheader(
                "🔍 Recall"
            )

            st.bar_chart(
                model_results[
                    [
                        "Model",
                        "Recall"
                    ]
                ].set_index(
                    "Model"
                )
            )

        if "F1_Score" in model_results.columns:

            st.subheader(
                "⚖️ F1 Score"
            )

            st.bar_chart(
                model_results[
                    [
                        "Model",
                        "F1_Score"
                    ]
                ].set_index(
                    "Model"
                )
            )

    st.divider()

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "📊 Confusion Matrix"
    )

    if confusion_matrix_df is None:

        st.warning(
            "confusion_matrix.csv not found."
        )

    else:

        st.dataframe(
            confusion_matrix_df,
            use_container_width=True
        )

        st.write(
            """
            The confusion matrix shows the relationship
            between actual student risk classes and the
            classes predicted by the selected model.
            """
        )

        st.bar_chart(
            confusion_matrix_df
        )

    st.divider()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "🔥 Feature Importance"
    )

    if feature_importance_df is None:

        st.warning(
            "feature_importance.csv not found."
        )

    elif feature_importance_df.empty:

        st.warning(
            "No feature importance data available."
        )

    else:

        display_importance = (
            feature_importance_df
            .copy()
        )

        # Remove preprocessing prefixes
        display_importance[
            "Feature"
        ] = (
            display_importance[
                "Feature"
            ]
            .astype(str)
            .str.replace(
                "num__",
                "",
                regex=False
            )
            .str.replace(
                "cat__",
                "",
                regex=False
            )
        )

        st.dataframe(
            display_importance,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "📈 Feature Importance Chart"
        )

        chart_data = (
            display_importance
            .head(10)
            .set_index(
                "Feature"
            )
        )

        st.bar_chart(
            chart_data[
                "Importance"
            ]
        )

        st.info(
            """
            Feature importance indicates which input
            variables contributed more strongly to the
            selected model's predictions. It should not
            be interpreted as proof that a feature causes
            academic risk.
            """
        )

    st.divider()

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "ℹ️ Model Information"
    )

    st.markdown(
        """
        ### Logistic Regression

        A linear classification algorithm used to classify
        students into academic risk categories.

        ### Decision Tree

        A tree-based classification algorithm that uses
        decision rules based on student features.

        ### Random Forest

        An ensemble learning algorithm that combines
        multiple decision trees.

        ### Evaluation Metrics

        **Accuracy:** Percentage of predictions that were
        correct.

        **Precision:** Proportion of predicted positive
        cases that were actually positive.

        **Recall:** Proportion of actual positive cases
        correctly identified.

        **F1 Score:** Harmonic mean of precision and recall.
        """
    )

    st.warning(
        """
        ⚠️ The current dataset contains only 5 students,
        with only 2 samples in the test set. Therefore,
        the displayed metrics are suitable for demonstrating
        the project pipeline but should not be treated as
        reliable real-world model performance.
        """
    )


# ============================================================
# RISK PREDICTION
# ============================================================

elif page == "Risk Prediction":

    st.title(
        "🔮 Student Risk Prediction"
    )

    st.write(
        """
        Enter student academic information to predict
        academic risk.
        """
    )

    if model is None:

        st.error(
            "Trained model not found."
        )

        st.info(
            "Run: python src/ml/train_model.py"
        )

        st.stop()

    col1, col2 = st.columns(2)

    with col1:

        branch = st.selectbox(
            "Branch",
            ["CSE"]
        )

        semester = st.number_input(
            "Semester",
            min_value=1,
            max_value=8,
            value=5
        )

        section = st.selectbox(
            "Section",
            ["A", "B"]
        )

        computer_networks = st.number_input(
            "Computer Networks Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        daa = st.number_input(
            "DAA Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

    with col2:

        dbms = st.number_input(
            "DBMS Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        operating_systems = st.number_input(
            "Operating Systems Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        highest_marks = st.number_input(
            "Highest Marks",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

        lowest_marks = st.number_input(
            "Lowest Marks",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        subject_count = st.number_input(
            "Number of Subjects",
            min_value=1,
            max_value=20,
            value=4
        )

    st.divider()

    if st.button(
        "🔮 Predict Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "Branch": [branch],
                "Semester": [semester],
                "Section": [section],
                "Computer Networks": [
                    computer_networks
                ],
                "DAA": [daa],
                "DBMS": [dbms],
                "Operating Systems": [
                    operating_systems
                ],
                "Highest_Marks": [
                    highest_marks
                ],
                "Lowest_Marks": [
                    lowest_marks
                ],
                "Subject_Count": [
                    subject_count
                ]
            }
        )

        try:

            prediction = model.predict(
                input_data
            )[0]

            st.divider()

            if prediction == 1:

                st.error(
                    "⚠️ Student is At Risk"
                )

            else:

                st.success(
                    "✅ Student is Not At Risk"
                )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                classes = list(
                    model.classes_
                )

                if 1 in classes:

                    risk_index = classes.index(1)

                    risk_probability = (
                        probabilities[
                            risk_index
                        ] * 100
                    )

                    st.subheader(
                        "📊 Risk Probability"
                    )

                    st.metric(
                        "Risk Probability",
                        f"{risk_probability:.2f}%"
                    )

                    st.progress(
                        min(
                            max(
                                risk_probability / 100,
                                0.0
                            ),
                            1.0
                        )
                    )

            st.subheader(
                "📋 Prediction Input"
            )

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Student Result Analysis System | "
    "Python + Pandas + Scikit-learn + Streamlit"
)