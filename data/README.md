# Student Result Analysis and Risk Prediction System

## 1. Project Overview

The Student Result Analysis and Risk Prediction System is a
machine-learning-based academic analysis application.

The system analyzes student examination results, performs
student-level feature engineering, evaluates academic performance,
and predicts whether a student may be academically at risk.

The project also provides an interactive Streamlit dashboard for
data analysis and student risk prediction.

---

## 2. Objectives

The main objectives of this project are:

- Analyze student academic results
- Calculate student-level performance metrics
- Perform subject-wise analysis
- Identify students who may require academic attention
- Train machine learning classification models
- Compare different machine learning models
- Predict student academic risk
- Provide an interactive dashboard
- Display model performance and feature importance

---

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

---

## 4. Project Workflow

The complete workflow is:

Student Result Dataset
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
Student-Level Dataset
        ↓
Exploratory Data Analysis
        ↓
Machine Learning Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Risk Prediction
        ↓
Streamlit Dashboard

---

## 5. Dataset

The dataset contains student academic records with the following
information:

- Student ID
- Student Name
- Branch
- Semester
- Section
- Subject
- Marks
- Credits

The current generated dataset contains:

- 50 students
- 200 subject records
- 4 subjects

Subjects included:

- DBMS
- Operating Systems
- DAA
- Computer Networks

---

## 6. Feature Engineering

The system converts subject-level records into student-level
features.

The generated features include:

- Average Marks
- Highest Marks
- Lowest Marks
- Total Marks
- Subject Count
- Failed Subjects
- Passed Subjects
- Percentage
- Performance Category
- Risk Target

---

## 7. Machine Learning

The system compares three classification algorithms:

### Logistic Regression

Used as a classification model for predicting student risk.

### Decision Tree

Used to learn decision rules from student academic features.

### Random Forest

Used as an ensemble classification model consisting of multiple
decision trees.

---

## 8. Features Used for Risk Prediction

The machine learning pipeline uses:

- Branch
- Semester
- Section
- Computer Networks
- DAA
- DBMS
- Operating Systems
- Highest Marks
- Lowest Marks
- Subject Count

Categorical features are encoded using OneHotEncoder.

Missing numerical values are handled using median imputation.

Missing categorical values are handled using most-frequent imputation.

---

## 9. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The model with the highest F1-score is selected as the final model.

---

## 10. Current Model Results

For the current generated dataset, all three models achieved:

- Accuracy: 1.00
- Precision: 1.00
- Recall: 1.00
- F1-score: 1.00

The selected model is:

**Logistic Regression**

Confusion Matrix:

```text
[[12, 0],
 [ 0, 8]]