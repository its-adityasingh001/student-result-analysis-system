import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

FEATURE_FILE = DATA_DIR / "student_features.csv"

MODEL_FILE = DATA_DIR / "student_risk_model.pkl"

RESULT_FILE = DATA_DIR / "model_results.csv"

CONFUSION_FILE = DATA_DIR / "confusion_matrix.csv"

IMPORTANCE_FILE = DATA_DIR / "feature_importance.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading student features...")

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature file not found:\n{FEATURE_FILE}"
    )

df = pd.read_csv(FEATURE_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

TARGET = "Risk_Target"

if TARGET not in df.columns:

    raise ValueError(
        "Risk_Target column not found in dataset."
    )

print("\nTarget distribution:")

print(
    df[TARGET].value_counts()
)


# ============================================================
# CHECK TARGET CLASSES
# ============================================================

if df[TARGET].nunique() < 2:

    raise ValueError(
        "ML training requires at least two target classes."
    )


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Branch",
    "Semester",
    "Section",
    "Computer Networks",
    "DAA",
    "DBMS",
    "Operating Systems",
    "Highest_Marks",
    "Lowest_Marks",
    "Subject_Count"
]


missing_columns = [
    col
    for col in FEATURES
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing feature columns: {missing_columns}"
    )


X = df[FEATURES]

y = df[TARGET]


print("\nFeatures:")

print(FEATURES)

print("\nTarget:")

print(TARGET)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = [
    "Branch",
    "Section"
]

numeric_features = [
    "Semester",
    "Computer Networks",
    "DAA",
    "DBMS",
    "Operating Systems",
    "Highest_Marks",
    "Lowest_Marks",
    "Subject_Count"
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

trained_models = {}

model_predictions = {}


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    model_predictions[
        model_name
    ] = predictions

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append(
        {
            "Model": model_name,

            "Accuracy": round(
                accuracy,
                4
            ),

            "Precision": round(
                precision,
                4
            ),

            "Recall": round(
                recall,
                4
            ),

            "F1_Score": round(
                f1,
                4
            )
        }
    )

    trained_models[
        model_name
    ] = pipeline

    print(
        f"{model_name} completed."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\n======================================"
)

print(
    "MODEL COMPARISON"
)

print(
    "======================================"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .sort_values(
        by="F1_Score",
        ascending=False
    )
    .iloc[0]["Model"]
)


best_model = trained_models[
    best_model_name
]


print(
    "\nSelected model based on F1-score:"
)

print(
    best_model_name
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = best_model.predict(
    X_test
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        final_predictions,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\n======================================"
)

print(
    "CONFUSION MATRIX"
)

print(
    "======================================"
)


classes = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    final_predictions,
    labels=classes
)


print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{c}"
        for c in classes
    ],
    columns=[
        f"Predicted_{c}"
        for c in classes
    ]
)

confusion_df.to_csv(
    CONFUSION_FILE
)

print(
    "\nConfusion matrix saved to:"
)

print(
    CONFUSION_FILE
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "\n======================================"
)

print(
    "FEATURE IMPORTANCE"
)

print(
    "======================================"
)


try:

    fitted_preprocessor = (
        best_model
        .named_steps[
            "preprocessor"
        ]
    )

    fitted_model = (
        best_model
        .named_steps[
            "model"
        ]
    )

    # Get transformed feature names

    transformed_features = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    # --------------------------------------------------------
    # MODEL COEFFICIENTS
    # --------------------------------------------------------

    if hasattr(
        fitted_model,
        "coef_"
    ):

        coefficients = (
            fitted_model
            .coef_[0]
        )

        importance_df = pd.DataFrame(
            {
                "Feature":
                    transformed_features,

                "Importance":
                    np.abs(
                        coefficients
                    )
            }
        )

    # --------------------------------------------------------
    # TREE FEATURE IMPORTANCE
    # --------------------------------------------------------

    elif hasattr(
        fitted_model,
        "feature_importances_"
    ):

        importances = (
            fitted_model
            .feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "Feature":
                    transformed_features,

                "Importance":
                    importances
            }
        )

    else:

        importance_df = pd.DataFrame(
            columns=[
                "Feature",
                "Importance"
            ]
        )


    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    print(
        importance_df.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE FEATURE IMPORTANCE
    # ========================================================

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False
    )


    print(
        "\nFeature importance saved to:"
    )

    print(
        IMPORTANCE_FILE
    )


except Exception as e:

    print(
        "\nWARNING: Could not calculate feature importance."
    )

    print(
        f"Reason: {e}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_FILE
)


print(
    "\n======================================"
)

print(
    "MODEL SAVED SUCCESSFULLY"
)

print(
    "======================================"
)

print(
    MODEL_FILE
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)


print(
    "\nModel results saved to:"
)

print(
    RESULT_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n======================================"
)

print(
    "TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "======================================"
)

print(
    "\nGenerated files:"
)

print(
    f"1. {MODEL_FILE}"
)

print(
    f"2. {RESULT_FILE}"
)

print(
    f"3. {CONFUSION_FILE}"
)

print(
    f"4. {IMPORTANCE_FILE}"
)