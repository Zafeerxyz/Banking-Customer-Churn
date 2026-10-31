
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


TARGET = "Exited"

ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

# Original dataset columns
ORIGINAL_NUMERICAL_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "Satisfaction_Score",
    "Point_Earned"
]

ORIGINAL_CATEGORICAL_COLUMNS = [
    "Geography",
    "Gender",
    "Card_Type"
]

ORIGINAL_BINARY_COLUMNS = [
    "HasCrCard",
    "IsActiveMember"
]

# Model input columns after feature engineering
NUMERICAL_COLUMNS = ORIGINAL_NUMERICAL_COLUMNS + [
    "Balance_To_Salary"
]

CATEGORICAL_COLUMNS = ORIGINAL_CATEGORICAL_COLUMNS + [
    "Age_Group"
]

BINARY_COLUMNS = ORIGINAL_BINARY_COLUMNS + [
    "Is_Zero_Balance",
    "Product_Risk_Flag"
]


def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    df = data.copy()
    df = df.drop_duplicates()

    numeric_columns = (
        ORIGINAL_NUMERICAL_COLUMNS
        + ORIGINAL_BINARY_COLUMNS
        + [TARGET]
    )

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    for col in ORIGINAL_CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.title()
                .replace("", pd.NA)
                .astype(object)
            )
            df[col] = df[col].where(
                pd.notna(df[col]), np.nan
            )

    if "Age" in df.columns:
        df.loc[
            (df["Age"] < 0) | (df["Age"] > 120),
            "Age"
        ] = np.nan

    df = df.dropna(subset=[TARGET]).copy()

    existing_ids = [
        col for col in ID_COLUMNS if col in df.columns
    ]

    df = df.drop(columns=existing_ids)

    return df


def build_preprocessor():

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("binary",
             SimpleImputer(strategy="most_frequent"),
             BINARY_COLUMNS)
        ],
        remainder="drop"
    )

    return preprocessor

def build_scaled_preprocessor():
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("binary", SimpleImputer(strategy="most_frequent"),
             BINARY_COLUMNS)
        ],
        remainder="drop"
    )