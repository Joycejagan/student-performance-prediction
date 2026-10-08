"""
preprocessing.py
----------------
Data loading and preprocessing pipeline for Student Performance Prediction.

This module provides:
  - load_data()        : load the CSV from disk
  - inspect_data()     : print basic info and missing-value summary
  - preprocess()       : clean, encode, and split the dataset
  - get_feature_names(): return the list of feature columns used

The returned objects are ready to be passed directly to model training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student_performance.csv")

# Target column name
TARGET_COL = "performance"

# Categorical feature columns (will be label-encoded)
CATEGORICAL_FEATURES = [
    "gender",
    "parental_education",
    "extracurricular",
    "internet_access",
]

# Numerical feature columns (used as-is after missing-value fill)
NUMERICAL_FEATURES = [
    "attendance_pct",
    "study_hours_per_week",
    "previous_marks",
    "assignments_completed",
    "sleep_hours_per_day",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ── public API ────────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the student dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.  Defaults to data/student_performance.csv.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame exactly as stored in the CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please run  python src/generate_data.py  first, or place your "
            "own CSV at that path.\n\n"
            "Required columns:\n"
            "  gender, parental_education, attendance_pct,\n"
            "  study_hours_per_week, previous_marks,\n"
            "  assignments_completed, extracurricular,\n"
            "  sleep_hours_per_day, internet_access, performance"
        )
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset useful for EDA / debugging.

    Prints
    ------
    * Shape (rows × columns)
    * Column dtypes
    * Missing-value counts and percentages
    * Basic descriptive statistics for numerical columns
    * Target column value counts
    """
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape : {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("Column data types:")
    print(df.dtypes.to_string())
    print()

    # ── missing values ────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        print("Missing values : None found ✓")
    else:
        print("Missing values:")
        print(missing_df.to_string())
    print()

    # ── descriptive stats ────────────────────────────────────────────────
    print("Numerical feature statistics:")
    print(df[NUMERICAL_FEATURES].describe().round(2).to_string())
    print()

    # ── target distribution ───────────────────────────────────────────────
    if TARGET_COL in df.columns:
        print(f"Target column  '{TARGET_COL}'  distribution:")
        vc = df[TARGET_COL].value_counts()
        for label, count in vc.items():
            print(f"  {label:<8}  {count:>5}  ({count / len(df) * 100:.1f} %)")
    print("=" * 60)


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Clean and transform the raw DataFrame into train / test arrays.

    Steps performed
    ---------------
    1. Drop duplicate rows.
    2. Fill missing numerical values with the column median.
    3. Fill missing categorical values with the column mode.
    4. Label-encode categorical features.
    5. Label-encode the target column (Low=1, Medium=2, High=0 — mapped
       consistently by LabelEncoder alphabetic order).
    6. Split into train / test sets (stratified on the target).

    Parameters
    ----------
    df         : Raw DataFrame (output of load_data).
    test_size  : Fraction of data reserved for testing (default 0.20).
    random_state : Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    label_encoders : dict[str, LabelEncoder]
        One fitted encoder per categorical column (incl. target).
        Needed to decode predictions back to "Low / Medium / High".
    feature_columns : list[str]
        Ordered list of feature column names matching X_train / X_test.
    """
    df = df.copy()

    # ── 1. remove exact duplicates ────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"Removed {dropped} duplicate row(s).")

    # ── 2 & 3. impute missing values ──────────────────────────────────────
    for col in NUMERICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    for col in CATEGORICAL_FEATURES + [TARGET_COL]:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    # ── 4. encode categorical features ───────────────────────────────────
    label_encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # ── 5. encode target ──────────────────────────────────────────────────
    le_target = LabelEncoder()
    df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL].astype(str))
    label_encoders[TARGET_COL] = le_target

    # ── 6. train / test split ─────────────────────────────────────────────
    X = df[ALL_FEATURES].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # keep class proportions in both splits
    )

    return X_train, X_test, y_train, y_test, label_encoders, ALL_FEATURES


def get_feature_names() -> list:
    """Return the ordered list of feature column names."""
    return ALL_FEATURES.copy()
