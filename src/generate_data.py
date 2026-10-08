"""
generate_data.py
----------------
Generates a realistic synthetic dataset for Student Performance Prediction.

Run this script once to create:  data/student_performance.csv

Each row represents one student.  The performance label (Low / Medium / High)
is derived deterministically from the feature values so the dataset is
internally consistent and suitable for a classifier to learn from.

Usage:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
import os

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

N_STUDENTS = 1000          # total rows in the generated dataset
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student_performance.csv")


def generate_dataset(n: int = N_STUDENTS) -> pd.DataFrame:
    """
    Build a synthetic student dataset with the following columns:

    Column                   | Type        | Range / Values
    -------------------------|-------------|-----------------------------
    gender                   | categorical | Male / Female
    parental_education       | categorical | None / High School / Bachelor / Master
    attendance_pct           | numerical   | 40 – 100  (%)
    study_hours_per_week     | numerical   | 0 – 40    (hours)
    previous_marks           | numerical   | 30 – 100  (out of 100)
    assignments_completed    | numerical   | 0 – 20    (count)
    extracurricular          | categorical | Yes / No
    sleep_hours_per_day      | numerical   | 4 – 10    (hours)
    internet_access          | categorical | Yes / No
    performance              | categorical | Low / Medium / High  ← TARGET
    """

    gender              = np.random.choice(["Male", "Female"], size=n)
    parental_education  = np.random.choice(
        ["None", "High School", "Bachelor", "Master"],
        size=n, p=[0.10, 0.35, 0.40, 0.15]
    )
    attendance_pct      = np.clip(np.random.normal(75, 12, n), 40, 100).round(1)
    study_hours         = np.clip(np.random.normal(15, 7, n), 0, 40).round(1)
    previous_marks      = np.clip(np.random.normal(65, 15, n), 30, 100).round(1)
    assignments         = np.clip(np.random.normal(12, 4, n), 0, 20).astype(int)
    extracurricular     = np.random.choice(["Yes", "No"], size=n, p=[0.55, 0.45])
    sleep_hours         = np.clip(np.random.normal(7, 1.2, n), 4, 10).round(1)
    internet_access     = np.random.choice(["Yes", "No"], size=n, p=[0.80, 0.20])

    # ── derive performance label from a weighted score ────────────────────
    # Each feature contributes a normalised value in [0, 1] to a composite
    # score.  Adding small noise keeps the boundaries fuzzy and realistic.
    score = (
        (attendance_pct   - 40) / 60 * 30   +   # 30 % weight
        (study_hours      /  40) * 25        +   # 25 % weight
        (previous_marks   - 30) / 70 * 30   +   # 30 % weight
        (assignments      /  20) * 15        +   # 15 % weight
        np.random.normal(0, 3, n)                # noise
    )

    performance = pd.cut(
        score,
        bins=[-np.inf, 30, 60, np.inf],
        labels=["Low", "Medium", "High"]
    )

    df = pd.DataFrame({
        "gender":               gender,
        "parental_education":   parental_education,
        "attendance_pct":       attendance_pct,
        "study_hours_per_week": study_hours,
        "previous_marks":       previous_marks,
        "assignments_completed":assignments,
        "extracurricular":      extracurricular,
        "sleep_hours_per_day":  sleep_hours,
        "internet_access":      internet_access,
        "performance":          performance,
    })

    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nPerformance distribution:")
    print(df["performance"].value_counts())
