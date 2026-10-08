"""
app.py
------
Streamlit web application for Student Performance Prediction.

This app loads the saved trained model and lets a user enter student
information to get an instant performance prediction (Low / Medium / High).

Run:
    streamlit run app.py
"""

import os
import sys

import joblib
import numpy as np
import streamlit as st

# Make sure src/ is on the path so we can import preprocessing helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_FEATURES,
)

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")

PERFORMANCE_COLORS = {
    "High":   "#2ecc71",   # green
    "Medium": "#f39c12",   # orange
    "Low":    "#e74c3c",   # red
}

PERFORMANCE_MESSAGES = {
    "High":   "🎉 Excellent! This student is predicted to perform at a HIGH level.",
    "Medium": "📘 Good effort! This student is predicted to perform at a MEDIUM level.",
    "Low":    "⚠️ This student may need extra support. Performance is predicted as LOW.",
}


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)


# ── load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved model payload from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ── encode a single student's inputs ─────────────────────────────────────────
def encode_inputs(raw: dict, label_encoders: dict) -> np.ndarray:
    """
    Convert the user-supplied raw values into the numeric array expected
    by the trained model.

    Parameters
    ----------
    raw            : dict mapping feature_name → raw value (string or number)
    label_encoders : dict of fitted LabelEncoders (from the model payload)

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    row = []
    for feature in ALL_FEATURES:
        val = raw[feature]
        if feature in CATEGORICAL_FEATURES:
            le = label_encoders[feature]
            # Handle unseen values gracefully
            if val in le.classes_:
                val = le.transform([val])[0]
            else:
                val = 0   # fallback to first class
        row.append(float(val))
    return np.array(row).reshape(1, -1)


# ── main app ──────────────────────────────────────────────────────────────────
def main():

    # ── header ────────────────────────────────────────────────────────────
    st.title("🎓 Student Performance Predictor")
    st.markdown(
        """
        Enter the student's details below and click **Predict Performance**
        to find out whether the student is likely to perform at a
        **Low**, **Medium**, or **High** level.
        """,
        unsafe_allow_html=False,
    )
    st.divider()

    # ── model check ───────────────────────────────────────────────────────
    payload = load_model()

    if payload is None:
        st.error(
            "**Model not found.**\n\n"
            "Please train the model first by running:\n\n"
            "```\n"
            "python src/generate_data.py\n"
            "python src/train_model.py\n"
            "```",
        )
        st.stop()

    model          = payload["model"]
    label_encoders = payload["label_encoders"]
    model_name     = payload["model_name"]
    class_names    = payload["class_names"]

    # ── sidebar: model info ───────────────────────────────────────────────
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.write(f"**Active model:** {model_name}")

        metrics = payload.get("metrics", {})
        if metrics:
            st.write("**Test-set metrics:**")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy",  f"{metrics.get('accuracy', 0):.2%}")
            col2.metric("F1-Score",  f"{metrics.get('f1', 0):.2%}")
            col1.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            col2.metric("Recall",    f"{metrics.get('recall', 0):.2%}")

        st.divider()
        st.caption(
            "This tool is for educational purposes only. "
            "Predictions are based on a trained ML model and should not "
            "replace professional academic assessments."
        )

    # ── input form ────────────────────────────────────────────────────────
    st.subheader("📝 Enter Student Details")

    with st.form("prediction_form"):

        # Layout: two columns for a cleaner look
        col_left, col_right = st.columns(2)

        with col_left:
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                help="Student's gender",
            )

            parental_education = st.selectbox(
                "Parental Education Level",
                options=["None", "High School", "Bachelor", "Master"],
                index=2,
                help="Highest education level of the student's parent/guardian",
            )

            attendance_pct = st.slider(
                "Attendance (%)",
                min_value=40,
                max_value=100,
                value=75,
                step=1,
                help="Percentage of classes attended",
            )

            study_hours = st.slider(
                "Study Hours per Week",
                min_value=0,
                max_value=40,
                value=15,
                step=1,
                help="Average number of hours the student studies per week",
            )

            previous_marks = st.slider(
                "Previous Marks (out of 100)",
                min_value=30,
                max_value=100,
                value=65,
                step=1,
                help="Marks scored in the previous exam/semester",
            )

        with col_right:

            assignments_completed = st.slider(
                "Assignments Completed (out of 20)",
                min_value=0,
                max_value=20,
                value=12,
                step=1,
                help="Number of assignments submitted out of 20",
            )

            extracurricular = st.selectbox(
                "Extracurricular Activities",
                options=["Yes", "No"],
                help="Whether the student participates in extracurricular activities",
            )

            sleep_hours = st.slider(
                "Sleep Hours per Day",
                min_value=4,
                max_value=10,
                value=7,
                step=1,
                help="Average hours of sleep the student gets per day",
            )

            internet_access = st.selectbox(
                "Internet Access at Home",
                options=["Yes", "No"],
                help="Whether the student has internet access at home",
            )

        # ── submit button ─────────────────────────────────────────────────
        submitted = st.form_submit_button(
            "🔍 Predict Performance",
            use_container_width=True,
            type="primary",
        )

    # ── prediction ────────────────────────────────────────────────────────
    if submitted:
        # Collect raw inputs into a dict
        raw_inputs = {
            "gender":                gender,
            "parental_education":    parental_education,
            "attendance_pct":        attendance_pct,
            "study_hours_per_week":  study_hours,
            "previous_marks":        previous_marks,
            "assignments_completed": assignments_completed,
            "extracurricular":       extracurricular,
            "sleep_hours_per_day":   sleep_hours,
            "internet_access":       internet_access,
        }

        # Encode and predict
        X_input    = encode_inputs(raw_inputs, label_encoders)
        pred_index = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0]

        # Decode numeric prediction back to label
        le_target   = label_encoders["performance"]
        pred_label  = le_target.inverse_transform([pred_index])[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        # Coloured result banner
        color   = PERFORMANCE_COLORS.get(pred_label, "#3b82d4")
        message = PERFORMANCE_MESSAGES.get(pred_label, f"Predicted: {pred_label}")

        st.markdown(
            f"""
            <div style="
                background-color:{color}22;
                border-left: 6px solid {color};
                padding: 16px 20px;
                border-radius: 6px;
                margin-bottom: 12px;
            ">
                <h2 style="color:{color}; margin:0 0 6px 0;">
                    Performance Level: {pred_label}
                </h2>
                <p style="margin:0; font-size:16px;">{message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability breakdown
        st.markdown("**Prediction Confidence:**")
        prob_cols = st.columns(len(class_names))
        for col, cls_name in zip(prob_cols, class_names):
            cls_index  = list(le_target.classes_).index(cls_name)
            prob_value = pred_proba[cls_index]
            col.metric(
                label=cls_name,
                value=f"{prob_value:.1%}",
                delta=None,
            )

        # Input summary table
        with st.expander("📋 View entered student details"):
            import pandas as pd
            summary = {
                "Feature": [k.replace("_", " ").title() for k in raw_inputs],
                "Value":   list(raw_inputs.values()),
            }
            st.table(pd.DataFrame(summary))


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
