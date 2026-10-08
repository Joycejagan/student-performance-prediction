# 🎓 Student Performance Prediction

A beginner-friendly machine learning project that predicts a student's
academic performance category (**Low**, **Medium**, or **High**) based on
key factors such as attendance, study hours, previous marks, and more.

---

## 📌 Problem Statement

Academic institutions need to identify students who may be at risk of
underperforming so that timely interventions can be made.  Manually
reviewing every student's record is time-consuming and inconsistent.
A machine learning model can automatically analyse student data and flag
those who need extra support.

---

## 🎯 Objective

Build a multi-class classification system that:
- Accepts student-related input features.
- Predicts performance as **Low**, **Medium**, or **High**.
- Compares two ML models and selects the best one.
- Exposes the prediction through a simple web interface.

---

## 🛠️ Technologies Used

| Library / Tool | Purpose |
|----------------|---------|
| Python 3.10+   | Programming language |
| pandas         | Data loading and manipulation |
| numpy          | Numerical operations |
| scikit-learn   | ML models, preprocessing, evaluation |
| matplotlib     | Data visualisation |
| seaborn        | Statistical visualisation |
| joblib         | Model serialisation (save/load) |
| Streamlit      | Interactive web application |

---

## 📁 Project Structure

```
Student Performance Prediction/
│
├── data/
│   └── student_performance.csv      ← dataset (generated or custom)
│
├── models/
│   └── best_model.pkl               ← saved trained model
│
├── src/
│   ├── generate_data.py             ← synthetic dataset generator
│   ├── preprocessing.py             ← data loading & preprocessing pipeline
│   ├── eda.py                       ← exploratory data analysis & plots
│   └── train_model.py               ← model training, evaluation & saving
│
├── reports/                         ← auto-created; stores EDA & model plots
│
├── app.py                           ← Streamlit web application
├── requirements.txt                 ← Python dependencies
└── README.md                        ← this file
```

---

## 📊 Dataset

### Required Columns

If you are supplying your own CSV, place it at `data/student_performance.csv`
and make sure it contains **exactly** these columns:

| Column | Type | Description |
|--------|------|-------------|
| `gender` | categorical | `Male` / `Female` |
| `parental_education` | categorical | `None` / `High School` / `Bachelor` / `Master` |
| `attendance_pct` | numerical | Attendance percentage (40–100) |
| `study_hours_per_week` | numerical | Weekly study hours (0–40) |
| `previous_marks` | numerical | Marks in previous exam (30–100) |
| `assignments_completed` | numerical | Assignments done out of 20 (0–20) |
| `extracurricular` | categorical | `Yes` / `No` |
| `sleep_hours_per_day` | numerical | Daily sleep hours (4–10) |
| `internet_access` | categorical | `Yes` / `No` |
| `performance` | categorical | **Target** — `Low` / `Medium` / `High` |

### Don't have a dataset?

Run the included generator to create 1 000 realistic synthetic rows:

```bash
python src/generate_data.py
```

---

## 🔄 Project Workflow

```
Raw CSV data
     │
     ▼
generate_data.py  ──► data/student_performance.csv
     │
     ▼
preprocessing.py  ──► cleans, encodes, splits → (X_train, X_test, y_train, y_test)
     │
     ▼
eda.py            ──► visualisation plots saved to reports/
     │
     ▼
train_model.py    ──► trains DT & RF, evaluates, picks best, saves to models/
     │
     ▼
app.py            ──► Streamlit UI — user inputs → model prediction
```

---

## 🤖 Machine Learning Algorithms

### 1. Decision Tree Classifier
- Builds a tree of if/else rules by splitting data on the most informative
  feature at each node.
- Easy to visualise and explain.
- Can over-fit on small datasets; controlled here with `max_depth` and
  `min_samples_leaf`.

### 2. Random Forest Classifier
- Trains many Decision Trees on random subsets of data and features,
  then takes a majority vote.
- More accurate and robust than a single Decision Tree.
- Provides feature importances that show which inputs matter most.

Both models are evaluated on the held-out test set (20 % of data).
The model with the **higher weighted F1-score** is saved and used by the app.

---

## ⚙️ How to Install Dependencies

> Recommended: create a virtual environment first.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install all required packages
pip install -r requirements.txt
```

---

## 🗂️ How to Train the Model

Run the following commands **in order** from the project root:

```bash
# Step 1: Generate the synthetic dataset (skip if you have your own CSV)
python src/generate_data.py

# Step 2: (Optional) Run EDA to explore the data and save plots
python src/eda.py

# Step 3: Train, evaluate, and save the best model
python src/train_model.py
```

After Step 3 you will see:
- Evaluation metrics printed to the console.
- Confusion matrices and comparison charts saved in `reports/`.
- The best model saved to `models/best_model.pkl`.

---

## 🚀 How to Run the Streamlit Application

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

1. Fill in the student details in the form.
2. Click **Predict Performance**.
3. The app will show the predicted category (**Low / Medium / High**) along
   with the model's confidence percentages for each class.

---

## 📈 Evaluation Metrics Explained

| Metric | What it measures |
|--------|-----------------|
| **Accuracy** | Percentage of all predictions that are correct |
| **Precision** | Of all "High" predictions, how many were actually "High"? |
| **Recall** | Of all actual "High" students, how many did we correctly identify? |
| **F1-Score** | Harmonic mean of Precision and Recall — balances both |
| **Confusion Matrix** | Table showing correct vs. incorrect predictions per class |

---

## 👨‍💻 Author Notes

- All code is heavily commented and structured for easy understanding.
- No accuracy values are hard-coded — every metric is computed live from
  the actual model predictions on real test data.
- The dataset generator creates **synthetic** data only; it does not
  represent any real students.

---

*Made for college presentation purposes.  Feel free to adapt and extend.*
