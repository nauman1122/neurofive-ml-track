import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Outcome Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "student_outcome_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Outcome Prediction")

st.write(
    """
    This application predicts a student's academic outcome
    using a trained Machine Learning model.
    """
)

st.info(
    "The model predicts one of three outcomes: "
    "**Dropout, Enrolled, or Graduate**."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Student Information")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Column 1
# ------------------------------------------------------------

with col1:

    marital_status = st.number_input(
        "Marital Status",
        min_value=1,
        max_value=6,
        value=1
    )

    application_mode = st.number_input(
        "Application Mode",
        min_value=1,
        max_value=60,
        value=1
    )

    application_order = st.number_input(
        "Application Order",
        min_value=0,
        max_value=9,
        value=1
    )

    course = st.number_input(
        "Course",
        min_value=1,
        max_value=100,
        value=1
    )

    daytime_evening = st.number_input(
        "Daytime/Evening Attendance",
        min_value=0,
        max_value=1,
        value=1
    )

    previous_qualification = st.number_input(
        "Previous Qualification",
        min_value=1,
        max_value=50,
        value=1
    )

    previous_qualification_grade = st.number_input(
        "Previous Qualification Grade",
        min_value=0.0,
        max_value=200.0,
        value=120.0
    )

    nationality = st.number_input(
        "Nationality",
        min_value=1,
        max_value=50,
        value=1
    )

    mothers_qualification = st.number_input(
        "Mother's Qualification",
        min_value=1,
        max_value=50,
        value=1
    )

    fathers_qualification = st.number_input(
        "Father's Qualification",
        min_value=1,
        max_value=50,
        value=1
    )

    mothers_occupation = st.number_input(
        "Mother's Occupation",
        min_value=0,
        max_value=200,
        value=0
    )

    fathers_occupation = st.number_input(
        "Father's Occupation",
        min_value=0,
        max_value=200,
        value=0
    )


# ------------------------------------------------------------
# Column 2
# ------------------------------------------------------------

with col2:

    admission_grade = st.number_input(
        "Admission Grade",
        min_value=0.0,
        max_value=200.0,
        value=120.0
    )

    displaced = st.selectbox(
        "Displaced",
        [0, 1]
    )

    educational_special_needs = st.selectbox(
        "Educational Special Needs",
        [0, 1]
    )

    debtor = st.selectbox(
        "Debtor",
        [0, 1]
    )

    tuition_fees = st.selectbox(
        "Tuition Fees Up to Date",
        [0, 1],
        index=1
    )

    gender = st.selectbox(
        "Gender",
        [0, 1]
    )

    scholarship_holder = st.selectbox(
        "Scholarship Holder",
        [0, 1]
    )

    age = st.number_input(
        "Age at Enrollment",
        min_value=15,
        max_value=80,
        value=20
    )

    international = st.selectbox(
        "International",
        [0, 1]
    )

    unemployment_rate = st.number_input(
        "Unemployment Rate",
        min_value=-10.0,
        max_value=30.0,
        value=10.0
    )

    inflation_rate = st.number_input(
        "Inflation Rate",
        min_value=-10.0,
        max_value=30.0,
        value=2.0
    )

    gdp = st.number_input(
        "GDP",
        min_value=-10.0,
        max_value=10.0,
        value=1.0
    )


# ------------------------------------------------------------
# Column 3
# ------------------------------------------------------------

with col3:

    st.subheader("1st Semester")

    sem1_credited = st.number_input(
        "1st Sem - Credited",
        min_value=0,
        max_value=30,
        value=0
    )

    sem1_enrolled = st.number_input(
        "1st Sem - Enrolled",
        min_value=0,
        max_value=30,
        value=6
    )

    sem1_evaluations = st.number_input(
        "1st Sem - Evaluations",
        min_value=0,
        max_value=30,
        value=6
    )

    sem1_approved = st.number_input(
        "1st Sem - Approved",
        min_value=0,
        max_value=30,
        value=5
    )

    sem1_grade = st.number_input(
        "1st Sem - Grade",
        min_value=0.0,
        max_value=20.0,
        value=12.0
    )

    sem1_without_eval = st.number_input(
        "1st Sem - Without Evaluations",
        min_value=0,
        max_value=30,
        value=0
    )

    st.subheader("2nd Semester")

    sem2_credited = st.number_input(
        "2nd Sem - Credited",
        min_value=0,
        max_value=30,
        value=0
    )

    sem2_enrolled = st.number_input(
        "2nd Sem - Enrolled",
        min_value=0,
        max_value=30,
        value=6
    )

    sem2_evaluations = st.number_input(
        "2nd Sem - Evaluations",
        min_value=0,
        max_value=30,
        value=6
    )

    sem2_approved = st.number_input(
        "2nd Sem - Approved",
        min_value=0,
        max_value=30,
        value=5
    )

    sem2_grade = st.number_input(
        "2nd Sem - Grade",
        min_value=0.0,
        max_value=20.0,
        value=12.0
    )

    sem2_without_eval = st.number_input(
        "2nd Sem - Without Evaluations",
        min_value=0,
        max_value=30,
        value=0
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([{

    "Marital status": marital_status,
    "Application mode": application_mode,
    "Application order": application_order,
    "Course": course,
    "Daytime/evening attendance": daytime_evening,
    "Previous qualification": previous_qualification,
    "Previous qualification (grade)": previous_qualification_grade,
    "Nacionality": nationality,
    "Mother's qualification": mothers_qualification,
    "Father's qualification": fathers_qualification,
    "Mother's occupation": mothers_occupation,
    "Father's occupation": fathers_occupation,
    "Admission grade": admission_grade,
    "Displaced": displaced,
    "Educational special needs": educational_special_needs,
    "Debtor": debtor,
    "Tuition fees up to date": tuition_fees,
    "Gender": gender,
    "Scholarship holder": scholarship_holder,
    "Age at enrollment": age,
    "International": international,

    "Curricular units 1st sem (credited)": sem1_credited,
    "Curricular units 1st sem (enrolled)": sem1_enrolled,
    "Curricular units 1st sem (evaluations)": sem1_evaluations,
    "Curricular units 1st sem (approved)": sem1_approved,
    "Curricular units 1st sem (grade)": sem1_grade,
    "Curricular units 1st sem (without evaluations)": sem1_without_eval,

    "Curricular units 2nd sem (credited)": sem2_credited,
    "Curricular units 2nd sem (enrolled)": sem2_enrolled,
    "Curricular units 2nd sem (evaluations)": sem2_evaluations,
    "Curricular units 2nd sem (approved)": sem2_approved,
    "Curricular units 2nd sem (grade)": sem2_grade,
    "Curricular units 2nd sem (without evaluations)": sem2_without_eval,

    "Unemployment rate": unemployment_rate,
    "Inflation rate": inflation_rate,
    "GDP": gdp
}])


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button("🔮 Predict Student Outcome", width="stretch"):

    prediction = model.predict(input_data)[0]

    # Probability if supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        probability_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities
        })

    st.subheader("Prediction Result")

    if prediction == "Graduate":
        st.success("🎓 Predicted Outcome: GRADUATE")

    elif prediction == "Dropout":
        st.error("⚠️ Predicted Outcome: DROPOUT")

    else:
        st.warning("📚 Predicted Outcome: ENROLLED")

    if hasattr(model, "predict_proba"):

        st.subheader("Prediction Probabilities")

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_df,
            width="stretch"
        )