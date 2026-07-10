import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.joblib")


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter the passenger information below to predict "
    "whether the passenger is likely to survive."
)

st.divider()


# =========================================================
# USER INPUTS
# =========================================================

col1, col2 = st.columns(2)


with col1:

    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        help="1 = First Class, 2 = Second Class, 3 = Third Class"
    )

    sex = st.selectbox(
        "Sex",
        options=["female", "male"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    sibsp = st.number_input(
        "Number of Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


with col2:

    parch = st.number_input(
        "Number of Parents / Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["C", "Q", "S"],
        help="C = Cherbourg, Q = Queenstown, S = Southampton"
    )


# =========================================================
# PREDICTION
# =========================================================

st.divider()

if st.button("🔮 Predict Survival", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked": [embarked]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_data)[0]

    survival_probability = probabilities[1] * 100
    death_probability = probabilities[0] * 100


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("### 🚢 Passenger is predicted to SURVIVE")

    else:

        st.error("### ❌ Passenger is predicted NOT to survive")


    # Display probabilities

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Survival Probability",
            f"{survival_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Not Survive Probability",
            f"{death_probability:.2f}%"
        )


    # Show input data

    with st.expander("View Passenger Information"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Python, Scikit-learn, Joblib and Streamlit"
)