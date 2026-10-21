
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="centered"
)

# Load the saved model and its input configuration
artifact = joblib.load("stroke_model.joblib")
model = artifact["model"]
feature_columns = artifact["feature_columns"]
threshold = artifact["threshold"]

st.title(" Brain Stroke Prediction System")

st.write(
    "An educational machine-learning project that compares "
    "classification methods and demonstrates a trained model."
)

st.info(
    "This application is for academic demonstration only. "
    "It does not diagnose stroke or provide medical advice."
)

st.subheader("Enter Patient Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 0, 100, 45)
        hypertension = st.selectbox(
            "Hypertension", ["No", "Yes"]
        )
        heart_disease = st.selectbox(
            "Heart Disease", ["No", "Yes"]
        )
        ever_married = st.selectbox(
            "Ever Married", ["No", "Yes"]
        )

    with col2:
        work_type = st.selectbox(
            "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ]
        )
        residence_type = st.selectbox(
            "Residence Type", ["Urban", "Rural"]
        )
        smoking_status = st.selectbox(
            "Smoking Status",
            [
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ]
        )
        avg_glucose_level = st.number_input(
            "Average Glucose Level",
            min_value=0.0,
            max_value=400.0,
            value=100.0
        )
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

    submitted = st.form_submit_button(
        "Predict Stroke Class",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "gender": 1 if gender == "Female" else 0,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": 1 if ever_married == "Yes" else 0,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    encoded_data = pd.get_dummies(
        input_data,
        columns=[
            "work_type",
            "Residence_type",
            "smoking_status"
        ]
    )

    encoded_data = encoded_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    probability = model.predict_proba(
        encoded_data
    )[0][1]

    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Model Output")

    st.metric(
        "Estimated model probability for class 1",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.warning("Predicted class: Stroke (1)")
    else:
        st.success("Predicted class: No Stroke (0)")

    st.caption(
        f"Experimental classification threshold: {threshold}. "
        "The displayed probability is not a validated estimate "
        "of an individual's clinical risk."
    )
