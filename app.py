import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Styling (UI only - no effect on the model or predictions)
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');

    html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
    #MainMenu, footer { visibility: hidden; }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 860px; }

    .hero {
        background: #12303a;
        border-radius: 14px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
        border-left: 6px solid #2dd4bf;
    }
    .hero h1 {
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.2rem;
        line-height: 1.15; color: #ffffff; margin: 0; padding: 0;
    }
    .hero p {
        color: #c5dde0; font-size: 1rem; line-height: 1.55;
        margin: 0.6rem 0 0 0; max-width: 60ch;
    }

    h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; }

    .stTabs [data-baseweb="tab"] { font-weight: 600; }

    [data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 14px;
        padding: 1.6rem 1.6rem 1.2rem 1.6rem;
    }
    .group-title {
        font-weight: 700; font-size: 1rem; margin: 0 0 0.6rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid rgba(45, 212, 191, 0.55);
    }

    [data-testid="stFormSubmitButton"] > button {
        background: #0f766e; color: #ffffff; border: none; border-radius: 10px;
        padding: 0.7rem 1rem; font-weight: 700; font-size: 1.02rem;
        transition: background 0.15s ease;
    }
    [data-testid="stFormSubmitButton"] > button:hover { background: #0b5f59; color: #ffffff; }
    [data-testid="stFormSubmitButton"] > button:focus-visible {
        outline: 3px solid #2dd4bf; outline-offset: 2px;
    }

    .result-card { border-radius: 14px; padding: 1.3rem 1.5rem; margin: 0.5rem 0 1rem 0; border: 1px solid; }
    .result-card .label { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600; margin: 0; }
    .result-card .sub { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
    .result-neg { background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.6); }
    .result-pos { background: rgba(245, 158, 11, 0.14); border-color: rgba(245, 158, 11, 0.7); }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px; padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Load the saved model and its input configuration
# ------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    return joblib.load("stroke_model.joblib")


artifact = load_artifact()
model = artifact["model"]
feature_columns = artifact["feature_columns"]
threshold = artifact["threshold"]


def final_estimator(m):
    """Return the last step if the model is a Pipeline, else the model itself."""
    return m.steps[-1][1] if hasattr(m, "steps") else m


model_name = type(final_estimator(model)).__name__


def get_importances(m, columns):
    """Feature importance (tree models) or |coefficient| (linear models)."""
    est = final_estimator(m)
    if hasattr(est, "feature_importances_"):
        values, label = est.feature_importances_, "Feature importance"
    elif hasattr(est, "coef_"):
        values, label = np.abs(np.ravel(est.coef_)), "Absolute coefficient"
    else:
        return None, None
    if len(values) != len(columns):
        return None, None
    series = pd.Series(values, index=[c.replace("_", " ") for c in columns])
    return series.sort_values(ascending=False), label


# ------------------------------------------------------------------
# Form defaults and example profiles (illustrative, not real patients)
# ------------------------------------------------------------------
DEFAULTS = {
    "gender": "Male", "age": 45, "hypertension": "No", "heart_disease": "No",
    "ever_married": "No", "work_type": "Private", "residence_type": "Urban",
    "smoking_status": "never smoked", "avg_glucose_level": 100.0, "bmi": 25.0,
}
PRESETS = {
    "Younger profile": {
        "gender": "Female", "age": 25, "hypertension": "No", "heart_disease": "No",
        "ever_married": "No", "work_type": "Private", "residence_type": "Urban",
        "smoking_status": "never smoked", "avg_glucose_level": 85.0, "bmi": 22.0,
    },
    "Middle-aged profile": {
        "gender": "Female", "age": 55, "hypertension": "Yes", "heart_disease": "No",
        "ever_married": "Yes", "work_type": "Govt_job", "residence_type": "Urban",
        "smoking_status": "smokes", "avg_glucose_level": 140.0, "bmi": 29.5,
    },
    "Older profile": {
        "gender": "Male", "age": 72, "hypertension": "Yes", "heart_disease": "Yes",
        "ever_married": "Yes", "work_type": "Self-employed", "residence_type": "Rural",
        "smoking_status": "formerly smoked", "avg_glucose_level": 210.0, "bmi": 31.0,
    },
    "Reset": DEFAULTS,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_preset(name):
    st.session_state.update(PRESETS[name])


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "Machine Learning lab mini-project: predicting whether a patient "
        "record falls in the stroke or no-stroke class."
    )
    st.markdown(
        f"""
        **Dataset**
        - 4,981 records, 11 columns
        - 80:20 train-test split

        **Saved model in this app**
        - `{model_name}`
        - Threshold: {threshold}

        **Evaluation in the notebook**
        - Confusion matrix
        - Precision, recall, F1-score
        - Accuracy and ROC-AUC
        """
    )
    st.divider()
    st.caption(
        "Educational use only. This is not a clinically validated "
        "diagnostic system."
    )

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Brain Stroke Prediction System</h1>
        <p>An educational machine-learning project that compares
        classification methods and demonstrates a trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This application is for academic demonstration only. "
    "It does not diagnose stroke or provide medical advice."
)

tab_predict, tab_explain = st.tabs(["Predict", "Explainability"])

# ==================================================================
# TAB 1: Predict
# ==================================================================
with tab_predict:
    st.subheader("Enter patient details")

    st.caption(
        "Quick fill with an illustrative example profile. These are made-up "
        "inputs for demos, not real patients."
    )
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, PRESETS):
        col.button(
            name,
            key=f"preset_{name}",
            on_click=apply_preset,
            args=(name,),
            use_container_width=True,
        )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<p class="group-title">Personal profile</p>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            age = st.slider("Age (years)", 0, 100, key="age")
            ever_married = st.selectbox("Ever married", ["No", "Yes"], key="ever_married")
            work_type = st.selectbox(
                "Work type",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                key="work_type",
            )
            residence_type = st.selectbox(
                "Residence type", ["Urban", "Rural"], key="residence_type"
            )

        with col2:
            st.markdown('<p class="group-title">Health and lifestyle</p>', unsafe_allow_html=True)
            hypertension = st.selectbox("Hypertension", ["No", "Yes"], key="hypertension")
            heart_disease = st.selectbox("Heart disease", ["No", "Yes"], key="heart_disease")
            smoking_status = st.selectbox(
                "Smoking status",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                key="smoking_status",
            )
            avg_glucose_level = st.number_input(
                "Average glucose level (mg/dL)",
                min_value=0.0, max_value=400.0, step=1.0,
                key="avg_glucose_level",
                help="Average blood glucose level, 0 to 400 mg/dL.",
            )
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=0.0, max_value=100.0, step=0.1,
                key="bmi",
                help="Body mass index, 0 to 100.",
            )

        submitted = st.form_submit_button("Predict stroke class", use_container_width=True)

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
            "smoking_status": smoking_status,
        }])

        encoded_data = pd.get_dummies(
            input_data,
            columns=["work_type", "Residence_type", "smoking_status"],
        )
        encoded_data = encoded_data.reindex(columns=feature_columns, fill_value=0)

        probability = model.predict_proba(encoded_data)[0][1]
        prediction = int(probability >= threshold)

        st.divider()
        st.subheader("Model output")

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card result-pos">
                    <p class="label">Predicted class: Stroke (1)</p>
                    <p class="sub">The model probability of {probability:.1%}
                    is at or above the threshold of {threshold}.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-neg">
                    <p class="label">Predicted class: No Stroke (0)</p>
                    <p class="sub">The model probability of {probability:.1%}
                    is below the threshold of {threshold}.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2 = st.columns(2)
        m1.metric("Estimated model probability for class 1", f"{probability:.1%}")
        m2.metric("Classification threshold", f"{threshold}")
        st.progress(float(min(max(probability, 0.0), 1.0)))

        with st.expander("View submitted inputs"):
            st.dataframe(
                pd.DataFrame({
                    "Feature": [
                        "Gender", "Age", "Hypertension", "Heart disease",
                        "Ever married", "Work type", "Residence type",
                        "Average glucose level", "BMI", "Smoking status",
                    ],
                    "Value": [
                        gender, str(age), hypertension, heart_disease,
                        ever_married, work_type, residence_type,
                        str(avg_glucose_level), str(bmi), smoking_status,
                    ],
                }),
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            f"Experimental classification threshold: {threshold}. "
            "The displayed probability is not a validated estimate "
            "of an individual's clinical risk."
        )

# ==================================================================
# TAB 2: Explainability (computed live from the saved model)
# ==================================================================
with tab_explain:
    st.subheader("What the model relies on")
    importances, imp_label = get_importances(model, feature_columns)

    if importances is None:
        st.info(
            f"The saved model (`{model_name}`) does not expose feature "
            "importances or coefficients, so this view is unavailable."
        )
    else:
        top_n = st.slider(
            "Number of features to show", 3, len(importances), min(10, len(importances))
        )
        st.bar_chart(importances.head(top_n))
        st.caption(
            f"{imp_label} from the saved `{model_name}`. This shows which inputs "
            "the model uses most, not whether they cause stroke."
        )