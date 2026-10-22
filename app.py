from datetime import datetime
from pathlib import Path

import altair as alt
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

BASE_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [BASE_DIR / "assets", BASE_DIR, BASE_DIR / "assets" / "assets"]

TEAL, CYAN, INDIGO, AMBER, ROSE, SLATE = (
    "#0f766e", "#0e7490", "#4338ca", "#f59e0b", "#be185d", "#64748b",
)
PALETTE = [TEAL, CYAN, INDIGO, AMBER, ROSE, SLATE]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def find_asset(filename):
    """Find a results file in assets/, next to app.py, or in assets/assets/."""
    for folder in SEARCH_DIRS:
        if (folder / filename).exists():
            return folder / filename
    stem, ext = Path(filename).stem, Path(filename).suffix
    for folder in SEARCH_DIRS:  # tolerate renames like "name (1).csv"
        if folder.exists():
            matches = sorted(folder.glob(f"{stem}*{ext}"))
            if matches:
                return matches[0]
    return None


def read_csv_if_exists(filename):
    path = find_asset(filename)
    return pd.read_csv(path) if path else None


def wide(fn, *args, **kwargs):
    """Full-width widget that works on older and newer Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def show_chart(chart):
    """Render an Altair chart full-width on any Streamlit version."""
    chart = (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFontSize=12, titleFontSize=12, gridColor="#e3edf1",
            domainColor="#c3d3da", labelColor="#33505c", titleColor="#33505c",
        )
        .configure_legend(
            labelFontSize=12, titleFontSize=12, orient="bottom",
            labelColor="#33505c", titleColor="#33505c",
        )
    )
    for kwargs in (
        {"width": "stretch", "theme": None},
        {"use_container_width": True, "theme": None},
        {"use_container_width": True},
    ):
        try:
            return st.altair_chart(chart, **kwargs)
        except Exception:
            continue
    return st.altair_chart(chart)


def show_image_if_exists(filename, caption):
    path = find_asset(filename)
    if path:
        wide(st.image, str(path), caption=caption)
        return True
    return False


# ------------------------------------------------------------------
# Styling (UI only - no effect on the model or predictions)
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');

    html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
    #MainMenu, footer { visibility: hidden; }

    .stApp {
        background:
            radial-gradient(900px 420px at 8% -5%, rgba(45,212,191,0.20), transparent 60%),
            radial-gradient(800px 400px at 100% 0%, rgba(99,102,241,0.16), transparent 60%),
            #f5f9fb;
    }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 900px; }

    .hero {
        background: linear-gradient(120deg, #0f766e 0%, #0e7490 48%, #4338ca 100%);
        border-radius: 20px; padding: 2.2rem 2.4rem 1.9rem 2.4rem;
        margin-bottom: 1.1rem; box-shadow: 0 14px 34px rgba(15,118,110,0.28);
        position: relative; overflow: hidden;
    }
    .hero::after { content: "🧠"; position: absolute; right: 1.6rem; top: 0.6rem;
                   font-size: 6.5rem; opacity: 0.16; }
    .hero h1 { font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.3rem;
               line-height: 1.15; color: #ffffff; margin: 0; padding: 0; }
    .hero p { color: #d9f3f1; font-size: 1.02rem; line-height: 1.55;
              margin: 0.6rem 0 1rem 0; max-width: 58ch; }
    .pill { display: inline-block; margin: 0 0.4rem 0.3rem 0; padding: 0.28rem 0.8rem;
            border-radius: 999px; font-size: 0.82rem; font-weight: 500; color: #ffffff;
            background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35); }

    h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: #12303a; }

    .stTabs [data-baseweb="tab-list"] { gap: 0.2rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; padding: 0.5rem 0.8rem; }

    [data-testid="stForm"] { background: #ffffff; border: 1px solid rgba(15,118,110,0.14);
        border-radius: 18px; padding: 1.7rem 1.7rem 1.3rem 1.7rem;
        box-shadow: 0 8px 26px rgba(18,48,58,0.08); }
    .group-title { font-weight: 700; font-size: 1rem; color: #0f766e;
        margin: 0 0 0.7rem 0; padding-left: 0.6rem; border-left: 4px solid #2dd4bf; }

    div.stButton > button { width: 100%; border-radius: 10px; font-weight: 600;
        border: 1px solid rgba(15,118,110,0.30); background: #ffffff; color: #0f766e;
        transition: all 0.15s ease; }
    div.stButton > button:hover { background: #e7f6f4; border-color: #0f766e; color: #0b5f59; }
    [data-testid="stDownloadButton"] > button { border-radius: 10px; font-weight: 600;
        border: 1px solid rgba(15,118,110,0.30); color: #0f766e; background: #ffffff; }
    [data-testid="stFormSubmitButton"] > button { width: 100%;
        background: linear-gradient(90deg, #0f766e, #0e7490); color: #ffffff; border: none;
        border-radius: 12px; padding: 0.8rem 1rem; font-weight: 700; font-size: 1.05rem;
        box-shadow: 0 8px 20px rgba(15,118,110,0.30);
        transition: transform 0.12s ease, box-shadow 0.12s ease; }
    [data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-1px);
        color: #ffffff; box-shadow: 0 12px 26px rgba(15,118,110,0.38); }
    [data-testid="stFormSubmitButton"] > button:focus-visible { outline: 3px solid #2dd4bf; outline-offset: 2px; }

    .result-card { border-radius: 16px; padding: 1.4rem 1.6rem; margin: 0.6rem 0 1rem 0;
        border: 1px solid; color: #12303a; box-shadow: 0 8px 22px rgba(18,48,58,0.07); }
    .result-card .label { font-family: 'Fraunces', serif; font-size: 1.55rem; font-weight: 600; margin: 0; }
    .result-card .sub { margin: 0.35rem 0 0 0; opacity: 0.85; font-size: 0.96rem; }
    .result-neg { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-color: #10b981; }
    .result-pos { background: linear-gradient(135deg, #fffbeb, #fde68a); border-color: #f59e0b; }

    .gauge { position: relative; height: 16px; border-radius: 999px; background: #dbe7ec; margin: 0.4rem 0 0.3rem 0; }
    .gauge-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2dd4bf, #f59e0b); }
    .gauge-marker { position: absolute; top: -5px; width: 3px; height: 26px; background: #12303a; border-radius: 2px; }
    .gauge-legend { display: flex; justify-content: space-between; font-size: 0.8rem; color: #4b6772; margin-bottom: 1rem; }

    [data-testid="stMetric"] { background: #ffffff; border: 1px solid rgba(15,118,110,0.14);
        border-radius: 14px; padding: 0.9rem 1.1rem; box-shadow: 0 6px 18px rgba(18,48,58,0.06); }

    .steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.8rem; margin: 0.6rem 0 1.2rem 0; }
    .step { background: #ffffff; border: 1px solid rgba(15,118,110,0.14); border-radius: 14px;
            padding: 0.9rem 1.1rem; box-shadow: 0 6px 18px rgba(18,48,58,0.06); color: #12303a; }
    .step .num { display: inline-block; width: 1.6rem; height: 1.6rem; line-height: 1.6rem; text-align: center;
                 border-radius: 50%; background: #0f766e; color: #fff; font-weight: 700; font-size: 0.85rem; margin-right: 0.5rem; }
    .step b { font-size: 1rem; }
    .step p { margin: 0.4rem 0 0 0; font-size: 0.92rem; color: #3b5662; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f3d46 0%, #12303a 100%); }
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e3f3f4 !important; }
    [data-testid="stSidebar"] code { background: rgba(255,255,255,0.12); color: #a7f3e6 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.18); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Load the saved model and its input configuration
# ------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    return joblib.load(BASE_DIR / "stroke_model.joblib")


artifact = load_artifact()
model = artifact["model"]
feature_columns = artifact["feature_columns"]
threshold = artifact["threshold"]


def final_estimator(m):
    return m.steps[-1][1] if hasattr(m, "steps") else m


model_name = type(final_estimator(model)).__name__


def get_importances(m, columns):
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
# Same preprocessing as the original app, wrapped in two small functions
# ------------------------------------------------------------------
def raw_row(v):
    return {
        "gender": 1 if v["gender"] == "Female" else 0,
        "age": v["age"],
        "hypertension": 1 if v["hypertension"] == "Yes" else 0,
        "heart_disease": 1 if v["heart_disease"] == "Yes" else 0,
        "ever_married": 1 if v["ever_married"] == "Yes" else 0,
        "work_type": v["work_type"],
        "Residence_type": v["residence_type"],
        "avg_glucose_level": v["avg_glucose_level"],
        "bmi": v["bmi"],
        "smoking_status": v["smoking_status"],
    }


def encode(rows):
    frame = pd.DataFrame(rows)
    frame = pd.get_dummies(
        frame, columns=["work_type", "Residence_type", "smoking_status"]
    )
    return frame.reindex(columns=feature_columns, fill_value=0)


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
st.session_state.setdefault("history", [])


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
        """
    )
    st.divider()
    st.caption("Educational use only. Not a clinically validated diagnostic system.")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>Brain Stroke Prediction System</h1>
        <p>An educational machine-learning project that compares
        classification methods and demonstrates a trained model.</p>
        <span class="pill">Academic demo</span>
        <span class="pill">Model: {model_name}</span>
        <span class="pill">Threshold: {threshold}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "This application is for academic demonstration only. "
    "It does not diagnose stroke or provide medical advice."
)

tab_predict, tab_overview, tab_compare, tab_threshold, tab_explain, tab_eda = st.tabs(
    ["Predict", "Overview", "Compare", "Thresholds", "Explain", "EDA"]
)

# ==================================================================
# PREDICT (form, result, what-if analysis, download, history)
# ==================================================================
with tab_predict:
    st.subheader("Enter patient details")
    st.caption("Quick fill with an illustrative example profile (made-up inputs, not real patients).")
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, PRESETS):
        col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,))

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
            residence_type = st.selectbox("Residence type", ["Urban", "Rural"], key="residence_type")
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
                "Average glucose level (mg/dL)", min_value=0.0, max_value=400.0,
                step=1.0, key="avg_glucose_level",
            )
            bmi = st.number_input(
                "BMI (kg/m²)", min_value=0.0, max_value=100.0, step=0.1, key="bmi",
            )
        submitted = st.form_submit_button("Predict stroke class")

    if submitted:
        vals = {
            "gender": gender, "age": age, "hypertension": hypertension,
            "heart_disease": heart_disease, "ever_married": ever_married,
            "work_type": work_type, "residence_type": residence_type,
            "smoking_status": smoking_status,
            "avg_glucose_level": avg_glucose_level, "bmi": bmi,
        }
        probability = float(model.predict_proba(encode([raw_row(vals)]))[0][1])
        prediction = int(probability >= threshold)
        st.session_state["last_result"] = {
            "vals": vals, "probability": probability, "prediction": prediction,
        }
        st.session_state["history"].append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Age": age, "Gender": gender, "Hypertension": hypertension,
            "Heart disease": heart_disease, "Glucose": avg_glucose_level,
            "BMI": bmi, "Probability": f"{probability:.1%}",
            "Class": "Stroke (1)" if prediction == 1 else "No Stroke (0)",
        })
        st.session_state["history"] = st.session_state["history"][-20:]

    last = st.session_state.get("last_result")
    if last:
        vals = last["vals"]
        probability = last["probability"]
        prediction = last["prediction"]

        st.divider()
        st.subheader("Model output")
        if prediction == 1:
            st.markdown(
                f"""<div class="result-card result-pos">
                <p class="label">Predicted class: Stroke (1)</p>
                <p class="sub">The model probability of {probability:.1%} is at or above the threshold of {threshold}.</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="result-card result-neg">
                <p class="label">Predicted class: No Stroke (0)</p>
                <p class="sub">The model probability of {probability:.1%} is below the threshold of {threshold}.</p>
                </div>""",
                unsafe_allow_html=True,
            )

        m1, m2 = st.columns(2)
        m1.metric("Estimated model probability for class 1", f"{probability:.1%}")
        m2.metric("Classification threshold", f"{threshold}")

        fill = min(max(probability, 0.0), 1.0) * 100
        marker = min(max(float(threshold), 0.0), 1.0) * 100
        st.markdown(
            f"""
            <div class="gauge">
                <div class="gauge-fill" style="width:{fill}%"></div>
                <div class="gauge-marker" style="left:{marker}%"></div>
            </div>
            <div class="gauge-legend"><span>0%</span>
            <span>Dark marker = threshold ({threshold})</span><span>100%</span></div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- What-if analysis ----------
        st.subheader("What-if analysis")
        st.caption(
            "Change one input while keeping the others fixed and watch how the "
            "model's probability responds. Computed live from the saved model."
        )
        sweeps = {
            "Age (years)": ("age", np.arange(0, 101, 1)),
            "Average glucose level (mg/dL)": ("avg_glucose_level", np.arange(50.0, 300.1, 5.0)),
            "BMI (kg/m²)": ("bmi", np.arange(10.0, 60.1, 0.5)),
        }
        chosen_label = st.selectbox("Feature to vary", list(sweeps), key="whatif_feature")
        feat_key, grid = sweeps[chosen_label]
        rows = []
        for g in grid:
            row = dict(vals)
            row[feat_key] = float(g) if feat_key != "age" else int(g)
            rows.append(raw_row(row))
        probs = model.predict_proba(encode(rows))[:, 1]
        sweep_df = pd.DataFrame({"x": grid.astype(float), "Probability": probs})
        current_df = pd.DataFrame(
            {"x": [float(vals[feat_key])], "Probability": [probability]}
        )
        line = alt.Chart(sweep_df).mark_line(color=TEAL, strokeWidth=3).encode(
            x=alt.X("x:Q", title=chosen_label),
            y=alt.Y("Probability:Q", title="Model probability for class 1",
                    scale=alt.Scale(domain=[0, max(0.05, float(probs.max()) * 1.15)]),
                    axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("x:Q", title=chosen_label),
                     alt.Tooltip("Probability:Q", format=".1%")],
        )
        rule = alt.Chart(pd.DataFrame({"t": [float(threshold)]})).mark_rule(
            color=AMBER, strokeDash=[6, 4], strokeWidth=2
        ).encode(y="t:Q")
        point = alt.Chart(current_df).mark_point(
            color=INDIGO, size=170, filled=True
        ).encode(x="x:Q", y="Probability:Q")
        show_chart((line + rule + point).properties(height=280))
        st.caption(
            "Teal line = model probability. Dashed amber line = classification "
            "threshold. Indigo dot = the values you submitted. This shows how the "
            "model behaves, not what would happen to a real person."
        )

        # ---------- Download + inputs ----------
        report = pd.DataFrame([{
            **vals,
            "probability_class_1": round(probability, 4),
            "threshold": threshold,
            "predicted_class": prediction,
        }])
        st.download_button(
            "Download this result (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="stroke_prediction_result.csv",
            mime="text/csv",
        )
        with st.expander("View submitted inputs"):
            wide(
                st.dataframe,
                pd.DataFrame({
                    "Feature": ["Gender", "Age", "Hypertension", "Heart disease",
                                "Ever married", "Work type", "Residence type",
                                "Average glucose level", "BMI", "Smoking status"],
                    "Value": [vals["gender"], str(vals["age"]), vals["hypertension"],
                              vals["heart_disease"], vals["ever_married"],
                              vals["work_type"], vals["residence_type"],
                              str(vals["avg_glucose_level"]), str(vals["bmi"]),
                              vals["smoking_status"]],
                }),
                hide_index=True,
            )
        st.caption(
            f"Experimental classification threshold: {threshold}. "
            "The displayed probability is not a validated estimate "
            "of an individual's clinical risk."
        )

    if st.session_state["history"]:
        with st.expander(f"Prediction history this session ({len(st.session_state['history'])})"):
            wide(st.dataframe, pd.DataFrame(st.session_state["history"][::-1]), hide_index=True)

# ==================================================================
# OVERVIEW
# ==================================================================
with tab_overview:
    st.subheader("Project workflow")
    st.markdown(
        """
        <div class="steps">
          <div class="step"><span class="num">1</span><b>Data</b>
            <p>Brain-stroke dataset with 4,981 records and 11 columns.</p></div>
          <div class="step"><span class="num">2</span><b>Exploration and EDA</b>
            <p>Data exploration and exploratory data analysis.</p></div>
          <div class="step"><span class="num">3</span><b>Preprocessing</b>
            <p>Data preprocessing and categorical encoding.</p></div>
          <div class="step"><span class="num">4</span><b>Train-test split</b>
            <p>80:20 split into training and test sets.</p></div>
          <div class="step"><span class="num">5</span><b>Model training</b>
            <p>Logistic Regression, Decision Tree and Random Forest.</p></div>
          <div class="step"><span class="num">6</span><b>Validation and tuning</b>
            <p>Cross-validation and hyperparameter tuning.</p></div>
          <div class="step"><span class="num">7</span><b>Evaluation</b>
            <p>Confusion matrix, precision, recall, F1-score, accuracy and ROC-AUC.</p></div>
          <div class="step"><span class="num">8</span><b>Threshold tuning</b>
            <p>Random Forest threshold study: recall vs false positives.</p></div>
          <div class="step"><span class="num">9</span><b>Deployment</b>
            <p>This Streamlit app, loading the saved model file.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Limitations")
    st.markdown(
        """
        - The model was trained on a single public dataset of 4,981 records.
        - The output is a class assigned by a threshold, not a diagnosis.
        - The probability shown has not been clinically validated.
        - This project is for learning and demonstration only.
        """
    )

# ==================================================================
# COMPARE (needs assets/metrics_comparison.csv)
# ==================================================================
with tab_compare:
    st.subheader("Model comparison")
    metrics_df = read_csv_if_exists("metrics_comparison.csv")
    if metrics_df is None:
        st.info("metrics_comparison.csv not found. Put it in an `assets` folder next to app.py.")
    else:
        metric_cols = [c for c in metrics_df.columns if c != "Model"]
        model_order = list(metrics_df["Model"])

        tile_cols = st.columns(len(metric_cols))
        for col, m in zip(tile_cols, metric_cols):
            best_row = metrics_df.loc[metrics_df[m].idxmax()]
            col.metric(f"Best {m}", f"{best_row[m]:.3f}")
            col.caption(str(best_row["Model"]))

        long_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
        base = alt.Chart(long_df).encode(
            x=alt.X("Metric:N", sort=metric_cols, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Model:N", sort=model_order, title=None),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("Value:Q", legend=None,
                            scale=alt.Scale(domain=[0, 1], range=["#e8f5f3", TEAL])),
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Value:Q", format=".3f")],
        )
        labels = base.mark_text(fontSize=13, fontWeight="bold").encode(
            text=alt.Text("Value:Q", format=".3f"),
            color=alt.condition(alt.datum.Value > 0.6, alt.value("white"), alt.value("#12303a")),
        )
        st.markdown("**All metrics at a glance**")
        show_chart((heat + labels).properties(height=max(120, 55 * len(model_order))))

        chosen = st.selectbox("Compare one metric", metric_cols, index=len(metric_cols) - 1)
        one = long_df[long_df["Metric"] == chosen]
        bar = alt.Chart(one).mark_bar(size=26).encode(
            y=alt.Y("Model:N", sort="-x", title=None),
            x=alt.X("Value:Q", title=chosen, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=None,
                            scale=alt.Scale(domain=model_order, range=PALETTE[: len(model_order)])),
            tooltip=["Model:N", alt.Tooltip("Value:Q", format=".3f")],
        )
        show_chart(bar.properties(height=max(120, 48 * len(model_order))))
        with st.expander("View table"):
            wide(st.dataframe, metrics_df, hide_index=True)

# ==================================================================
# THRESHOLDS (needs assets/threshold_results.csv)
# ==================================================================
with tab_threshold:
    st.subheader("Threshold tuning")
    thr_df = read_csv_if_exists("threshold_results.csv")
    if thr_df is None:
        st.info("threshold_results.csv not found. Put it in an `assets` folder next to app.py.")
    else:
        st.write(
            "Lowering the threshold catches more stroke-class records (higher recall) "
            "but raises false positives."
        )
        plot_cols = [c for c in ["Precision", "Recall", "F1"] if c in thr_df.columns]
        color_map = {"Precision": INDIGO, "Recall": TEAL, "F1": AMBER}
        tl = thr_df.melt(id_vars="Threshold", value_vars=plot_cols,
                         var_name="Metric", value_name="Value")
        lines = alt.Chart(tl).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("Threshold:Q"),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", scale=alt.Scale(
                domain=plot_cols, range=[color_map[c] for c in plot_cols])),
            tooltip=["Metric:N", alt.Tooltip("Threshold:Q", format=".2f"),
                     alt.Tooltip("Value:Q", format=".3f")],
        )
        rule = alt.Chart(pd.DataFrame({"Threshold": [float(threshold)]})).mark_rule(
            color=SLATE, strokeDash=[6, 4], strokeWidth=2
        ).encode(x="Threshold:Q")
        st.markdown("**Precision, recall and F1 by threshold**")
        show_chart((lines + rule).properties(height=300))

        if "False Positives" in thr_df.columns and "Recall" in thr_df.columns:
            tdf = thr_df.rename(columns={"False Positives": "FalsePositives"})
            base = alt.Chart(tdf).encode(x=alt.X("Threshold:Q"))
            bars = base.mark_bar(color=AMBER, opacity=0.5, size=16).encode(
                y=alt.Y("FalsePositives:Q", title="False positives (bars)"),
                tooltip=[alt.Tooltip("Threshold:Q", format=".2f"), "FalsePositives:Q"],
            )
            recall_line = base.mark_line(color=TEAL, strokeWidth=3, point=True).encode(
                y=alt.Y("Recall:Q", title="Recall (line)",
                        axis=alt.Axis(orient="right"), scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("Threshold:Q", format=".2f"),
                         alt.Tooltip("Recall:Q", format=".3f")],
            )
            st.markdown("**The trade-off: recall vs false positives**")
            show_chart(alt.layer(bars, recall_line).resolve_scale(y="independent").properties(height=300))

        nearest = thr_df.iloc[(thr_df["Threshold"] - float(threshold)).abs().argmin()]
        st.caption(
            f"Dashed grey line = threshold used by this app ({threshold}). "
            f"Closest row in the table: {nearest['Threshold']:.2f}."
        )
        with st.expander("View full table"):
            wide(st.dataframe, thr_df, hide_index=True)

# ==================================================================
# EXPLAIN (computed live from the saved model)
# ==================================================================
with tab_explain:
    st.subheader("What the model relies on")
    importances, imp_label = get_importances(model, feature_columns)
    if importances is None:
        st.info(f"The saved model (`{model_name}`) does not expose importances or coefficients.")
    else:
        if len(importances) > 3:
            top_n = st.slider("Number of features to show", 3, len(importances),
                              min(10, len(importances)))
        else:
            top_n = len(importances)
        imp_df = importances.head(top_n).reset_index()
        imp_df.columns = ["Feature", "Value"]
        imp_chart = alt.Chart(imp_df).mark_bar(color=TEAL).encode(
            x=alt.X("Value:Q", title=imp_label),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature:N", alt.Tooltip("Value:Q", format=".4f")],
        )
        show_chart(imp_chart.properties(height=max(160, 32 * top_n)))
        st.caption(
            f"{imp_label} from the saved `{model_name}`. This shows which inputs "
            "the model uses most, not whether they cause stroke."
        )

# ==================================================================
# EDA (needs assets/*.png)
# ==================================================================
with tab_eda:
    st.subheader("EDA and evaluation")
    images = [
        ("class_distribution.png", "Class distribution"),
        ("age_by_stroke.png", "Age distribution by class"),
        ("confusion_matrices.png", "Confusion matrices"),
        ("roc_curves.png", "ROC curves"),
    ]
    found_any = False
    for filename, caption in images:
        found_any = show_image_if_exists(filename, caption) or found_any
    if not found_any:
        st.info("No images found. Put them in an `assets` folder next to app.py.")