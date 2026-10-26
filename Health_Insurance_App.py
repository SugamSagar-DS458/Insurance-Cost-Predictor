from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "Health_Insurance.csv"
FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1e3c72 0%, #2a9d8f 100%);
        padding: 2rem 1.5rem; border-radius: 16px; color: white;
        text-align: center; margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p  { margin: .4rem 0 0; opacity: .9; }
    .result {
        background: #f0f9f7; border: 2px solid #2a9d8f; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin-top: 1rem;
    }
    .result .label  { color: #555; font-size: 1rem; }
    .result .amount { color: #1e3c72; font-size: 2.8rem; font-weight: 700; }
    div.stButton > button {
        width: 100%; background: #2a9d8f; color: white; font-weight: 600;
        border-radius: 10px; padding: .7rem; border: none;
    }
    div.stButton > button:hover { background: #1e3c72; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏥 Insurance Cost Predictor</h1>
        <p>Estimate annual medical insurance charges from a few personal details</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Load model ----------
@st.cache_resource(show_spinner="Training the prediction model...")
def load_artifacts():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Keep Health_Insurance.csv beside this app."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = set(FEATURE_COLUMNS + ["charges"])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if data.empty or data[list(required_columns)].isna().any().any():
        raise ValueError("Dataset is empty or contains missing feature or charge values.")

    encoded = pd.get_dummies(
        data[FEATURE_COLUMNS],
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int,
    )
    scaler = StandardScaler()
    encoded[["age", "bmi"]] = scaler.fit_transform(encoded[["age", "bmi"]])

    model = GradientBoostingRegressor(
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        n_estimators=150,
        random_state=42,
    )
    model.fit(encoded, np.log1p(data["charges"]))
    return model, scaler, encoded.columns.tolist()


try:
    model, scaler, columns = load_artifacts()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


# ---------- Inputs ----------
st.subheader("Your details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 64, 30)
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)

with col2:
    height_cm = st.number_input("Height (cm)", 120.0, 220.0, 170.0, step=1.0)
    weight_kg = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, step=1.0)
    children = st.selectbox("Number of children", [0, 1, 2, 3, 4, 5])

region = st.selectbox("Region (US dataset)", ["northeast", "northwest", "southeast", "southwest"])

bmi = weight_kg / ((height_cm / 100) ** 2)
st.info(f"**BMI: {bmi:.1f}** — {bmi_category(bmi)}")

# ---------- Prediction ----------
if st.button("Predict insurance charge"):
    raw_row = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    row = pd.get_dummies(
        raw_row,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int,
    )
    row = row.reindex(columns=columns, fill_value=0)
    row[["age", "bmi"]] = scaler.transform(row[["age", "bmi"]])

    prediction = max(0.0, float(np.expm1(model.predict(row)[0])))

    st.markdown(
        f"""
        <div class="result">
            <div class="label">Estimated annual insurance charge</div>
            <div class="amount">${prediction:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Rough range using the test MAE (~$2,000) from the notebook
    st.caption(f"Typical error is around ±$2,000 → likely range ${max(prediction - 2000, 0):,.0f} – ${prediction + 2000:,.0f}")

    if smoker == "yes":
        st.warning("Smoking is the single biggest driver of higher charges in this dataset.")
    if bmi >= 30:
        st.warning("A BMI of 30+ is associated with higher predicted charges.")

st.markdown("---")
st.caption("Model: Tuned Gradient Boosting trained on the 1,338-row insurance dataset. For educational use only — not an actual quote.")