# 🏥 Health Insurance Cost Predictor

An end-to-end Machine Learning web application built with **Streamlit** and **Scikit-Learn** to estimate annual individual health insurance charges based on demographic, lifestyle, and biometric factors.

---

## 📌 Project Overview

Predicting medical insurance charges accurately is crucial for both insurance providers setting risk-adjusted premiums and individuals planning personal healthcare budgets. This project encompasses:
- Comprehensive Exploratory Data Analysis (EDA) and outlier diagnosis.
- Analysis of feature skewness and log transformation of target charges.
- Multi-model evaluation comparing Linear Regression, Decision Trees, Random Forests, SVR, KNN, and Gradient Boosting.
- Overfitting diagnosis through Train vs. Test $R^2$, RMSE, and MAE benchmarks.
- Hyperparameter tuning using `GridSearchCV`.
- An interactive web application displaying cost estimates in both **US Dollars (USD)** and **Indian Rupees (INR)** with an automated BMI calculator.

---

## 🚀 Live Demo & Web App Features

- **Personalized Estimation**: Input applicant details and receive customized cost outputs.
- **Dynamic BMI Calculation**: Computes Body Mass Index in real-time from height (cm) and weight (kg), accompanied by clinical categories (Underweight, Normal, Overweight, Obese).
- **Dual Currency Support**: Displays estimated charges in **USD ($)** and **INR (₹)**.
- **Model Caching**: Uses `@st.cache_resource` to train and cache the model pipeline on startup for low-latency inferences.

---

## 📊 Dataset & Features

The model is trained on the standard `Health_Insurance.csv` dataset comprising 1,338 records and 7 attributes:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Integer | Age of primary beneficiary (18 – 64) |
| `sex` | Categorical | Gender (`female`, `male`) |
| `bmi` | Float | Body Mass Index ($kg/m^2$) |
| `children` | Integer | Number of children / dependents covered |
| `smoker` | Categorical | Smoking status (`yes`, `no`) |
| `region` | Categorical | Beneficiary's residential area in the US (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | Float | **Target Variable**: Individual medical costs billed by health insurance |

---

## 🔬 Modeling & Performance Summary

Because health insurance claims are heavily right-skewed, the target variable `charges` is modeled using a logarithmic transformation (`log1p` / `expm1`).

### Benchmark Comparison

| Model | Train $R^2$ | Test $R^2$ | Test MAE ($) | Test RMSE ($) | Overfitting Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Default)** | 0.9595 | 0.8778 | $2,082.93 | $4,355.89 | Overfitting detected |
| **Gradient Boosting (Default)** | 0.8817 | 0.8757 | $2,027.74 | $4,393.76 | Balanced & Generalizable |
| **Tuned Random Forest** | 0.8886 | 0.8771 | $2,062.70 | $4,367.25 | Resolved overfitting |
| **Tuned Gradient Boosting** | 0.8678 | **0.8752** | **$1,978.70** | **$4,401.81** | **Best Generalization** |
| **Decision Tree** | 0.8751 | 0.8468 | $2,461.24 | $4,876.23 | Moderate fit |
| **Support Vector Regressor (SVR)** | 0.8381 | 0.8188 | $2,319.59 | $5,303.80 | Baseline fit |
| **Linear Regression** | 0.5123 | 0.6067 | $3,888.44 | $7,814.06 | High bias (Underfit) |
| **K-Nearest Neighbors (KNN)** | 0.6968 | 0.5760 | $4,210.29 | $8,113.13 | Overfitting detected |

> **Selected Production Model**: **Tuned Gradient Boosting Regressor** (`n_estimators=150`, `learning_rate=0.03`, `max_depth=3`, `min_samples_leaf=5`) integrated inside a Scikit-Learn `Pipeline` with `StandardScaler` and `OneHotEncoder`.

---

## 📁 Repository Structure

```text
├── Health_Insurance.csv          # Insurance dataset
├── Health_Insurance_App.py       # Streamlit application UI & inference logic
├── medicalinsurance.ipynb        # EDA, modeling experiments & tuning notebook
├── requirements.txt              # Production dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run Health_Insurance_App.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Cloud)

1. Push your repository to GitHub (ensure repository visibility is set to **Public**).
2. Visit [Streamlit Community Cloud](https://insurance-cost-predictor-sugamsagar-ds458.streamlit.app/) and log in with GitHub.
3. Click **New app** and configure:
   - **Repository**: `<your-username>/<your-repo-name>`
   - **Branch**: `main`
   - **Main file path**: `Health_Insurance_App.py`
4. Click **Deploy!**.

---

## 📜 License & Disclaimers

This project is open-source and intended for educational and portfolio demonstration purposes. Estimated insurance charges are generated by a statistical machine learning model and do not constitute actual medical underwriting quotes.
