import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------- Load model & preprocessor -----------------
MODEL_PATH = "models/health_model.pkl"
PREPROC_PATH = "models/preprocessor.pkl"
FEAT_NAMES_PATH = "models/feature_names.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROC_PATH)
feature_names = joblib.load(FEAT_NAMES_PATH)

st.set_page_config(page_title="Personalised Healthcare App", layout="wide")
st.title("🩺 Personalised Healthcare Risk Prediction")

st.markdown("Provide your health parameters and get a real-time prediction with risk insights and visualizations.")

# ----------------- User Inputs -----------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)

with col2:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 190)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 400, 100)
    bmi = st.number_input("BMI", 10.0, 60.0, 24.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

with col3:
    physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"])
    sleep_hours = st.slider("Average Sleep Hours", 3, 12, 7)

user_data = {
    "Age": age,
    "Gender": gender,
    "Systolic_BP": systolic_bp,
    "Diastolic_BP": diastolic_bp,
    "Cholesterol": cholesterol,
    "Glucose_Level": glucose,
    "BMI": bmi,
    "Smoking_Status": smoking_status,
    "Physical_Activity_Level": physical_activity,
    "Alcohol_Consumption": alcohol,
    "Sleep_Hours": sleep_hours
}

# ----------------- Health Warnings -----------------
def health_warnings(data):
    if data["Systolic_BP"] > 140 or data["Diastolic_BP"] > 90:
        st.warning("⚠️ **High Blood Pressure Detected** - Consider consulting a doctor.")
    if data["Cholesterol"] > 240:
        st.warning("⚠️ **High Cholesterol** - Risk of cardiovascular issues.")
    if data["BMI"] > 30:
        st.warning("⚠️ **Obesity Risk** - BMI above 30.")
    if data["Glucose_Level"] > 180:
        st.warning("⚠️ **High Blood Sugar** - Possible diabetes risk.")

# ----------------- Prediction -----------------
if st.button("Predict Health Risk"):
    health_warnings(user_data)

    # Convert input to DataFrame
    user_df = pd.DataFrame([user_data], columns=user_data.keys())

    # Align columns with training features
    user_df = user_df.reindex(columns=feature_names, fill_value=0)

    # Transform input
    X_input = preprocessor.transform(user_df)

    # Predict
    prediction = model.predict(X_input)[0]
    st.subheader(f"**Predicted Risk Level: {prediction}**")

    # ----------------- Visualization -----------------
    st.markdown("### 📊 Your Health Metrics Overview")
    radar_data = pd.DataFrame({
        "Metric": ["Systolic BP", "Diastolic BP", "Cholesterol", "Glucose", "BMI"],
        "Value": [systolic_bp, diastolic_bp, cholesterol, glucose, bmi],
        "Normal_Range": [120, 80, 200, 100, 25]
    })
    fig = px.bar(radar_data, x="Metric", y="Value", color=(radar_data["Value"] > radar_data["Normal_Range"]),
                 color_discrete_map={True: "red", False: "green"}, title="Health Metrics vs. Normal Ranges")
    st.plotly_chart(fig, use_container_width=True)
