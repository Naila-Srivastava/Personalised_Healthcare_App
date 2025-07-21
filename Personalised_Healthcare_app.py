import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and preprocessor
model = joblib.load("models/health_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Recommendation logic
def generate_recommendations(prediction):
    if prediction == "High":
        return [
            "⚠️ High health risk detected! Consult a doctor ASAP.",
            "Start high-priority lifestyle changes (exercise & diet).",
            "Track key health metrics weekly (BP, glucose, etc.)."
        ]
    elif prediction == "Moderate":
        return [
            "⚠️ Moderate risk. Consider lifestyle adjustments.",
            "Exercise 3–4 times/week.",
            "Cut down processed foods and monitor weight."
        ]
    else:
        return [
            "✅ Low risk. Keep up the healthy habits!",
            "Maintain balanced diet and daily activity.",
            "Regular health check-ups (once a year)."
        ]

# UI
st.title("🏥 Personalized Healthcare Dashboard")
st.write("Get your health risk assessment and personalized recommendations.")

# Sidebar Inputs
st.sidebar.header("User Health Input")
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
exercise = st.sidebar.selectbox("Exercise Level", ["Low", "Moderate", "High"])
smoking = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
alcohol = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])

user_data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Exercise": [exercise],
    "Smoking": [smoking],
    "Alcohol": [alcohol]
})

st.subheader("Your Input Summary")
st.table(user_data)

# Prediction Button 
if st.button("Predict Health Risk"):
    user_data_flat = np.array(user_data).flatten()  # Ensure it's 1D
    user_data_df = pd.DataFrame([user_data_flat], columns=feature_names)

    # Load feature names from training
    feature_names = joblib.load("models/feature_names.pkl")

    # Ensure all columns exist 
    user_data_full = pd.DataFrame(columns=feature_names)
    user_data_full.loc[0] = 0 

    # Update with actual user input values
    for col in user_data_df.columns:
        if col in user_data_full.columns:
            user_data_full[col] = user_data_df[col].values[0]

    # Apply preprocessor
    X_input = preprocessor.transform(user_data_full)

    # Predict
    prediction = model.predict(X_input)[0]

    st.subheader(f"Predicted Risk Level: **{prediction}**")

    st.write("### Personalized Recommendations:")
    for rec in generate_recommendations(prediction):
        st.write(f"- {rec}")

    # Feature importance chart
    st.write("### Feature Importance (Random Forest)")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [feature_names[i] for i in sorted_idx[:5]]
    top_importances = importances[sorted_idx[:5]]

    fig, ax = plt.subplots()
    sns.barplot(x=top_importances, y=top_features, palette="coolwarm", ax=ax)
    ax.set_title("Top 5 Important Features")
    st.pyplot(fig)
