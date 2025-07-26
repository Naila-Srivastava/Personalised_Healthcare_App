from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load models
model = joblib.load("models/health_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
feature_names = joblib.load("models/feature_names.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    warnings = []

    if request.method == "POST":
        # Grab form data
        data = {
            "Age": int(request.form["age"]),
            "Gender": request.form["gender"],
            "Systolic_BP": float(request.form["systolic"]),
            "Diastolic_BP": float(request.form["diastolic"]),
            "Cholesterol": float(request.form["cholesterol"]),
            "Glucose_Level": float(request.form["glucose"]),
            "BMI": float(request.form["bmi"]),
            "Smoking_Status": request.form["smoking"],
            "Physical_Activity_Level": request.form["activity"],
            "Alcohol_Consumption": request.form["alcohol"],
            "Sleep_Hours": float(request.form["sleep"])
        }

        # Health warnings
        if data["Systolic_BP"] > 140 or data["Diastolic_BP"] > 90:
            warnings.append("⚠️ High Blood Pressure Detected.")
        if data["Cholesterol"] > 240:
            warnings.append("⚠️ High Cholesterol.")
        if data["BMI"] > 30:
            warnings.append("⚠️ Obesity Risk.")
        if data["Glucose_Level"] > 180:
            warnings.append("⚠️ High Blood Sugar.")

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # Reindex to feature list
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Transform and predict
        X = preprocessor.transform(input_df)
        prediction = model.predict(X)[0]

    return render_template("index.html", prediction=prediction, warnings=warnings)

if __name__ == "__main__":
    app.run(debug=True)
