from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load ML components
model = joblib.load("models/health_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
feature_names = joblib.load("models/feature_names.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Collect input from form
            data = {
                "Age": int(request.form.get("age", 0)),
                "Gender": request.form.get("gender", "Other"),
                "Systolic_BP": float(request.form.get("systolic", 0)),
                "Diastolic_BP": float(request.form.get("diastolic", 0)),
                "Cholesterol": float(request.form.get("cholesterol", 0)),
                "Glucose_Level": float(request.form.get("glucose", 0)),
                "BMI": float(request.form.get("bmi", 0)),
                "Smoking_Status": request.form.get("smoking", "Never"),
                "Physical_Activity_Level": request.form.get("activity", "Low"),
                "Alcohol_Consumption": request.form.get("alcohol", "None"),
                "Sleep_Hours": float(request.form.get("sleep", 0)),
            }

            # Health Warnings
            warnings = []
            if data["Systolic_BP"] > 140 or data["Diastolic_BP"] > 90:
                warnings.append("⚠️ High Blood Pressure Detected.")
            if data["Cholesterol"] > 240:
                warnings.append("⚠️ High Cholesterol.")
            if data["BMI"] > 30:
                warnings.append("⚠️ Obesity Risk.")
            if data["Glucose_Level"] > 180:
                warnings.append("⚠️ High Blood Sugar.")

            # Preprocess for model
            df = pd.DataFrame([data])
            df = df.reindex(columns=feature_names, fill_value=0)
            transformed = preprocessor.transform(df)
            prediction = model.predict(transformed)[0]

            return render_template("result.html", prediction=prediction, warnings=warnings)

        except Exception as e:
            return f"Something went wrong: {str(e)}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
