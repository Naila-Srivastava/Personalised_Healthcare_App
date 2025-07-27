from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = joblib.load('./models/health_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filepath = os.path.join(UPLOAD_FOLDER, f"{uuid4()}.csv")
    file.save(filepath)

    df = pd.read_csv(filepath)

    required_cols = [
        "Age", "Gender", "Systolic_BP", "Diastolic_BP", "Cholesterol",
        "Glucose_Level", "BMI", "Smoking_Status", "Physical_Activity_Level",
        "Alcohol_Consumption", "Sleep_Hours"
    ]

    # Validation
    if not all(col in df.columns for col in required_cols):
        return "Missing columns in the uploaded file.", 400

    # Predictions
    predictions = model.predict(df[required_cols])
    df['Health_Risk'] = predictions

    # Recommendation logic
    def get_recommendation(row):
        if row['Health_Risk'] == 'High':
            return "Urgent: Consult a physician. Adopt a strict healthy lifestyle."
        elif row['Health_Risk'] == 'Medium':
            return "Moderate risk. Regular checkups and lifestyle changes advised."
        else:
            return "Low risk. Maintain current lifestyle and regular exercise."

    df['Recommendation'] = df.apply(get_recommendation, axis=1)

    # Save results temporarily
    result_path = os.path.join(UPLOAD_FOLDER, f"result_{uuid4()}.csv")
    df.to_csv(result_path, index=False)

    # Generate plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Health_Risk', palette='Set2')
    plt.title("Health Risk Distribution")
    plot_path = os.path.join('static', f"plot_{uuid4()}.png")
    plt.savefig(plot_path)
    plt.close()

    return render_template('result.html', tables=df.to_dict(orient='records'), plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
