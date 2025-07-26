from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load your models
model_path = os.path.join('models', 'health_model.pkl')
model = joblib.load(model_path)

# Required columns
REQUIRED_COLUMNS = {
    'Gender', 'Smoking_Status', 'Alcohol_Consumption', 'Physical_Activity_Level',
    'Diet_Type', 'Sleep_Quality', 'Blood_Pressure',
    'Heart_Disease_Risk', 'Diabetes_Risk',
    'Exercise_Recommendation', 'Diet_Recommendation'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Failed to read CSV: {e}", 400

    # Check for required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return f"Missing required columns: {missing_cols}", 400

    # Proceed with prediction if all good
    predictions = model.predict(df[REQUIRED_COLUMNS])
    df['Prediction'] = predictions

    return df.to_html(classes='table table-striped')

if __name__ == '__main__':
    app.run(debug=True)
