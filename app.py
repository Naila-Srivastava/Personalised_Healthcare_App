from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load("model/life_expectancy_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    df = pd.read_csv(file)
    required_columns = {'Gender','Smoking_Status','Alcohol_Consumption','Physical_Activity_Level',
                        'Diet_Type','Blood_Pressure','Sleep_Quality','Diabetes_Risk',
                        'Heart_Disease_Risk','Exercise_Recommendation','Diet_Recommendation'}

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        return f"Missing columns: {missing}", 400

    predictions = model.predict(df)
    df['Predicted_Life_Expectancy'] = predictions
    results = df[['Gender', 'Predicted_Life_Expectancy']]

    return render_template('result.html', tables=[results.to_html(classes='table table-bordered', index=False)])

if __name__ == '__main__':
    app.run(debug=True)
