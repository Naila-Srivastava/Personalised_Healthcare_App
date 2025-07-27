from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/health_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "Empty filename", 400

    try:
        df = pd.read_csv(file)

        # Define required columns
        required_columns = {'Age', 'Gender', 'Smoking_Status', 'Alcohol_Consumption',
                            'Physical_Activity_Level', 'Sleep_Quality', 'Blood_Pressure'}

        # Check for missing columns
        missing = required_columns - set(df.columns)
        if missing:
            return f"Missing columns: {missing}", 400

        # Predict and append results
        df["Health_Risk_Prediction"] = model.predict(df)

        return render_template("result.html", tables=[df.to_html(classes="table table-bordered", index=False)])
    
    except Exception as e:
        return f"Error processing file: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
