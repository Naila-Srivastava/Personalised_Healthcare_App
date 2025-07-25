# Personalized Healthcare Recommendations

Building a machine learning pipeline to assess individual health risks and provide **personalized lifestyle and healthcare recommendations**, deployed via a clean Streamlit dashboard.

---

## 🧠 Objectives

* Analyze and visualize health-related indicators (age, BMI, lifestyle factors, etc.).
* Build and evaluate multiple Machine Learning models for **risk level classification (Low, Moderate, High)**.
* Generate actionable, personalized healthcare recommendations.
* Deploy an **interactive Streamlit app** for real-time prediction and visualization.

---

## 🔧 Tools & Technologies

* **Python** (Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-Learn and XGBoost)
* **Streamlit** – For building the interactive dashboard and app
* **Jupyter Notebook** – For data analysis and EDA
* **Joblib** – For saving and loading model artifacts
* **Git & GitHub** – Version control and collaboration

---

## 📂 Dataset

- The dataset includes **health indicators** like age, BMI, exercise frequency, smoking habits, alcohol consumption, and other lifestyle factors.
- Due to privacy and size constraints, the full dataset is **not included** in this repository.

➡️ [Download dataset here](https://www.kaggle.com/datasets/nailasrivastava/personal_healthcare_recommendations)

---

## 💻 How to Run

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks:
   * `EDA_Healthcare.ipynb` for data exploration and visualization.
   * `ML_Healthcare.ipynb` for machine learning pipeline and training.
4. Launch the Streamlit app: `streamlit run app.py`
   
---

## 🤖 Models and their Features

**Feature Engineering**

* Missing value imputation, and encoding of categorical variables.
* Feature scaling (StandardScaler) and feature selection via `SelectKBest`.

**Machine Learning Models**

* Logistic Regression
* Random Forest (Best performer)
* XGBoost (Competitive performance)

**Evaluation Metrics**

* Accuracy, Precision, Recall, F1-Score, ROC-AUC.
* Confusion Matrix, ROC curves, and cross-validation.

---

## 🧠 Model Evaluation Summary

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| Logistic Reg. | 1.00     | 1.00      | 1.00   | 1.00     |
| Random Forest | 1.00     | 1.00      | 1.00   | 1.00     |
| XGBoost       | 1.00     | 1.00      | 1.00   | 1.00     |

> Note:- **All the models have the same accuracy, because:
> Perfect class separation 
> The dataset is too clean/simple, with no noise
> The target variable is heavily correlated with one or two main features, so all models are learning the same obvious thing.**

---

## 📊 Key Analysis

* **BMI and smoking habits** were strong predictors of high health risk.
* **Exercise frequency** and **alcohol consumption** had a significant impact on risk levels.
* Correlation heatmaps revealed interesting interactions between lifestyle factors and predicted risks.

---

## 📈 Sample Visualizations

* Correlation Heatmap (Top 10 features)
* Distribution Plots (Age, BMI, etc.)
* Feature Importance (Random Forest & XGBoost)
* Confusion Matrix & ROC Curve

---

## 🌍 Streamlit Dashboard Features

* User-friendly input panel for entering health data.
* Real-time prediction of risk category (Low/Moderate/High).
* Personalized healthcare recommendations (exercise, diet, monitoring).
* Visual display of **feature importance** for interpretability.

👉 **Check out the Streamlit app (add link after deployment).**[ ]

---

## 📝 Key Takeaways

* Lifestyle factors strongly influence health risk and can be targeted for recommendations.
* Ensemble models like Random Forest & XGBoost offer high accuracy while being interpretable.
* A simple, actionable **recommendation engine** adds practical value to predictive models.
* Deployment with Streamlit makes the project accessible and interactive.

---

## 🔐 Model Artifacts

Pre-trained model and preprocessor files are available in the `/models` directory:

* `health_model.pkl` – Trained Random Forest model
* `preprocessor.pkl` – ColumnTransformer for data preprocessing
* `feature_names.pkl` – List of final feature names

---

## 🧩 What’s Next?

* Integrate SHAP for model explainability (why a risk prediction was made).
* Add **batch prediction** (CSV upload in the Streamlit app).
* Deploy on **Streamlit Cloud** with a custom domain.

---

## 🙌 Acknowledgments

* Thanks to Streamlit community for making model deployment seamless.
* This project was completed as part of an This project was completed as part of a 12-week Upskilling Sprint (Data Analyst Internship), focused on healthcare analytics.

---

## 📂 Project Structure

```plaintext
Personalized_Healthcare_App/
│
├── README.md                                  # You're reading this now
├── requirements.txt                           # Dependencies list
├── personalised_data.csv                      # Folder for the dataset (if any)
├── Data_analysis.ipynb                        # Data analysis, visualization and ML pipeline script
├── charts/                                    # Folder for all generated plots
├── models/                                    # Folder for trained models
│   ├── Model_README.md
│   ├── health_model.pkl
│   ├── preprocessor.pkl
│   └── feature_names.pkl
├── train_model.py                             # Script to train and save model
└── Personalised_healthcare_app.py             # Streamlit dashboard script
