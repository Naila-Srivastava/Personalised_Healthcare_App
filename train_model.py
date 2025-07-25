import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# === 1. Load dataset ===
df = pd.read_csv('personalised_dataset.csv')
print(f"Dataset loaded: {df.shape}")

# === 2. Define target and features ===
target = 'Health_Risk' 
# Define columns we actually collect in the app
required_cols = [
    "Age", "Gender", "Systolic_BP", "Diastolic_BP", "Cholesterol",
    "Glucose_Level", "BMI", "Smoking_Status", "Physical_Activity_Level",
    "Alcohol_Consumption", "Sleep_Hours"
]

X = df[required_cols]
y = df[target]

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Numeric cols: {list(numeric_cols)}")
print(f"Categorical cols: {list(categorical_cols)}")

# === 3. Preprocessor ===
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# === 4. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Pipeline with RandomForest ===
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
model.fit(X_train, y_train)
print("Model training completed!")

# === 6. Save Artifacts (like last project) ===
os.makedirs('./models', exist_ok=True)

# Save the whole pipeline (includes preprocessor + model)
joblib.dump(model, './models/health_model.pkl')
print("health_model.pkl saved!")

# Save preprocessor separately
joblib.dump(preprocessor, './models/preprocessor.pkl')
print("preprocessor.pkl saved!")

# Save feature names
feature_names = list(numeric_cols) + list(
    model.named_steps['preprocessor']
        .named_transformers_['cat']
        .get_feature_names_out(categorical_cols)
)
joblib.dump(feature_names, './models/feature_names.pkl')
print("feature_names.pkl saved!")

print("All artifacts saved in /models")
