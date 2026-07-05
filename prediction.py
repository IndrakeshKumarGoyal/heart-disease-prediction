import joblib
import pandas as pd

# Load trained model
model = joblib.load("disease_prediction_model.pkl")

# Sample patient data
sample_data = pd.DataFrame([{
    "age": 52,
    "sex": 1,
    "cp": 2,
    "trestbps": 172,
    "chol": 199,
    "fbs": 1,
    "restecg": 1,
    "thalach": 162,
    "exang": 0,
    "oldpeak": 0.5,
    "slope": 2,
    "ca": 0,
    "thal": 3
}])

# Predict
prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")