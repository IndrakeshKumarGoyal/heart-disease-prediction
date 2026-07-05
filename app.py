from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask App
app = Flask(__name__)

# Load Trained Model
model = joblib.load("disease_prediction_model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = {
            "age": float(request.form["age"]),
            "sex": int(request.form["sex"]),
            "cp": int(request.form["cp"]),
            "trestbps": float(request.form["trestbps"]),
            "chol": float(request.form["chol"]),
            "fbs": int(request.form["fbs"]),
            "restecg": int(request.form["restecg"]),
            "thalach": float(request.form["thalach"]),
            "exang": int(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": int(request.form["slope"]),
            "ca": int(request.form["ca"]),
            "thal": int(request.form["thal"])
        }

        df = pd.DataFrame([data])

        prediction = model.predict(df)

        if prediction[0] == 1:
            result = "❤️ Heart Disease Detected"
        else:
            result = "✅ No Heart Disease"

        return f"""

        Prediction: {result}<br>
        Probability: {round(model.predict_proba(df)[0][1]*100,2)}%
       """
    except Exception as e:
        return f"Error: {e}"


# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)