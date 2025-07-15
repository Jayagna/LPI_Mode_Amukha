from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and the expected feature columns
model = pickle.load(open("interest_rate_model.pkl", "rb"))
columns = pickle.load(open("feature_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("form.html", columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        input_data = [float(request.form[col]) for col in columns]
        input_df = pd.DataFrame([input_data], columns=columns)

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template("result.html", prediction=prediction)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)