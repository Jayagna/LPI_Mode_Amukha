import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and feature columns
model = pickle.load(open("interest_rate_model.pkl", "rb"))
columns = pickle.load(open("feature_columns.pkl", "rb"))

st.title("📈 Interest Rate Predictor")

st.write("Enter the required features below to predict the expected loan interest rate:")

# Dynamically generate input fields
user_input = []
for col in columns:
    val = st.number_input(f"{col}", value=0.0, format="%.4f")
    user_input.append(val)

# Predict on button click
if st.button("Predict Interest Rate"):
    input_df = pd.DataFrame([user_input], columns=columns)
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Interest Rate: {round(prediction, 2)}%")
