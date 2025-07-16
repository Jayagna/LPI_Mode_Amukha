import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and feature columns
model = pickle.load(open("interest_rate_model.pkl", "rb"))
columns = pickle.load(open("feature_columns.pkl", "rb"))

st.title("📈 Interest Rate Predictor")

st.write("Select or enter the required features below to get the predicted interest rate:")

# Manually define which categorical values you want to treat as dropdowns
state_options = ['MS', 'IN', 'SD', 'MT']  # replace with actual states in your model
loan_purpose_options = [
    'educational', 'house', 'other', 'credit_card', 'small_business',
    'debt_consolidation', 'home_improvement', 'moving', 'major_purchase'
]  # replace with actual loan purposes in your model

# Dropdowns for state and loan purpose
selected_state = st.selectbox("Select State", state_options)
selected_purpose = st.selectbox("Select Loan Purpose", loan_purpose_options)

# Other numerical inputs
other_cols = [
    col for col in columns
    if not col.startswith('State_') and not col.startswith('Loan_Purpose_')
]

user_input = {}

for col in other_cols:
    user_input[col] = st.number_input(col, value=0.0, format="%.4f")

# Prepare final input vector
input_vector = []

for col in columns:
    if col.startswith("State_"):
        state_code = col.replace("State_", "")
        input_vector.append(1 if state_code == selected_state else 0)
    elif col.startswith("Loan_Purpose_"):
        purpose_code = col.replace("Loan_Purpose_", "")
        input_vector.append(1 if purpose_code == selected_purpose else 0)
    else:
        input_vector.append(user_input[col])

# Predict on button click
if st.button("Predict Interest Rate"):
    input_df = pd.DataFrame([input_vector], columns=columns)
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Interest Rate: {round(prediction, 2)}%")
