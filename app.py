"""
Streamlit app for Medical Premium Price Prediction
"""

import streamlit as st
import numpy as np
import joblib

# Load the trained regression model
model = joblib.load("model/model.pkl")

# Streamlit application
st.title("Medical Premium Price Prediction")
st.write("Enter patient details below to predict the insurance premium price.")

# Define a mapping dictionary
# This allows the UI to show 'Yes/No' while the model receives '1/0'
mapping = {"No": 0, "Yes": 1}

# Input Features
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# We use the keys ("No", "Yes") for the UI
diabetes = st.selectbox("Diabetes", options=list(mapping.keys()))
bp = st.selectbox("Blood Pressure Problems", options=list(mapping.keys()))
transplants = st.selectbox("Any Transplants", options=list(mapping.keys()))
chronic = st.selectbox("Any Chronic Diseases", options=list(mapping.keys()))

height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

allergies = st.selectbox("Known Allergies", options=list(mapping.keys()))
cancer_history = st.selectbox("History of Cancer in Family", options=list(mapping.keys()))
surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=10, value=1)

# Prepare the input data by mapping "Yes"/"No" back to 1/0
input_data = np.array([[
    age,
    mapping[diabetes],
    mapping[bp],
    mapping[transplants],
    mapping[chronic],
    height,
    weight,
    mapping[allergies],
    mapping[cancer_history],
    surgeries
]])

# Predict the premium
if st.button("Predict Premium"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Premium Price: ₹ {round(prediction, 2)}")