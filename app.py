import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Random Forest Prediction", layout="centered")

st.title("🔮 Random Forest Prediction App")
st.write("Enter feature values to get prediction")

# Input form
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0
    )

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"✅ Predicted Value: {prediction[0]}")
