import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="MS/MS Spectrum Prediction",
    layout="centered"
)

st.title("🔬 MS/MS Spectrum Analysis Prediction")

# ---------------------------------
# Base directory
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")

# ---------------------------------
# Load model
# ---------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Model file not found or failed to load.")
    st.stop()

# ---------------------------------
# Load features (SAFE)
# ---------------------------------
try:
    features = joblib.load(FEATURES_PATH)
except Exception:
    # Fallback: get feature names from model itself
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
        st.warning("⚠️ model_features.pkl not found. Using model feature names.")
    else:
        st.error("❌ Feature names not available.")
        st.stop()

# ---------------------------------
# Input form
# ---------------------------------
st.write("Enter feature values to predict **OE/O6T P value**")

with st.form("prediction_form"):
    input_data = {}

    for feature in features:
        input_data[feature] = st.number_input(
            label=str(feature),
            value=0.0
        )

    submit = st.form_submit_button("Predict")

# ---------------------------------
# Prediction
# ---------------------------------
if submit:
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"✅ Predicted OE/O6T P value: **{prediction[0]:.6f}**")
