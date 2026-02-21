import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ---------------- UI ----------------

st.title("Solar Power Prediction System")
st.write("Enter weather conditions:")

irr = st.number_input("Irradiation", 0.0, 2.0, 0.8)
temp = st.number_input("Module Temperature (°C)", 0.0, 80.0, 35.0)
hour = st.slider("Hour of Day", 6, 18, 12)
prev_power = st.number_input("Previous Power (kW)", 0.0, 1000.0, 200.0)

# ---------------- PREDICTION ----------------

if st.button("Predict Power"):
    # Feature engineering
    hour_sin = np.sin(2*np.pi*hour/24)
    hour_cos = np.cos(2*np.pi*hour/24)

    temp_diff = temp - 25
    temp_eff = 1 - (0.004 * temp_diff)

    # Create dataframe
    input_data = pd.DataFrame(
        [[irr, temp, hour_sin, hour_cos, temp_diff, temp_eff, prev_power]],
        columns=[
            'IRRADIATION',
            'MODULE_TEMPERATURE',
            'HOUR_SIN',
            'HOUR_COS',
            'TEMP_DIFF',
            'TEMP_EFFICIENCY',
            'PREV_POWER'
        ]
    )

    # Align with training features
    input_data = input_data.reindex(columns=features, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Output
    st.success(f"Predicted AC Power: {prediction[0]:.2f} kW")
