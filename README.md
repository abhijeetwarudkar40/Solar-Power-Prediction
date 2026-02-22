# 🌞 Solar Power Prediction System

A Machine Learning project that predicts solar AC power output using environmental and temporal features.  
The model is deployed as a Streamlit web application for real-time predictions.

---

## 📌 Project Overview

This project combines solar inverter generation data with weather sensor data to build a photovoltaic power forecasting model.

The system:
- Cleans and merges SCADA + weather data
- Engineers solar physics-based features
- Trains a Random Forest regression model
- Deploys the trained model using Streamlit

Final Model Performance:
- **R² Score:** ~0.92
- **MAE:** ~40
- **RMSE:** ~89

---

## 🧠 Key Features Used

- Irradiation
- Module Temperature
- Cyclic Time Encoding (Sun Position)
- Temperature Efficiency Loss
- Interaction Features
- Lag Features (Previous Power, Irradiation, Temperature)
- Inverter-specific encoding

---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 🚀 How to Run Locally

1. Clone the repository
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run Solar_Power_Predictor.ipynb
4. Add model.pkl, features.pkl, and scaler.pkl in Project Folder
5. Run the app
```bash
 streamlit run app.py
```

## 🌐 Deployment
The model is deployed using Streamlit Cloud.
Users can input weather conditions and get real-time AC power predictions.

## 📊 Model Workflow
1. Data Merging (Generation + Weather)
2. Missing Value Handling
3. Feature Engineering (Physics + Time + Lag)
4. One-Hot Encoding
5. Scaling
6. Random Forest Training
7. Model Evaluation
8. Web Deployment

## 🎯 Project Goal
To demonstrate how environmental, temporal, and photovoltaic physics features can effectively model and predict solar energy generation.
