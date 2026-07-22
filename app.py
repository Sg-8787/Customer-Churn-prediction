import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("models/customer_churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn or not.")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox("Gender", label_encoders["gender"].classes_)

    senior = st.selectbox("Senior Citizen", [0, 1])

    partner = st.selectbox("Partner", label_encoders["Partner"].classes_)

    dependents = st.selectbox("Dependents", label_encoders["Dependents"].classes_)

    tenure = st.number_input("Tenure", 0, 100, 12)

    phone = st.selectbox("Phone Service", label_encoders["PhoneService"].classes_)

    multiple = st.selectbox("Multiple Lines", label_encoders["MultipleLines"].classes_)

    internet = st.selectbox("Internet Service", label_encoders["InternetService"].classes_)

    security = st.selectbox("Online Security", label_encoders["OnlineSecurity"].classes_)

with col2:

    backup = st.selectbox("Online Backup", label_encoders["OnlineBackup"].classes_)

    device = st.selectbox("Device Protection", label_encoders["DeviceProtection"].classes_)

    tech = st.selectbox("Tech Support", label_encoders["TechSupport"].classes_)

    tv = st.selectbox("Streaming TV", label_encoders["StreamingTV"].classes_)

    movies = st.selectbox("Streaming Movies", label_encoders["StreamingMovies"].classes_)

    contract = st.selectbox("Contract", label_encoders["Contract"].classes_)

    paper = st.selectbox("Paperless Billing", label_encoders["PaperlessBilling"].classes_)

    payment = st.selectbox("Payment Method", label_encoders["PaymentMethod"].classes_)

monthly = st.number_input("Monthly Charges", value=70.0)

total = st.number_input("Total Charges", value=800.0)

if st.button("Predict"):

    data = [
        label_encoders["gender"].transform([gender])[0],
        senior,
        label_encoders["Partner"].transform([partner])[0],
        label_encoders["Dependents"].transform([dependents])[0],
        tenure,
        label_encoders["PhoneService"].transform([phone])[0],
        label_encoders["MultipleLines"].transform([multiple])[0],
        label_encoders["InternetService"].transform([internet])[0],
        label_encoders["OnlineSecurity"].transform([security])[0],
        label_encoders["OnlineBackup"].transform([backup])[0],
        label_encoders["DeviceProtection"].transform([device])[0],
        label_encoders["TechSupport"].transform([tech])[0],
        label_encoders["StreamingTV"].transform([tv])[0],
        label_encoders["StreamingMovies"].transform([movies])[0],
        label_encoders["Contract"].transform([contract])[0],
        label_encoders["PaperlessBilling"].transform([paper])[0],
        label_encoders["PaymentMethod"].transform([payment])[0],
        monthly,
        total
    ]

    df = pd.DataFrame([data])

    df = scaler.transform(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.metric("Churn Probability", f"{probability*100:.2f}%")