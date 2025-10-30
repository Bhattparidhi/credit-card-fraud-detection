# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('rf_model_8features.pkl')  # retrained 8-feature model
scaler = joblib.load('scaler_8features.pkl')   # scaler used during retraining

# Features expected by model
model_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                  'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_TRANSFER']

# Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and click **Predict**.")

# Input fields
step = st.number_input("Step (time interval)", min_value=1)
amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)
transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT"])

# Convert transaction type to model features
type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

if st.button("Predict"):
    # Prepare DataFrame
    data = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type_CASH_OUT': type_CASH_OUT,
        'type_TRANSFER': type_TRANSFER
    }])

    data = data[model_features]
    data_scaled = scaler.transform(data)

    # Predict probabilities and class
    proba = model.predict_proba(data_scaled)[0]
    pred = model.predict(data_scaled)[0]
    fraud_index = list(model.classes_).index(1)

    # Threshold for demo
    threshold = 0.1

    # Display result only
    if proba[fraud_index] >= threshold:
        st.markdown(f"<h2 style='color:red;'>⚠️ FRAUDULENT TRANSACTION</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:green;'>✅ LEGITIMATE TRANSACTION</h2>", unsafe_allow_html=True)

    st.write(f"Probability of FRAUD: **{proba[fraud_index]:.2f}**")
