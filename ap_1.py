import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

# Page configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict churn probability.")

# --- User Inputs ---
credit_score = st.slider("Credit Score", 300, 850, 650)
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5)
balance = st.number_input("Account Balance ($)", value=50000.0, step=100.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary ($)", value=60000.0, step=100.0)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])

# --- One-hot Encoding (as per model's training) ---
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# --- Create input DataFrame with correct feature order ---
input_data = pd.DataFrame([[
    credit_score, age, tenure, balance, num_products, has_cr_card,
    is_active_member, estimated_salary, geo_germany, geo_spain, gender_male
]], columns=[
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male'
])

# --- Prediction ---
if st.button("Predict Churn"):
    prob = model.predict_proba(input_data)[0][1]
    st.metric(label="Churn Probability", value=f"{prob*100:.2f}%")

    if prob >= 0.3:
        st.warning("⚠️ High churn risk! Consider retention strategies.")
    else:
        st.success("✅ Low churn risk. Customer likely to stay.")
