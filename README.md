# Customer-Churn-Prediction
📊 Customer Churn Prediction App
Overview
This project is a machine learning-based web app designed to predict whether a customer is likely to churn from a bank. Built using Python, scikit-learn, and Streamlit, the app allows users to input customer details and receive a churn probability prediction, enabling data-driven decisions to retain valuable customers.

🔍 Problem Statement
Customer churn is a major challenge in the banking sector. Predicting which customers are at risk of leaving allows banks to take proactive retention measures. This project aims to build a classification model that can accurately predict churn based on customer attributes.

💡 Key Features
Built using a Decision Tree Classifier trained on real-world bank customer data

Feature inputs include:

Demographics (Age, Gender, Geography)

Financials (Balance, Estimated Salary, Credit Score)

Account activity (Tenure, Products, Credit Card usage, Membership status)

Interactive Streamlit UI to simulate customer scenarios and predict churn in real-time

Displays churn probability and suggests action based on a configurable cutoff threshold

🛠️ Technologies Used
Python (pandas, scikit-learn, joblib)

Streamlit for web app interface

VS Code for development

📁 Files Included
train_model.py – Preprocesses data and trains/saves the model

ap_1.py – Streamlit app for real-time predictions

churn_model.pkl – Trained model file

Churn_Modelling.csv – Original dataset used for model training

📌 How to Run
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run ap_1.py
🔮 Future Enhancements
Add feature importance visualization

Log and export predictions

Integrate SHAP/LIME for model explainability

Deploy to Streamlit Cloud for public access

