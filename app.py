import streamlit as st
import numpy as np
import joblib  # Replaced pickle with joblib

# Load the trained ensemble model
model = joblib.load("ensemble_model.pkl")

st.set_page_config(page_title="Employee Retention App", layout="centered")

st.title("🏢 Employee Retention Predictor")
st.markdown("Use this app to predict whether an employee will be **retained** or **likely to leave** based on their details.")

# Input form
with st.form("employee_form"):
    st.subheader("📝 Employee Details")

    age = st.slider("Age", 20, 60, 30)
    experience = st.slider("Years of Experience", 0, 40, 5)
    salary = st.number_input("Salary (in ₹)", min_value=5000, max_value=200000, value=50000, step=1000)
    satisfaction = st.slider("Job Satisfaction (0-1)", 0.0, 1.0, 0.5)
    work_life = st.slider("Work-Life Balance (0-1)", 0.0, 1.0, 0.5)
    performance = st.slider("Performance Rating (0-1)", 0.0, 1.0, 0.5)
    training = st.slider("Training Hours", 0, 100, 10)
    promotion = st.selectbox("Promotion in Last 5 Years", [0, 1])
    department = st.selectbox("Department", ["Management", "Sales", "Support", "Technical"])

    submitted = st.form_submit_button("🔎 Predict Retention")

if submitted:
    # One-hot encode department (order must match training data)
    dept_list = ['Management', 'Sales', 'Support', 'Technical']
    dept_encoded = [1 if department == d else 0 for d in dept_list]

    # Final input vector (match training order)
    input_data = np.array([[
        age, experience, salary, satisfaction,
        work_life, performance, training, promotion
    ] + dept_encoded])

    # Prediction
    prediction = model.predict(input_data)[0]
    risk_score = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success("✅ The employee is likely to be **Retained**.")
    else:
        st.error("❌ The employee is at **Risk of Leaving**.")
    
    st.metric("🔐 Retention Risk Score", f"{risk_score:.2f}")
