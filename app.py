import streamlit as st
import numpy as np
import joblib

# Load your trained machine learning model from a .sav file
classifier = joblib.load('model.sav')

st.title("Diabetes Prediction Using Machine Learning")

st.write("Enter the following information:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=100)
glucose = st.number_input("Glucose", min_value=0, max_value=500)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=1000)
bmi = st.text_input("BMI", value="0.000")
bmi = float(bmi)  # Convert to float
# Use a text input for Diabetes Pedigree Function
diabetes_pedigree = st.text_input("Diabetes Pedigree Function", value="0.000")
diabetes_pedigree = float(diabetes_pedigree)  # Convert to float
age = st.number_input("Age", min_value=0, max_value=120)

input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)

if st.button("Predict"):
    prediction = classifier.predict(input_data)

    if (prediction == 0):
        st.write('The person is not diabetic')
    else:
        st.write('The person is diabetic')
