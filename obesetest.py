import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load assets
model = pickle.load(open('random_forest_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# The exact order used in X_train
train_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'Gender_Male', 'family_history_with_overweight_yes', 'FAVC_yes', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'SMOKE_yes', 'SCC_yes', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']

st.title("Obesity Level Predictor")

# Input fields
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 1.0, 100.0, 25.0)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.70)
    weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0)
    family_history = st.selectbox("Family history with overweight?", ["yes", "no"])
    favc = st.selectbox("High caloric food?", ["yes", "no"])
    fcvc = st.slider("Vegetable consumption", 1.0, 3.0, 2.0)
    ncp = st.slider("Main meals", 1.0, 4.0, 3.0)

with col2:
    caec = st.selectbox("Food between meals", ["Sometimes", "Frequently", "Always", "no"])
    smoke = st.selectbox("Smoke?", ["yes", "no"])
    ch2o = st.slider("Water (L)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Monitor calories?", ["yes", "no"])
    faf = st.slider("Physical activity", 0.0, 3.0, 1.0)
    tue = st.slider("Tech usage", 0.0, 2.0, 1.0)
    calc = st.selectbox("Alcohol", ["Sometimes", "Frequently", "Always", "no"])
    mtrans = st.selectbox("Transportation", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

if st.button("Predict"):
    bmi = weight / (height ** 2)
    
    # Build dictionary with exact keys
    inputs = {
        'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp, 
        'CH2O': ch2o, 'FAF': faf, 'TUE': tue, 'BMI': bmi,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'family_history_with_overweight_yes': 1 if family_history == 'yes' else 0,
        'FAVC_yes': 1 if favc == 'yes' else 0,
        'CAEC_Frequently': 1 if caec == 'Frequently' else 0,
        'CAEC_Sometimes': 1 if caec == 'Sometimes' else 0,
        'CAEC_no': 1 if caec == 'no' else 0,
        'SMOKE_yes': 1 if smoke == 'yes' else 0,
        'SCC_yes': 1 if scc == 'yes' else 0,
        'CALC_Frequently': 1 if calc == 'Frequently' else 0,
        'CALC_Sometimes': 1 if calc == 'Sometimes' else 0,
        'CALC_no': 1 if calc == 'no' else 0,
        'MTRANS_Bike': 1 if mtrans == 'Bike' else 0,
        'MTRANS_Motorbike': 1 if mtrans == 'Motorbike' else 0,
        'MTRANS_Public_Transportation': 1 if mtrans == 'Public_Transportation' else 0,
        'MTRANS_Walking': 1 if mtrans == 'Walking' else 0
    }
    
    # Convert to DF and REORDER columns to match training
    df_input = pd.DataFrame([inputs])
    df_input = df_input[train_cols]
    
    # Scale and Predict
    scaled_data = scaler.transform(df_input.values)
    pred = model.predict(scaled_data)
    label = le.inverse_transform(pred)
    
    st.success(f"Result: {label[0]}")
# ---- PREDICTION ----
if st.button("Predict"):
    data = preprocess()
    result = model.predict(data)

    st.success(f"Predicted Class: {result[0]}")
