import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logistic_reg_heart_pred.pkl")
scaler = joblib.load("scaler.pkl")
exp_columns = joblib.load("columns.pkl")

if callable(exp_columns):
    exp_columns = exp_columns()

st.title("Heart stroke prediction")
st.markdown("provide the following details")

age = st.slider("Age",18,100,40)
Gender = st.selectbox("Gender",['M','F'])
Chest_pain = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
RestingBP = st.number_input("Resting buld Pressure(mm Hg)",80,200,120)
Cholesterol = st.number_input("Cholesterol (mg/dL)",100,600,200)
FastingBS = st.selectbox("FastingBS >120 mg/dL",[0,1])
Resting_ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])
Max_hr = st.slider("Max Heart Rate",60,220,150)
Exercise_Angina = st.selectbox("Exercise Angina",["Y","N"])
Oldpeak = st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
St_slope = st.selectbox("St Slope",["Up","Flat","Down"])

if st.button("Predict"):
    raw_input = {
    'Age': age,
    'RestingBP': RestingBP,
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'MaxHR': Max_hr,
    'Oldpeak': Oldpeak,
    'Sex_' + Gender: 1,
    'ChestPainType_' + Chest_pain: 1,
    'RestingECG_' + Resting_ecg: 1,
    'ExerciseAngina_' + Exercise_Angina: 1,
    'ST_Slope_' + St_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # for col in exp_columns:
    #     if col not in input_df.columns:
    #         input_df[col] = 0
            
    input_df = input_df.reindex(columns=exp_columns, fill_value=0)

    # scaled_input = scaler.transform(input_df)
    # prediction = model.predict(scaled_input)
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Prediction: Heart Disease Detected")
    else:
        st.success("Prediction: Normal")