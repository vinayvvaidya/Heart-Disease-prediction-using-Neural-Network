import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load your pre-trained model (make sure the model file is in the same directory)
model = load_model('heart_disease_model.h5')
data = pd.read_csv('heart.csv')

mean = data.mean(axis=0)
std = data.std(axis=0)

mean_age = mean['age']
mean_sex = mean['sex']
mean_cp = mean['cp']
mean_trestbps = mean['trestbps']
mean_chol = mean['chol']
mean_fbs = mean['fbs']
mean_restecg = mean['restecg']
mean_thalach = mean['thalach']
mean_exang = mean['exang']
mean_oldpeak = mean['oldpeak']
mean_slope = mean['slope']
mean_ca = mean['ca']
mean_thal = mean['thal']

std_age = std['age']
std_sex = std['sex']
std_cp = std['cp']
std_trestbps = std['trestbps']
std_chol = std['chol']
std_fbs = std['fbs']
std_restecg = std['restecg']
std_thalach = std['thalach']
std_exang = std['exang']
std_oldpeak = std['oldpeak']
std_slope = std['slope']
std_ca = std['ca']
std_thal = std['thal']

st.title("Heart Disease Prediction")
st.write("Enter the patient details below to predict the likelihood of heart disease:")

# Input fields (adjust these fields based on your dataset features)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type", options=["Type 1", "Type 2", "Type 3", "Type 4"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
restecg = st.selectbox("Resting ECG Results", options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
oldpeak = st.number_input("ST Depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST segment", options=["Upsloping", "Flat", "Downsloping"])
ca = st.number_input("Number of major vessels colored by fluoroscopy", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia", options=["Normal", "Fixed defect", "Reversible defect"])

mean = np.array([mean_age, mean_sex, mean_cp, mean_trestbps, mean_chol, mean_fbs, mean_restecg, mean_thalach, mean_exang, mean_oldpeak, mean_slope, mean_ca, mean_thal])
std = np.array([std_age, std_sex, std_cp, std_trestbps, std_chol, std_fbs, std_restecg, std_thalach, std_exang, std_oldpeak, std_slope, std_ca, std_thal])

# Preprocessing function to convert inputs to the format expected by the model
def preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Convert categorical inputs to numerical values based on your training mapping.
    sex_val = 1 if sex == "Male" else 0

    cp_mapping = {"Type 1": 0, "Type 2": 1, "Type 3": 2, "Type 4": 3}
    cp_val = cp_mapping[cp]

    fbs_val = 1 if fbs == "Yes" else 0

    restecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    restecg_val = restecg_mapping[restecg]

    exang_val = 1 if exang == "Yes" else 0

    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope_val = slope_mapping[slope]

    thal_mapping = {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}
    thal_val = thal_mapping[thal]

    # Construct an array in the order expected by your model. This order must match your training phase.
    input_data = np.array([age, sex_val, cp_val, trestbps, chol, fbs_val,
                           restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val])
    
    # If your model was trained on normalized data, apply the same normalization here.
    # For demonstration purposes, we're assuming the model handles scaling internally
    # or that the input values are pre-scaled appropriately.

     # Apply the same normalization as during training
    input_data = (input_data - mean) / std

    print("Input Data:", input_data)
    
    return input_data.reshape(1, -1)
    # return input_data

# When the user clicks the "Predict" button, preprocess inputs and perform prediction
if st.button("Predict"):
    input_data = preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    # temp_data = np.array([1.05491812, 0.66150409, -0.91575542, -0.66321646, 0.0, -0.41887792,
    #       -1.00404855, -2.3098632, 1.40392824, 0.96084044, -2.24367514, 0.23862459,
    #       -0.52212231])
    
    # temp_data = temp_data.reshape(1, -1)
    # prediction = model.predict(input_data)
    prediction = model.predict(input_data)

    # Depending on your model's architecture, handle the output appropriately.
    # If using a binary classification with sigmoid activation:
    probability = prediction[0][0]
    print("Probability:", prediction)
    threshold = 0.5
    if probability > threshold:
        result = "The model predicts that the person may have heart disease."
    else:
        result = "The model predicts that the person may not have heart disease."
    
    st.subheader("Prediction Result")
    st.write(result)
    st.write(f"Predicted Probability: {probability:.2f}")
