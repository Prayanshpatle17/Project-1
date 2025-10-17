import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

MODEL_PATH = "model/diabetes_model.pkl"

# Ensure folders exist
os.makedirs("model", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/diabetes.csv")

# Train model if not exists
if not os.path.exists(MODEL_PATH):
    st.write("🔄 Training model, please wait...")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    joblib.dump(model, MODEL_PATH)
    st.write(f"✅ Model trained and saved! Accuracy: {acc*100:.2f}%")
else:
    st.write("✅ Using saved model.")

# Load model
model = joblib.load(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
st.title("🩺 Diabetes Risk Prediction")
st.write("Enter the following details to predict diabetes risk:")

# Input fields
Pregnancies = st.number_input("Pregnancies", 0, 20, 0)
Glucose = st.number_input("Glucose Level", 0, 200, 120)
BloodPressure = st.number_input("Blood Pressure", 0, 150, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 79)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.number_input("Age", 1, 120, 25)

if st.button("Predict"):
    data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
                        columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
