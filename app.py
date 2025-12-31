import numpy as np
import pandas as pd
import sqlite3
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("diabetes.csv")

cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
    data[col] = data[col].replace(0, data[col].median())

X = data.drop("Outcome", axis=1)
y_clf = data["Outcome"]
y_reg = data["Glucose"]

# -----------------------------
# Train Models
# -----------------------------
clf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
reg_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)

clf_model.fit(X, y_clf)
reg_model.fit(X, y_reg)

# -----------------------------
# SQLite Database
# -----------------------------
conn = sqlite3.connect("diabetes_predictions.db", check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Pregnancies INTEGER,
    Glucose REAL,
    BloodPressure REAL,
    SkinThickness REAL,
    Insulin REAL,
    BMI REAL,
    DiabetesPedigreeFunction REAL,
    Age INTEGER,
    Predicted_Glucose REAL,
    Diabetes_Outcome INTEGER
)
""")
conn.commit()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction System")

st.sidebar.header("Enter Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20)
glucose = st.sidebar.number_input("Glucose", 0.0, 300.0)
bp = st.sidebar.number_input("Blood Pressure", 0.0, 200.0)
skin = st.sidebar.number_input("Skin Thickness", 0.0, 100.0)
insulin = st.sidebar.number_input("Insulin", 0.0, 900.0)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.sidebar.number_input("Age", 1, 120)

if st.sidebar.button("Predict"):
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [bp],
        "SkinThickness": [skin],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    predicted_outcome = clf_model.predict(input_data)[0]
    predicted_glucose = reg_model.predict(input_data)[0]

    st.subheader("🔍 Prediction Result")
    st.write("**Diabetes Status:**", "🟥 Diabetic" if predicted_outcome == 1 else "🟩 Non-Diabetic")
    st.write("**Predicted Glucose Level:**", round(predicted_glucose, 2))

    # Save to DB
    conn.execute("""
        INSERT INTO predictions
        (Pregnancies, Glucose, BloodPressure, SkinThickness,
         Insulin, BMI, DiabetesPedigreeFunction, Age,
         Predicted_Glucose, Diabetes_Outcome)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pregnancies, glucose, bp, skin, insulin,
        bmi, dpf, age, round(predicted_glucose, 2), predicted_outcome
    ))
    conn.commit()

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(["Predicted Glucose", "Diabetes Outcome"], [predicted_glucose, predicted_outcome])
    st.pyplot(fig)

# -----------------------------
# View Stored Predictions
# -----------------------------
st.subheader("📊 Stored Predictions")
df = pd.read_sql("SELECT * FROM predictions", conn)
st.dataframe(df)
