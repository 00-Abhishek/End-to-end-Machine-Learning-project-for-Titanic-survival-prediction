import streamlit as st
import pickle
import pandas as pd

st.title("🚢 Titanic Survival Predictor")

# Load pipeline
model = pickle.load(open('models/titanic_pipeline.pkl', 'rb'))

# Inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ['male', 'female'])
Age = st.number_input("Age", 0.0, 100.0, 30.0)
SibSp = st.number_input("Siblings/Spouses", 0, 10, 0)
Parch = st.number_input("Parents/Children", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])

# Create input DataFrame (IMPORTANT)
input_df = pd.DataFrame([{
    'Pclass': Pclass,
    'Sex': Sex,
    'Age': Age,
    'SibSp': SibSp,
    'Parch': Parch,
    'Fare': Fare,
    'Embarked': Embarked
}])

# Predict
prediction = model.predict(input_df)[0]

if prediction == 1:
    st.success("Passenger Survived")
else:
    st.error("Passenger Did Not Survive")

# Probability (safe)
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(input_df)[0][1]
    st.info(f"Survival Probability: {prob*100:.2f}%")