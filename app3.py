import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

st.title("Student Mark Prediction")

# User Inputs
duration_of_study = st.number_input("Enter hours of study", min_value=0, max_value=24, value=0)
class_attendence = st.number_input("Enter attendance", min_value=60, max_value=100)
Access_to_Resources = st.selectbox("Resources", ['high', 'medium', 'low'])
motivation_level = st.selectbox("Motivation", ['high', 'medium', 'low'])


motivation_mapping = {'high': 2, 'medium': 1, 'low': 0}
resource_mapping = {'high': 2, 'medium': 1, 'low': 0}


motvtn_lvl_numeric = motivation_mapping[motivation_level]
resource_lvl_numeric = resource_mapping[Access_to_Resources]


input_data = {
    "Hours_Studied": duration_of_study,
    "Attendance": class_attendence,
    "Access_to_Resources_m": resource_lvl_numeric,
    "Motivation_Level_m": motvtn_lvl_numeric
}


new_data = pd.DataFrame([input_data])


df = pd.read_csv("students.csv")
columns_list = [col for col in df.columns if col != 'Unnamed: 0']


new_data = new_data.reindex(columns=columns_list, fill_value=0)


with open("LinearRegression_model.pkl", "rb") as regression_file:
    loaded_model = pickle.load(regression_file)


prediction = loaded_model.predict(new_data)


if prediction[0] <= 60:
    st.error("Prediction: You have failed.")
else:
    st.success("Prediction: You have passed.")
