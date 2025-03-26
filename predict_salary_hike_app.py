import streamlit as st
import pandas as pd
import joblib

# Load the trained model and training columns list
model = joblib.load('salary_hike_model.pkl')
train_columns = joblib.load('train_columns.pkl')

st.title("Salary Hike Prediction App")
st.write("Enter the employee details to predict the Percent Salary Hike.")

# Numeric inputs
performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
job_level = st.number_input("Job Level", min_value=1, max_value=5, step=1)
years_in_current_role = st.number_input("Years In Current Role", min_value=0.0, step=1.0)
hourly_rate = st.number_input("Hourly Rate", min_value=0.0)
age = st.number_input("Age", min_value=18, max_value=80, step=1)

# Categorical inputs (raw values)
business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox("Job Role", [
    "Sales Representative", "Research Scientist", "Manufacturing Director",
    "Research Director", "Manager", "Laboratory Technician", "Sales Executive", "Human Resources"
])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
overtime = st.selectbox("OverTime", ["Yes", "No"])

# Create a DataFrame for the input data
input_dict = {
    "PerformanceRating": [performance_rating],
    "JobLevel": [job_level],
    "YearsInCurrentRole": [years_in_current_role],
    "HourlyRate": [hourly_rate],
    "Age": [age],
    "BusinessTravel": [business_travel],
    "Department": [department],
    "EducationField": [education_field],
    "Gender": [gender],
    "JobRole": [job_role],
    "MaritalStatus": [marital_status],
    "OverTime": [overtime]
}

new_data = pd.DataFrame(input_dict)

# Apply pd.get_dummies to encode the categorical columns (same as during training)
new_data_encoded = pd.get_dummies(new_data, columns=[
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"
], drop_first=False)

# Reindex the DataFrame to match the training columns, filling missing columns with 0
new_data_encoded = new_data_encoded.reindex(columns=train_columns, fill_value=0)

# Predict on the new input data when button is clicked
if st.button("Predict Salary Hike"):
    prediction = model.predict(new_data_encoded)[0]
    st.subheader(f"Predicted Percent Salary Hike: {prediction:.2f}%")
