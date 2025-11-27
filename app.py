
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Model and Columns
model = joblib.load('student_risk_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Page Configuration
st.set_page_config(page_title="Student Risk Detector", page_icon="🎓")

# Title and Description
st.title('🎓 Student Risk Detection System')
st.write('Enter student details below to predict their academic outcome.')

# Sidebar for Inputs
st.sidebar.header('Student Profile')

# Input 1: Tuition Fees
tuition = st.sidebar.selectbox('Are tuition fees up to date?', 
                               options=[1, 0], 
                               format_func=lambda x: 'Yes (Paid)' if x == 1 else 'No (Overdue)')

# Input 2: Debtor Status
debtor = st.sidebar.selectbox('Is the student a debtor?', 
                              options=[1, 0], 
                              format_func=lambda x: 'Yes (Has Debt)' if x == 1 else 'No (Debt Free)')

# Input 3: Approved Units (2nd Semester)
approved_units = st.sidebar.slider('Units Approved (2nd Sem)', 0, 20, 5)

# Input 4: Grade Average (2nd Semester)
grade = st.sidebar.slider('Grade Average (2nd Sem)', 0.0, 20.0, 10.0)

# Prepare Data for Prediction
# Create a dataframe with zeros for all columns the model expects
input_data = pd.DataFrame(0, index=[0], columns=model_columns)

# Fill in the user inputs
input_data['Tuition fees up to date'] = tuition
input_data['Debtor'] = debtor
input_data['Curricular units 2nd sem (approved)'] = approved_units
input_data['Curricular units 2nd sem (grade)'] = grade
# Note: Other features are kept at 0 or default for this demo.

# Predict Button
if st.button('🔍 Predict Outcome'):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    st.write('---')
    st.subheader('Prediction Results:')
    
    # Logic for displaying results (0=Dropout, 1=Enrolled, 2=Graduate)
    if prediction == 0:
        st.error(f'🚨 Prediction: DROPOUT (High Risk)')
        st.write(f'Probability of Dropout: {proba[0]*100:.1f}%')
        st.info("Recommendation: Immediate intervention required. Financial counseling suggested.")
        
    elif prediction == 1:
        st.warning(f'⚠️ Prediction: ENROLLED (At Risk/Struggling)')
        st.write(f'Probability of staying Enrolled: {proba[1]*100:.1f}%')
        st.info("Recommendation: Academic tutoring support suggested.")
        
    else:
        st.success(f'✅ Prediction: GRADUATE (On Track)')
        st.write(f'Probability of Graduation: {proba[2]*100:.1f}%')
