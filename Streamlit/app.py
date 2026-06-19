import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import streamlit as st
import warnings

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Get the current directory
current_dir = os.getcwd()
print(f"Directorio actual: {current_dir}")

# Specify the name of the CSV file
file_name = "diabetes_prediction_dataset.csv"  # Change this to your real file name

# Build the full path of the file
file_path = os.path.join(current_dir, file_name)

# Read the CSV file
df = pd.read_csv(file_path)

# data pre-processing
enc=OrdinalEncoder()
df["smoking_history"]=enc.fit_transform(df[["smoking_history"]])
df["gender"]=enc.fit_transform(df[["gender"]])


# Split data into dependent and independent variables
x= df.drop("diabetes",axis=1)
y=df["diabetes"]

# Split data into training and test
x_train , x_test , y_train, y_test = train_test_split(x,y,test_size=0.3)

# train the model
model = RandomForestClassifier().fit(x_train,y_train) # Random Forest Classifier
y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test,y_pred)

# Show model accuracy
st.set_page_config(page_title='Diabetes Prediction', page_icon=':dna:')
st.markdown(f'<h1 style="text-align: center;">Diabetes Prediction</h1>', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap='large')

# We structured the interface in two columns for better visual organization with streamlit
with col1:
    gender = st.selectbox(label='Gender', options=['Male', 'Female', 'Other'])
    gender_dict = {'Female':0.0, 'Male':1.0, 'Other':2.0}

    age = st.text_input(label='Age')

    hypertension = st.selectbox(label='Hypertension', options=['No', 'Yes'])
    hypertension_dict = {'No':0, 'Yes':1}

    heart_disease = st.selectbox(label='Heart Disease', options=['No', 'Yes'])
    heart_disease_dict = {'No':0, 'Yes':1}

with col2:
    smoking_history = st.selectbox(label='Smoking History',
                                   options=['Never', 'Current', 'Former', 'Ever', 'Not Current', 'No Info'])
    smoking_history_dict = {'Never':4.0, 'No Info':0.0, 'Current':1.0,
                            'Former':3.0, 'Ever':2.0, 'Not Current':5.0}

    bmi = st.text_input(label='BMI')

    hba1c_level = st.text_input(label='HbA1c Level')

    blood_glucose_level = st.text_input(label='Blood Glucose Level')

st.write('')
st.write('')
col1,col2 = st.columns([0.438,0.562])
with col2:
    submit = st.button(label='Submit')
st.write('')

# make the prediction
if submit:
    try:
        user_data = np.array( [[ gender_dict[gender], age, hypertension_dict[hypertension], heart_disease_dict[heart_disease],
                                smoking_history_dict[smoking_history], bmi, hba1c_level, blood_glucose_level ]] )

        test_result = model.predict(user_data)

        # Show the result of the prediction
        if test_result[0] == 0:
            col1,col2,col3 = st.columns([0.33,0.30,0.35])
            with col2:
                st.success('Diabetes Result: Negative')
            st.balloons()

        else:
            col1,col2,col3 = st.columns([0.215,0.57,0.215])
            with col2:
                st.error('Diabetes Result: Positive (Please Consult with Doctor)')

    except:
        st.warning('Please fill the all required informations')
