import streamlit as st
import pandas as pd
from src.preprocessing import cleaning
#from notebook.data_preprocessing import df
import pickle 


model = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\model\log_reg_model.pkl", "rb" ))
col_transformer = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\src\preprocessing\col_transformer.pkl", "rb" ))
# import col_transformer.pkl
# import model.pkl

st.write("""
# Simple HR Prediction App!
""")


st.sidebar.header('Data Set')

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ("All","Jan","Feb","Mar")
)


from src.predictions.sql_connection import SQL
c = SQL("localhost","root",123456,"hr_analytics","aug_test")
data = c.query_data()

data = cleaning.data_cleaning(data)
#print(data[data['experience'].isnull()])
#data.fillna("unknown",inplace=True)
print(data.isnull().sum())
print(data[data.experience == "unknown"])

#st.write(data.info())

scaled_X = col_transformer.transform(data)
#X = data.drop(columns="target",inplace=True)

#data -> coltrans

y_pred=model.predict(scaled_X)
y_pred_proba= model.predict_proba(scaled_X)

#st.subheader('Class labels and their corresponding index number')


st.subheader('Looking for a new Job')
st.write(y_pred)

st.subheader('Prediction Probability')
st.write(y_pred_proba)
