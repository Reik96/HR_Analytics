import streamlit as st
import pandas as pd
from src.preprocessing import cleaning
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
    ("Training","Test")
)

#if dataset_name == "Test":

from src.predictions.sql_connection import SQL
c = SQL("localhost","root",123456,"hr_analytics","aug_test")
df = c.query_data()

df = cleaning.data_cleaning(df)



scaled_X = col_transformer.transform(df)

y_pred=model.predict(scaled_X)
y_pred_proba= model.predict_proba(scaled_X)

print(y_pred)
st.subheader('Class Labels')
st.write(pd.DataFrame.from_dict(data = {'Label': ["No","Yes"]}))

st.subheader('Data Scientists that are looking for a new job')

st.write(y_pred)

st.subheader('Prediction Probability')
st.write(y_pred_proba)
