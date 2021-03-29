import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from src.preprocessing import cleaning
from src.predictions.sql_connection import SQL
import pickle 
import base64
import time
import gspread
from gspread_dataframe import set_with_dataframe
import df2gspread as d2g
from oauth2client.service_account import ServiceAccountCredentials


timestring = time.strftime("%Y%m%d")

model = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\model\log_reg_model.pkl", "rb" ))
col_transformer = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\src\preprocessing\col_transformer.pkl", "rb" ))


def data_loading(data_action):

    if data_action =="SQL":    
        conn = SQL("localhost","root",123456,"hr_analytics","aug_test")
        df = conn.query_data()
    
    elif data_action == "CSV": 
        upload_file = st.sidebar.file_uploader("Upload CSV file")
       
        if upload_file is not None:
            df = pd.read_csv(upload_file)

    return df

def data_storing_sql(predictions):

       # conn = SQL("localhost","root",123456,"hr_analytics")
        conn = SQL("localhost","root",123456,"hr_analytics","predictions")
        conn.insert_data(predictions)


def data_storing_gs(df_conc):

    #Loads df into Google Spreadsheets
    gc = gspread.service_account(filename=r'C:\Users\rsele\Downloads\hr-analytics-309111-dd74d95fb582.json') #-> Credential File needed
    sh = gc.open('HR_Analytics')
    worksheet = sh.get_worksheet(0) 
    set_with_dataframe(worksheet, df_conc) 
    worksheet.update([df_conc.columns.values.tolist()] + df_conc.values.tolist())

def predict_values(scaled_X,df):

    y_pred =model.predict(scaled_X)
    y_pred_proba = model.predict_proba(scaled_X)

    df = df.reset_index(drop=True)

    y_pred_proba_0 = pd.Series(y_pred_proba[:,0], name= "No - Probability")
    y_pred_proba_1 = pd.Series(y_pred_proba[:,1], name= "Yes - Probability")
    
    y_pred_s = pd.Series(y_pred, name= "Prediction")
    df_conc = pd.concat([df,y_pred_s,y_pred_proba_0,y_pred_proba_1], axis=1)

    return df_conc


def csv_downloader(df_conc):
    #https://github.com/Jcharis/Streamlit_DataScience_Apps/blob/master/File_Downloader_App/app.py

    csvfile = df_conc.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "HR_Predictions_{}_.csv".format(timestring)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">HR_Predictions.csv File</a>'
    st.markdown(href,unsafe_allow_html=True)



