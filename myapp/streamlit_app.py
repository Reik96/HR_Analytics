import streamlit as st
import pandas as pd
from src.preprocessing import cleaning
from src.predictions.sql_connection import SQL
import pickle 
import base64
import time

timestring = time.strftime("%Y%m%d")

model = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\model\log_reg_model.pkl", "rb" ))
col_transformer = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\src\preprocessing\col_transformer.pkl", "rb" ))


def data_loading(data_action):

    if data_action =="SQL":    
        from src.predictions.sql_connection import SQL
        conn = SQL("localhost","root",123456,"hr_analytics","aug_test")
        df = conn.query_data()
    
    elif data_action == "CSV": 
        upload_file = st.sidebar.file_uploader("Upload CSV file")
       
        if upload_file is not None:
            df = pd.read_csv(upload_file)

    return df


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
    st.markdown("#### Download Predictions###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">HR_Predictions.csv File</a>'
    st.markdown(href,unsafe_allow_html=True)




   # def plot_values():


def main():

    st.write("""
    # Simple HR Prediction App!""")
    st.sidebar.header('Menu')
    
    data_action = st.sidebar.selectbox(
        'Data Source',
        ("SQL","CSV")
    )

    df = data_loading(data_action)
    df_id = df["enrollee_id"]
    df = cleaning.data_cleaning(df)
    scaled_X = col_transformer.transform(df)

    menu_action = st.sidebar.selectbox(
        'Data Prediction & Analytics',
        ("Predictions","Visualizations")
    )

    if menu_action == "Predictions":
        st.subheader('Class Labels')
        st.write(pd.DataFrame.from_dict(data = {'Label': ["No","Yes"]}))

        st.subheader('Data Scientists that are looking for a new job')
        df_conc = predict_values(scaled_X,df)
        st.write(df_conc)
        csv_downloader(df_conc)


    if menu_action == "Visualizations":
        st.vega_lite_chart(df_conc, {
            'mark': {'type': 'circle', 'tooltip': True},
            'encoding': {
                'x': {'field': 'training_hours', 'type': 'quantitative'},
                'y': {'field': 'city_development_index', 'type': 'quantitative'},
            'size': {'field': 'Prediction', 'type': 'quantitative'},
                'color': {'field': 'Prediction', 'type': 'quantitative'},
            },
        })

if __name__ == '__main__':
	main()