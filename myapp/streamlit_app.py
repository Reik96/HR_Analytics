import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
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
        conn = SQL("localhost","root",123456,"hr_analytics","aug_test")
        df = conn.query_data()
    
    elif data_action == "CSV": 
        upload_file = st.sidebar.file_uploader("Upload CSV file")
       
        if upload_file is not None:
            df = pd.read_csv(upload_file)

    return df

def data_storing(predictions):

       # conn = SQL("localhost","root",123456,"hr_analytics")
        conn = SQL("localhost","root",123456,"hr_analytics","predictions")
        conn.insert_data(predictions)

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
      
        
        df_conc = predict_values(scaled_X,df)
        st.subheader('Class Labels')
        st.write(pd.DataFrame.from_dict(data = {'Average Probability': [np.mean(df_conc["No - Probability"]),np.mean(df_conc["Yes - Probability"])]}))
        
        st.subheader('Data Scientists that are looking for a new job')
        df_conc = pd.concat([df_id,df_conc], axis=1)
        st.write(df_conc)
        
        st.subheader("Save Predictions")
        csv_downloader(df_conc)
        
        sql_button = st.button("Store Data in MySQL")

        if sql_button == True:
            
            data_storing(df_conc)
       

    if menu_action == "Visualizations":
        html_temp = """<div class='tableauPlaceholder' id='viz1616928180032' style='position: relative'><noscript><a href='#'><img alt='Blatt 4 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Te&#47;Test_Viz_16169281534280&#47;Blatt4&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Test_Viz_16169281534280&#47;Blatt4' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Te&#47;Test_Viz_16169281534280&#47;Blatt4&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='de' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1616928180032');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
        components.html(html_temp,width = 800, height = 600)
        html_temp = """<div class='tableauPlaceholder' id='viz1616928180032' style='position: relative'><noscript><a href='#'><img alt='Blatt 4 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Te&#47;Test_Viz_16169281534280&#47;Blatt4&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Test_Viz_16169281534280&#47;Blatt4' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Te&#47;Test_Viz_16169281534280&#47;Blatt4&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='de' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1616928180032');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
        components.html(html_temp,width = 800, height = 600)





if __name__ == '__main__':
	main()