import streamlit as st
import pandas as pd
from src.preprocessing import cleaning
import pickle 
import base64

model = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\model\log_reg_model.pkl", "rb" ))
col_transformer = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\src\preprocessing\col_transformer.pkl", "rb" ))


class Data():
    
    def __init__(self,data):
    
        self.data = data 

    def transform(self):

        df_id = self.data["enrollee_id"]
        df = cleaning.data_cleaning(self.data)

        scaled_X = col_transformer.transform(df)
        
        return scaled_X, df, df_id

    def predict(self):

        y_pred,_,_=model.predict(self.transform())

        y_pred_proba, _ , _= model.predict_proba(self.transform())

        df = self.data.reset_index(drop=True)

        y_pred_proba_0 = pd.Series(y_pred_proba[:,0], name= "No - Probability")
        y_pred_proba_1 = pd.Series(y_pred_proba[:,1], name= "Yes - Probability")


        y_pred_s = pd.Series(y_pred, name= "Prediction")

        df_conc = pd.concat([df_id, df,y_pred_s,y_pred_proba_0,y_pred_proba_1], axis=1)

        return df_conc


class FileDownloader():

    def __init__(self, data):
		
        self.data = data
	
    def csv_downloader(self):

        csvfile = self.data.to_csv()
        b64 = base64.b64encode(csvfile.encode()).decode()
        new_filename = "new_text_file_{}_.csv".format(timestr)
        st.markdown("#### Download File ###")
        href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
        st.markdown(href,unsafe_allow_html=True)



def main():

    st.write("""
    # Simple HR Prediction App!
    """)
    st.sidebar.header('Menu')

    data_action = st.sidebar.selectbox(
        'Data Source',
        ("SQL","CSV")
    )

    if data_action =="SQL":
        
        from src.predictions.sql_connection import SQL
        conn = SQL("localhost","root",123456,"hr_analytics","aug_test")
        df = conn.query_data()

    if data_action == "CSV":

        upload_file = st.sidebar.file_uploader("Upload CSV file")
        
        if upload_file is not None:

            df = pd.read_csv(upload_file)


    menu_action = st.sidebar.selectbox(
        'Data Prediction & Analytics',
        ("Predictions","Visualizations")
    )


    if menu_action == "Predictions":

        st.subheader('Class Labels')
        st.write(pd.DataFrame.from_dict(data = {'Label': ["No","Yes"]}))

        st.subheader('Data Scientists that are looking for a new job')

        data = Data(df)
        df = data.transform()
        df_conc = data.predict()
        
        st.write(df_conc)

        

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