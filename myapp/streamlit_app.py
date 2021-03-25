import streamlit as st
import pandas as pd
from src.preprocessing import cleaning
import pickle 


model = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\model\log_reg_model.pkl", "rb" ))
col_transformer = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\src\preprocessing\col_transformer.pkl", "rb" ))


st.write("""
# Simple HR Prediction App!
""")


st.sidebar.header('Data Set')

menu_action = st.sidebar.selectbox(
    'Menu',
    ("Predict","Visualize","Save")
)




from src.predictions.sql_connection import SQL
c = SQL("localhost","root",123456,"hr_analytics","aug_test")
df = c.query_data()



df_id = df["enrollee_id"]
df = cleaning.data_cleaning(df)


scaled_X = col_transformer.transform(df)

y_pred=model.predict(scaled_X)

y_pred_proba= model.predict_proba(scaled_X)



if menu_action == "Predict":

    st.subheader('Class Labels')
    st.write(pd.DataFrame.from_dict(data = {'Label': ["No","Yes"]}))

    st.subheader('Data Scientists that are looking for a new job')

    y_pred_proba_0 = pd.Series(y_pred_proba[:,0], name= "No - Probability")
    y_pred_proba_1 = pd.Series(y_pred_proba[:,1], name= "Yes - Probability")
    y_pred = pd.Series(y_pred, name= "Prediction")

    st.write(pd.concat([df_id, y_pred,y_pred_proba_0,y_pred_proba_1], axis=1))


if menu_action == "Visualize":

    st.bar_chart(y_pred)