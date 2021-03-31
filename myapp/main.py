import numpy as np
import pandas as pd

from streamlit_app import *


def main():

    st.write("""# Simple HR Prediction App!""")
    st.subheader("Find out which Data Scientists are looking for a new Job Opportunity")
    st.sidebar.header('Menu')
    
    data_action = st.sidebar.selectbox(
        'Data Source',
        ("All SQL Data","Latest SQL Data","CSV File")
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
        
        st.subheader('Data Frame with Features and corresponding Predictions')
        df_conc = pd.concat([df_id,df_conc], axis=1)
        st.write(df_conc)
        
        st.subheader("Save Predictions")
        csv_downloader(df_conc)

        
        sql_button = st.button("Append Data in MySQL")

        if sql_button == True:
            st.write("Data stored in MySQL")
            data_storing_sql(df_conc)
        
        gs_clear_button = st.button("Replace Data in Google Sheet")
        if gs_clear_button == True:
         
            try:
                st.write("Replaced Values in Google Sheet")
                data_storing_gs(df_conc,True)
            except Exception:
                pass

        gs_button = st.button("Append Data in Google Sheet")
        
        if gs_button == True:
        
            try:
                st.write("Updated Google Sheet")
                data_storing_gs(df_conc,False)
            except Exception:
                pass
        

    
            
       

    if menu_action == "Visualizations":
        st.subheader('Tableau Public Dashboard')
        st.write("Based on the  Google Sheet, the Dashboard refreshes daily or by requesting an update on Tableau Public ")
        html_temp = """<div class='tableauPlaceholder' id='viz1617009991413' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HR&#47;HR_Analytics_16170086783530&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='HR_Analytics_16170086783530&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HR&#47;HR_Analytics_16170086783530&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='de' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1617009991413');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
        components.html(html_temp,width = 1960, height = 1080)
    
       

if __name__ == '__main__':
	main()
