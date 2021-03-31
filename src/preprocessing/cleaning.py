
def data_cleaning(df):
    import pandas as pd
    import numpy as np

    # Drop id
    df.drop(columns= ["enrollee_id"],inplace=True)
    # Create a new feature to indicate wheter an individual provided all information or not
    df.loc[df.isnull().values.any(), 'all_information'] = 0
    df.loc[df.notnull().values.any(), 'all_information'] = 1
    df["all_information"]=df["all_information"].fillna(0)

    ## Replace NAN and empty fields with unknown     
    df["gender"].fillna("unknown",inplace=True)
    df["gender"].replace("","unknown",inplace=True)
    df["major_discipline"].fillna("unknown",inplace=True)
    df["major_discipline"].replace("","unknown",inplace=True)
    df["company_type"].fillna("unknown",inplace=True)
    df["company_type"].replace("","unknown",inplace=True)
    df["education_level"].fillna("unknown",inplace=True)
    df["education_level"].replace("","unknown",inplace=True)

    # Order education
    education = {"unknown":0,"Primary School":1,"High School":2, "Graduate":3,"Masters":4,"Phd":5}
    df["education_level"].map(education)
   

    # Order company size
    company_size={"unknown":0, "<10":1,"10/49":2,  "50-99":3,"100-500":4,"500-999":5,"1000-4999":6,"5000-9999":7,  '10000+':8}
    df["company_size"]=df["company_size"].map(company_size)

    # Order enrolled_university
    enrolled_university={"unknown":0, "no_enrollment":1, "Part time course":2, "Full time course":3}
    df["enrolled_university"]=df["enrolled_university"].map(enrolled_university)

    # Order relevemt experience
    relevent_experience={"unknown":0,"No relevent experience":0, "Has relevent experience":1}
    df['relevent_experience']=df['relevent_experience'].map(relevent_experience)
    
    # Clean Experience Column
    df.experience.replace(">20","20",inplace=True)
    df.experience.replace("<1","0",inplace=True)
    df.experience.replace("unknown","0",inplace=True)
    df.experience = pd.to_numeric(df.experience)

    # Clean last new job Column
    df.last_new_job.replace(">4","4",inplace=True)
    df.last_new_job.replace("never","0",inplace=True)
    df.last_new_job.replace("unknown","0",inplace=True)
    df.last_new_job = pd.to_numeric(df.last_new_job)

    # Fill NAN in numerical columns with the mean
    df.fillna(df.mean().round(0), inplace=True)

    return df

if __name__ == "__main__":
    data_cleaning(df)