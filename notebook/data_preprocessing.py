# Imports
import pandas as pd 
import numpy as np

# Seed number
seed = 42

# Read File
df = pd.read_csv(r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\data\raw\aug_train.csv")


# Create a new feature to indicate wheter an individual provided all information or not
df.loc[df.isnull().values.any(), 'all_information'] = 0
df.loc[df.notnull().values.any(), 'all_information'] = 1

# Drop id

df.drop(columns= "enrollee_id",inplace=True)

# Replace NANs
df["all_information"].fillna(0,inplace=True)
df.fillna("unknown",inplace=True)

# Clean Company Size Column
df.company_size.replace('50-99',"<100",inplace=True)
df.company_size.replace('10000+',">10000",inplace=True)
df.company_size.replace('5000-9999',"<10000",inplace=True)
df.company_size.replace('1000-4999',"<5000",inplace=True)
df.company_size.replace('10/49',"<50",inplace=True)
df.company_size.replace('100-500',"<501",inplace=True)
df.company_size.replace('500-999',"<1000",inplace=True)


# Clean Experience Column
df.drop(df.index[df["experience"] == "unknown"], inplace = True)
df.replace(">20","20",inplace=True)
df.replace("<1","0",inplace=True)
df["experience"] = df["experience"].astype(int)

# Clean last new job Column
df.last_new_job.replace(">4","4",inplace=True)
df.last_new_job.replace("never","0",inplace=True)
df.last_new_job.replace("unknown","0",inplace=True)
df.last_new_job = pd.to_numeric(df.last_new_job)

# Data split in Training and Test data
from preprocessing.data_split import Split


split = Split(df,target="target")
X_train,X_test,y_train,y_test=split.train_test()

# Transformation of columns
from col_transformer import ColTransformer

col_trans = ColTransformer(X_train,X_test)
scaled_X_train, scaled_X_test = col_trans.col_transformer()


