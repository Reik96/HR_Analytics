from model_evaluation import lr
import pandas as pd 
import numpy as np

# Seed number
seed = 42

# Read File
#df = pd.read_csv(r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\data\raw\aug_train.csv")

#df = pd.read_csv(r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\data\raw\aug_train.csv")

#Data Cleaning
from src.preprocessing import cleaning

df = cleaning.data_cleaning(df)

# Data split in Training and Test data
from src.preprocessing.data_split import Split
X_train,X_test,y_train,y_test = Split(df,"target").train_test()


# Transformation of columns
from src.preprocessing.col_transformer import ColTransformer
y=df["target"]
df.drop("target", inplace=True)
X = df

X = ColTransformer.col_tramsformer.transform(df)

y_pred = lr.predict(X,y)