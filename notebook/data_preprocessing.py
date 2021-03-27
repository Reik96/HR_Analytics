# Imports
import pandas as pd 
import numpy as np

# Seed number
seed = 42

# Read File
df = pd.read_csv(r"C:\Users\rsele\OneDrive\Data Science\Projekt\HR_Prediction_App\data\raw\aug_test.csv")

#df = pd.read_csv(r"C:\Users\rsele\OneDrive\Data Science\Projekt\HR_Prediction_App\data\raw\aug_train.csv")

from src.preprocessing import cleaning

df = cleaning.data_cleaning(df)

 ## Call function to create new category for variables
print(df.isnull().sum())
# Data split in Training and Test data
from src.preprocessing.data_split import Split


#split = Split(df,target_name="target")
X_train,X_test,y_train,y_test=Split(df,"target",test_size = 0.2,shuffle = True).train_test()

# Transformation of columns
from src.preprocessing.col_transformer import ColTransformer

col_trans = ColTransformer(X_train,X_test)

scaled_X_train,scaled_X_test= col_trans.col_transformer()

