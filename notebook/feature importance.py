
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.modelling.feature_names import get_feature_names

from data_preprocessing import X_train, scaled_X_train, y_train

#import col transformer and logistic regression
col_transformer = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\src\preprocessing\col_transformer.pkl", "rb" ))

model = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\model\log_reg_model.pkl", "rb" ))

# get feature names from transformed X-Values
scaled_X_names = get_feature_names(col_transformer)

from sklearn.feature_selection import RFE

#rekursive Feature Elimination
rfe = RFE(model,10)
rfe.fit(scaled_X_train, y_train)
df_feat = pd.DataFrame({"Ranking":rfe.ranking_,
                        "Feature":scaled_X_names
                        })
   
df_feat = df_feat.sort_values(by="Ranking",ascending=True)

print(df_feat[df_feat["Ranking"]<11])


