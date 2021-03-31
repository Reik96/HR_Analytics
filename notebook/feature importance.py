
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.modelling.feature_names import get_feature_names

from data_preprocessing import scaled_X_train, y_train

#import col transformer and logistic regression
col_transformer = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekte\HR_Analytics\src\preprocessing\col_transformer.pkl", "rb" ))

model = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekte\HR_Analytics\model\log_reg_model.pkl", "rb" ))

# get feature names from transformed X-Values
scaled_X_names = get_feature_names(col_transformer)

from sklearn.feature_selection import RFE

rfe = RFE(model, 20)
rfe.fit(scaled_X_train, y_train)
coefs= np.transpose(rfe.estimator_.coef_)
feat = [feature for feature, rank in zip(scaled_X_names, rfe.ranking_) if rank==1]

print(feat)

df_feat = pd.DataFrame(data=coefs,columns=["coefficients"])
df_feat["Feature"]=feat
df_feat = df_feat.sort_values(by="coefficients", ascending=False)

# Diagramm der wichtigsten Features
plt.barh(df_feat["Feature"],df_feat["coefficients"])
plt.title("Relevant Features")
plt.ylabel("Features")
plt.xlabel("Coefficient")
plt.show()