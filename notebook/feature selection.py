#from model_evaluation import lr
import pandas as pd 
import numpy as np 
from src.modelling.feature_importance import get_feature_names
from data_preprocessing import scaled_X_train,y_train
import pickle 
import matplotlib.pyplot as plt

col_transformer = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\src\preprocessing\col_transformer.pkl", "rb" ))

model = pickle.load( open( r"C:\Users\rsele\OneDrive\Data Science\Projekt\ML_with_SQL_Tableau\model\log_reg_model.pkl", "rb" ))


scaled_X_names = get_feature_names(col_transformer)

from sklearn.feature_selection import RFE

#rekursive Feature Elimination, begrenzt auf die 20 relevantesten Features
rfe = RFE(model, 20)
rfe.fit(scaled_X_train, y_train)
coefs= np.transpose(rfe.estimator_.coef_)
feat = [feature for feature, rank in zip(scaled_X_names, rfe.ranking_) if rank==1]

print(feat)

df_feat = pd.DataFrame(data=coefs,columns=["coefficients"])
df_feat["Feature"]=feat

#df_feat.to_excel("Feature_Importance_Wrapper.xlsx")


# Diagramm der wichtigsten Features
plt.barh(df_feat["Feature"],df_feat["coefficients"])
plt.title("RelevantFeatures")
plt.ylabel("Features")
plt.xlabel("Coefficient")
plt.show()