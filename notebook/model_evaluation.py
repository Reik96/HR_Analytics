#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
            
from data_preprocessing import scaled_X_train,scaled_X_test,y_train,y_test, seed,X_train
from imblearn.over_sampling import SMOTE

#lr.fit(scaled_X_train,y_train)
sm_X_train,sm_y_train = SMOTE(random_state=42).fit_resample(scaled_X_train,y_train)
#scaled_X_train,y_train = RandomUnderSampler(random_state=42).fit_resample(scaled_X_train,y_train)
lr = LogisticRegression(max_iter=1000)
#lr = XGBClassifier(random_state=42)
lr.fit(sm_X_train,sm_y_train)
y_pred = lr.predict(scaled_X_test)
y_pred_proba = lr.predict_proba(scaled_X_test)

#Save model
import pickle
#pickle.dump(lr,open("log_reg_model.pkl","wb"))

from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,accuracy_score,classification_report
print(f"Roc-Auc score: {roc_auc_score(y_test,y_pred)},f1_score: {f1_score(y_test,y_pred)},Accuracy: {accuracy_score(y_test,y_pred)}")
print(classification_report(y_test,y_pred))
print(y_pred)
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix= confusion_matrix(y_test, y_pred,labels=[0,1])
f, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(conf_matrix, annot=True, 
            fmt="d", linewidths=.5, cmap=plt.cm.Reds)

plt.title("Confusion Matrix LR", fontsize=20)
ax.set_xticklabels(["Predicted 0",
                    "Predicted 1",
                  ])
ax.set_yticklabels(["Actual 0",
                     "Actual 1",
                    ], rotation=360)
plt.show()


