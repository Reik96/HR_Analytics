#imports
import seaborn
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from data_preprocessing import (X_train, scaled_X_test, scaled_X_train, seed,
                                y_test, y_train)

#Oversampling with SMOTE
sm_X_train,sm_y_train = SMOTE(random_state=42).fit_resample(scaled_X_train,y_train)

#Training and Predicting
lr = LogisticRegression(C=10, max_iter=10000, solver='saga')

lr.fit(sm_X_train,sm_y_train)
y_pred = lr.predict(scaled_X_test)
y_pred_proba = lr.predict_proba(scaled_X_test)

#Save model
import pickle

# Plot results
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

#pickle.dump(lr,open("log_reg_model.pkl","wb"))

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


