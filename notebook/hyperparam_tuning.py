import pandas
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score
import numpy as np
from data_preprocessing import scaled_X_train,scaled_X_test,y_test,y_train
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
params = {'C': [0.001,0.01,0.1,1,10,100], 
          'solver': ["newton-cg","lbfgs","sag","saga"],
          'max_iter':[1000,2000,5000,10000]}


smote = SMOTE(random_state=42)          
imba_pipeline=make_pipeline(smote,lr)
new_params = {'logisticregression__' + key: params[key] for key in params}
cv_clf = RandomizedSearchCV(imba_pipeline,new_params, cv=5,scoring=roc_auc_score)
lr_opt=cv_clf.fit(scaled_X_train,y_train)
print(lr_opt.best_estimator_)
y_pred = lr_opt.predict(scaled_X_test)
print(classification_report(y_test,y_pred))