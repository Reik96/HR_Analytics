
from notebook.data_preprocessing import scaled_X_train,scaled_X_test,y_train,y_test, seed,X_train
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skopt.space import Real,Categorical,Integer
from skopt import BayesSearchCV
from skopt.space import Integer, Real

#Zufallszahl
SEED=42
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
#clf der logistischen Regression zuweisen
clf = lr
#Suchraum für Bayes`sche Optimierung mit Gauss-Prozess
param = {"solver":['newton-cg', 'lbfgs', 'liblinear','sag','saga'],
        "C":[0.0001,0.001,0.01,1,10,100],
        "max_iter":[100,1000,1000,5000]}



#Bayes´sche Optimierung
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
random_search=GridSearchCV(clf, param,cv=5,n_jobs=-1,scoring="f1")

clf_opt=random_search.fit(scaled_X_train,y_train)

# Beste Parametereinstellungen
print(clf_opt.best_params_)
print(clf_opt.best_estimator_)


# Modell trainieren und Vorhersagekraft an Testdaten testen
lr.fit(scaled_X_train,y_train)
y_pred= clf_opt.predict(scaled_X_test)

# Konfusionsmatrix
conf_matrix= confusion_matrix(y_test, y_pred,labels=[0,1])
f, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(conf_matrix, annot=True, 
            fmt="d", linewidths=.5, cmap=plt.cm.Reds)

plt.title("Konfusionsmatrix optimierte logisische Regression", fontsize=20)
ax.set_xticklabels(["Ermittelte Klasse 0",
                    "Ermittelte Klasse 1",
                   ])
ax.set_yticklabels(["Tatsächliche Klasse 0",
                    "Tatsächliche Klasse 1",
            
                    ], rotation=360)
plt.show()


import sklearn.metrics as metrics

#AUC und ROC
probs = clf_opt.predict_proba(scaled_X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

print("{:s}: {:.2f}".format("AUC optimierte logistische Regression: ",roc_auc))


      
plt.title('ROC optimierte logistische Regression')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print("optimierte logistische Regression: \n",classification_report(y_test,y_pred,digits=4))