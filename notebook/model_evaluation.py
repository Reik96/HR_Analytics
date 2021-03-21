#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
            
from data_preprocessing import scaled_X_train,scaled_X_test,y_train,y_test, seed
from sklearn.metrics import classification_report

#svm = SVC(random_state=seed,probability=True)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

#lr.fit(scaled_X_train,y_train)
scaled_X_train,y_train = SMOTE(random_state=42).fit_resample(scaled_X_train,y_train)
lr = LogisticRegression(random_state=42, max_iter=5000)
lr.fit(scaled_X_train,y_train)
y_pred = lr.predict(scaled_X_test)

print(classification_report(y_test,y_pred))
