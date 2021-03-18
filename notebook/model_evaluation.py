#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
            
from data_preprocessing import scaled_X_train,scaled_X_test,y_train,y_test, seed
from sklearn.metrics import classification_report

#svm = SVC(random_state=seed,probability=True)

lr = LogisticRegression(random_state=seed, max_iter=5000)
lr.fit(scaled_X_train,y_train)

y_pred = svm.predict(scaled_X_test)

print(classification_report(y_test,y_pred))



def mysql(user, password, host, db):

    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine("mysql+pymysql://{0}:{1}@{2}/{3}".format(user, password, host, db))
    connection = engine.connect()
    df = pd.to_sql(select, engine)
    return df

#from src.predictions.sql_connection import mysql

predictions = mysql("root",123456,"localhost","mydb")

predictions.to_sql(name="mydb")