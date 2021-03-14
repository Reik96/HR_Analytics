
class ModelComp:

    def __init__(self,seed=42):
        self.seed = seed 
    
    def models(self):
       # from imblearn.over_sampling import SMOTE,RandomOverSampler
        # Classifier

       
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(random_state=self.seed, max_iter=5000)

    
        from sklearn.ensemble import RandomForestClassifier
        rf=RandomForestClassifier(random_state=self.seed)

    
        from sklearn.neighbors import KNeighborsClassifier as KNN
        knn = KNN()

    
        from sklearn.naive_bayes import GaussianNB as GNB
        nb = GNB()

    
        from sklearn.svm import SVC
        svm = SVC(random_state=self.seed,probability=True)

        return [("Logistische Regression",lr),
              ("KNN",knn),
              ("Random Forest",rf),
              ("Naiver Bayes",nb),
              ("SVM",svm)
              ]
   
    def comparison(self,scaled_X_train,scaled_X_test,y_train,y_test,clf=None, splits=5):
        from sklearn.model_selection import cross_validate
        import pandas as pd
        #from src.preprocessing import scaled_X_train,scaled_X_test,y_test,y_train,seed
        from sklearn.model_selection import StratifiedKFold, cross_validate
       
        kfold = StratifiedKFold(n_splits=splits,shuffle=True, random_state=self.seed)
        classifier = self.models()

        cv_scores = []
        cv_score_time =[]
        cv_fit_time = []
        df_clf = []

        for clf_name,clf in classifier:
    
            if clf is not None:
                score = cross_validate(clf,scaled_X_train,y_train,cv=kfold)
                cv_scores.append(score["test_score"].mean()*100) # durchschnittliches Validierungsergebnis in Prozent
                cv_score_time.append(score["score_time"].mean()) # durchschnittliche Inferenzzeit in Sekunden
                cv_fit_time.append(score["fit_time"].mean()) # durchschnittliche Trainingszeit in Sekunden
                df_clf.append(clf_name)

        # Dataframe mit Ergebnissen
        df_scores = pd.DataFrame(cv_scores,columns=["Testgenauigkeit"])
        df_scortime = pd.DataFrame(cv_score_time,columns=["Vorhersagedauer in Sekunden"])
        df_fit = pd.DataFrame(cv_fit_time,columns=["Trainingsdauer in Sekunden"])
        df_clf = pd.DataFrame(df_clf,columns=["Klassifikator"])
        df_clf_scores= pd.concat([df_clf, df_scores,df_scortime,df_fit], axis=1)
        df_clf_scores = df_clf_scores.sort_values(by=["Testgenauigkeit"], ascending=False)
        return df_clf_scores

