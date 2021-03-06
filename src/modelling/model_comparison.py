
class ModelComp:
    
    """Choice of algorithms and validation based on accuracy,
        training time and inference time"""

    def __init__(self,seed=42):
        self.seed = seed 
    
    def models(self, lr=None,knn=None,rf=None,svm=None, xgboost=None):

        # Classifier
        self.lr = lr
        self.rf = rf
        self.knn = knn 
        self.svm = svm 
        self.xgboost = xgboost
        
        if self.lr =="lr":
       
            from sklearn.linear_model import LogisticRegression
            self.lr = LogisticRegression(random_state=self.seed, max_iter=5000)
        
        if self.rf =="rf":
            from sklearn.ensemble import RandomForestClassifier
            self.rf=RandomForestClassifier(random_state=self.seed)

        if self.knn == "knn":
            from sklearn.neighbors import KNeighborsClassifier as KNN
            self.knn = KNN()

        if self.svm == "svm":
            from sklearn.svm import SVC
            self.svm = SVC(random_state=self.seed,probability=True)
        
        if self.xgboost== "xgboost":
            from xgboost import XGBClassifier
            self.xgboost = XGBClassifier(random_state=self.seed)

        return [("Logistische Regression",lr),
                 ("KNN",knn),
              ("Random Forest",rf),
              ("SVM",svm),
              ("XGBOOST",xgboost)
              ]
   
    def comparison(self,scaled_X_train,y_train,clf=None,splits=5):
       
        #Compare the given classifiers based on validation score`s accuracy

        from sklearn.model_selection import cross_validate
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from imblearn.over_sampling import SMOTE

        kfold = StratifiedKFold(n_splits=splits,shuffle=True, random_state=self.seed)
        classifier = self.models(self.lr,self.knn,self.rf,self.svm,self.xgboost)

        cv_scores = []
        cv_score_time =[]
        cv_fit_time = []
        df_clf = []

        
        for clf_name,clf in classifier:
    
            if clf is not None:
                scaled_X_train,y_train = SMOTE(random_state=self.seed).fit_resample(scaled_X_train,y_train)
                score = cross_validate(clf,scaled_X_train,y_train,cv=kfold,scoring="accuracy")
                cv_scores.append(score["test_score"].mean()*100) 
                cv_score_time.append(score["score_time"].mean()) 
                cv_fit_time.append(score["fit_time"].mean()) 
                df_clf.append(clf_name)

        # Dataframe with results
        df_scores = pd.DataFrame(cv_scores,columns=["Test Accuracy"])
        df_scortime = pd.DataFrame(cv_score_time,columns=["Inference time in seconds"])
        df_fit = pd.DataFrame(cv_fit_time,columns=["Training time in seconds"])
        df_clf = pd.DataFrame(df_clf,columns=["Classifier"])
        df_clf_scores= pd.concat([df_clf, df_scores,df_scortime,df_fit], axis=1)
        df_clf_scores = df_clf_scores.sort_values(by=["Test Accuracy"], ascending=False)
        return df_clf_scores



if __name__ == "__main__":
   ModelComp()