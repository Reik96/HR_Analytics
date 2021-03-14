class Split:

    """ Class to determine the dataframe and name of the target column
    to split it into a X and y Variable and training and test set."""

    def __init__(self, df,target_name,test_size=0.3,seed=42):
    
        self.df = df 
        self.y = target_name
        self.test_size = test_size 
        self.seed = seed

    def train_test(self):
        
        from sklearn.model_selection import train_test_split
        y = self.df[self.y].astype("category")
        X = self.df
        X.drop(columns=self.y,inplace=True)
        # Split in train and test set
        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y, test_size=self.test_size,random_state=self.seed)
        return X_train,X_test,y_train,y_test