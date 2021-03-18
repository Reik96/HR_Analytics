class ColTransformer:

    def __init__(self,X_train,X_test):

        self.X_train = X_train
        self.X_test = X_test

    def col_transformer(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        # Distinguish columns into numerical and categorical
        numerical_col = self.X_train.select_dtypes(include=["int64","float64"]).columns
        categorical_col = self.X_train.select_dtypes(include=["object","bool"]).columns
        
        # Column transformation
        col_transformer = ColumnTransformer(
                            transformers=[
                                ("scaler", StandardScaler(), 
                                numerical_col
                                ),
                            ("ohe",OneHotEncoder(handle_unknown="ignore"),
                                categorical_col
                                )
                            ],remainder="drop",
                            n_jobs=-1
                            )

        scaled_X_train = col_transformer.fit_transform(self.X_train).toarray()
        scaled_X_test = col_transformer.transform(self.X_test).toarray()
        return scaled_X_train, scaled_X_test
