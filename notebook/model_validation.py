
from notebook.data_preprocessing import scaled_X_train,scaled_X_test, y_train,y_test
from src.modelling.model_comparison import ModelComp

mc = ModelComp()

comp = mc.models(lr="lr",knn="knn",rf="rf",svm="svm")

dfs = mc.comparison(scaled_X_train, scaled_X_test, y_train,y_test)

print(dfs)