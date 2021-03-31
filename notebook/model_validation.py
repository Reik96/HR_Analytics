
from src.modelling.model_comparison import ModelComp

from notebook.data_preprocessing import (scaled_X_test, scaled_X_train, y_test,
                                         y_train)

mc = ModelComp()


comp = mc.models(lr="lr",rf="rf",xgboost="xgboost")
clf = mc.comparison(scaled_X_train, y_train)

print(clf)
