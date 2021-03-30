import time
tic=time.clock() # Timer starten
import pandas
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

import numpy as np
from data_preprocessing import scaled_X_train,scaled_X_test,y_test,y_train
from keras.callbacks import EarlyStopping
#import h5py
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold, cross_validate
from keras.optimizers import Adam,SGD
#from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV

#Zufallszahl
SEED=42

#Foldanzahl
kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=SEED)

#Eingabevektor
X_col = scaled_X_train[-1]

#Function zur Erstellung eines neuronalen Netzes
def create_model(learning_rate, activation, n_neurons,hidden_layer,drop):

    # Eingabevektor
    X_col = scaled_X_train.shape[-1]
    #Adam-Optimizer mit Lernrate
    opt = Adam(lr = learning_rate)
    #Abbruchkriterium, wenn Validierungsverlust über 3 Epochen nicht kleiner wird
    early_stopping_monitor = [EarlyStopping(monitor="val_loss",patience=3)]
   
    # Modell
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_col, activation= activation))
    
    #Für Ermittlung der geeigneten Anzahl an Hidden Layer
    for i in range(int(hidden_layer)):
        model.add(Dense(n_neurons, activation=activation))
        model.add(Dropout(drop))
    
    model.add(Dense(n_neurons, input_dim=X_col, activation= activation))

    #Output Layer mit Sigmoidfunktion - Ermittlung der Klassenwahrscheinlichkeit
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    
    return model
  

# Import KerasClassifier aus keras scikit learn wrappers
from keras.wrappers.scikit_learn import KerasClassifier
from skopt.space import Real,Categorical,Integer

# KerasClassifier
model = KerasClassifier(build_fn = create_model)

# Suchraum
# params = {"activation":Categorical(["relu","sigmoid"]), "batch_size":Integer(128,512), 
#           "epochs":Integer(50,100), "n_neurons":Integer(16,64),"learning_rate":Real(0.001, 0.1),
#           "hidden_layer":Integer(2,4),"drop":Real(0.1,0.5)}

params = {"activation":["relu","sigmoid"], "batch_size":[32,64,128,512], 
          "epochs":[10,25,50,100], "n_neurons":[4,8,16,32,64],"learning_rate":[0.001, 0.01,0.1],
          "hidden_layer":[2,3,4],"drop":[0.1,0.2,0.3]}

# Bayes´sche Optimierung mit Gauss-Prozess
# Die erzeugte Function und der angegebene Suchraum bewirken, dass verschiedene Netzwerkarchitekturen berücksichtigt werden
bayes_search=RandomizedSearchCV(model, params,cv=kfold,scoring="accuracy",random_state=SEED, njobs = -1)

clf_opt=bayes_search.fit(scaled_X_train,y_train)

# Optimale Parametereinstellungen
print(clf_opt.best_params_)
print(clf_opt.best_estimator_)


#save model
#clf_opt.best_estimator_.save(my_model)

#Klassifikationsreport
y_pred=clf_opt.best_estimator_.predict(scaled_X_test.toarray())
y_pred =(y_pred>0.5)
print(classification_report(y_test,y_pred))
toc=time.clock() # Timer beenden

print(toc-tic)