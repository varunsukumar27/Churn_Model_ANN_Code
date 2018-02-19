# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:48:54 2018

@author: varun
"""

#Data Preprocessing
#Importing the Dataset.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

churn_data = pd.read_csv('Churn_Modelling.csv')


#Extracting the independent variables from the dataset based on the analyis performed.
x = churn_data.iloc[:,3:13].values


#Extracting the target variable from the dataset.
y = churn_data.iloc[:,13].values


#Encoding the categorical variables in the independent variables dataset.
#Encoding country.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_country = LabelEncoder()
x[:,1] = label_encoder_country.fit_transform(x[:,1])

#Encoding Gender.
label_encoder_gender = LabelEncoder()
x[:,2] = label_encoder_gender.fit_transform(x[:,2])

hotencoder = OneHotEncoder(categorical_features = [1])
X = hotencoder.fit_transform(x).toarray()
X = X[:, 1:]


#Splitting data into testing and training set.
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)

#As we are going to use ANN it absolutely necesssary to perform Standardization or Normalization.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.fit_transform(Xtest)


#Import the keras library.
import keras
#To initialize the neural networks
from keras.models import Sequential
#To create the layers in the neural networks
from keras.layers import Dense
from keras.layers import Dropout

#Initializing the ANN
model = Sequential()

#Creating input layer, hiddenlayers and the output layers.
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
model.add(Dropout(p = 0.1))
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(p = 0.1))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#Compiling the neural network
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Fitting the ANN to the training set
model.fit(Xtrain, Ytrain, batch_size = 10, nb_epoch = 100)


#Predicting.
predictions = model.predict(Xtest)
predictions_10 = (predictions > 0.5)

new_pred = model.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000, 2,1,1,50000]])))
new_pred = (new_pred > 0.5)

#Usng a confusion matrix to evaluate the model.
from sklearn.metrics import confusion_matrix
cnfmx = confusion_matrix(Ytest, predictions_10)

#churn_data.loc[10000] = [10001, 12312312, 'Sukumar', 600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000, 1]


#Evaluating and tuning the model
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    model = Sequential()
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    model = Sequential()
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn = build_classifier)
param = {'batch_size':[25,32],
         'nb_epoch':[100, 500],
         'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = model,
                           param_grid = param,
                           scoring = 'accuracy',
                           cv = 10)
grid_search.fit(Xtrain,Ytrain)
best_params = grid_search.best_params_
best_acc = grid_search.best_score_
accuracies = cross_val_score(estimator = model, X = Xtrain, y = Ytrain, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
